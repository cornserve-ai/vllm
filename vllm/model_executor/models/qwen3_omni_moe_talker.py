"""Inference-only Qwen3-Omni-Moe model (talker part)."""

from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
    Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration,
)

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.qwen2_moe import Qwen2MoeSparseMoeBlock
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from .interfaces import (
    MultiModalEmbeddings,
    PastHiddenStatesProcessing,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
)
from .qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
)
from .qwen3_omni_moe_thinker import (
    Qwen3OmniMoeConditionalGenerationMixin,
    Qwen3OmniMoeThinkerDummyInputsBuilder,
    Qwen3OmniMoeThinkerMultiModalProcessor,
    Qwen3OmniMoeThinkerProcessingInfo,
    _get_feat_extract_output_lengths,
)
from .utils import (
    AutoWeightsLoader,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from .utils import AutoWeightsLoader, WeightsMapper, maybe_prefix
from .vision import get_llm_pos_ids_for_vision

try:
    import flash_attn
except (ImportError, ModuleNotFoundError):
    flash_attn = None

logger = init_logger(__name__)

class Qwen3OmniMoeTalkerResizeMLP(nn.Module):
    """ResizeMLP to project from thinker hidden size to talker hidden size.

    Uses regular nn.Linear instead of parallel layers to match HF numerics exactly.
    This is important because even small numerical differences here compound through
    the entire forward pass.
    """
    def __init__(
        self,
        thinker_hidden_size: int,
        text_intermediate_size: int,
        text_hidden_size: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(
            thinker_hidden_size,
            text_intermediate_size,
            bias=True,
        )
        self.act_fn = nn.SiLU()
        self.linear_fc2 = nn.Linear(
            text_intermediate_size,
            text_hidden_size,
            bias=True,
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3MoeTalkerDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super(Qwen3MoeDecoderLayer, self).__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.self_attn = Qwen3MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.mlp = Qwen2MoeSparseMoeBlock(
            config=vllm_config.model_config.hf_config,  # type: ignore
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3OmniMoeTalkerModel(Qwen3MoeModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3MoeModel, self).__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen3MoeTalkerDecoderLayer(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        # Track layers for auxiliary hidden state outputs (EAGLE3)
        self.aux_hidden_state_layers: tuple[int, ...] = ()

class Qwen3MoeTalkerLLMForCausalLM(Qwen3MoeForCausalLM):
    """Talker LLM

    This is very similar to thinker.
    """
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3MoeForCausalLM, self).__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        # codec_embedding now becomes embed_tokens
        self.model = Qwen3OmniMoeTalkerModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class Qwen3OmniMoeTalkerForConditionalGeneration(
    nn.Module,
    SupportsPP,
    SupportsMRoPE,
    SupportsMultiModal,
    PastHiddenStatesProcessing,
    Qwen3OmniMoeConditionalGenerationMixin,
):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "thinker.model.embed_tokens": "thinker_text_embed",
            "talker.model.codec_embedding": "language_model.model.embed_tokens",
            "talker.model.": "language_model.model.",
            "talker.code_predictor.": "code_predictor.",
            "talker.text_projection.": "text_projection.",
            "talker.hidden_projection.": "hidden_projection.",
            "talker.codec_head.": "codec_head.",
            "talker.": "",
        }
    )
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        hf_config = vllm_config.model_config.hf_config
        talker_config: Qwen3OmniMoeTalkerConfig = hf_config.talker_config

        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = talker_config
        self.thinker_config = vllm_config.model_config.hf_config.thinker_config
        self.multimodal_config = multimodal_config

        talker_vllm_config = vllm_config.with_hf_config(
            talker_config.text_config, architectures=["Qwen3MoeForCausalLM"]
        )
        talker_vllm_config = talker_vllm_config.with_hf_text_config(talker_config.text_config)

        # we use exact mapping from HF
        self.language_model = Qwen3MoeTalkerLLMForCausalLM(
            vllm_config=talker_vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        # Add thinker's text embedding for standalone talker mode
        # This allows the talker to process text tokens directly
        # (tts_pad=151671, tts_bos=151672, tts_eos=151673, tokenizer max=151675)
        self.thinker_text_embed = VocabParallelEmbedding(
            num_embeddings=self.thinker_config.text_config.vocab_size,
            embedding_dim=self.thinker_config.text_config.hidden_size,
            quant_config=quant_config,
        )

        self.text_projection = Qwen3OmniMoeTalkerResizeMLP(
            thinker_hidden_size=talker_config.thinker_hidden_size,
            text_intermediate_size=talker_config.text_config.intermediate_size,
            text_hidden_size=talker_config.text_config.hidden_size,
            quant_config=quant_config,
            reduce_results=True,
            prefix=maybe_prefix(prefix, "text_projection"),
        )
        self.hidden_projection = Qwen3OmniMoeTalkerResizeMLP(
            thinker_hidden_size=talker_config.thinker_hidden_size,
            text_intermediate_size=talker_config.text_config.intermediate_size,
            text_hidden_size=talker_config.text_config.hidden_size,
            quant_config=quant_config,
            reduce_results=True,
            prefix=maybe_prefix(prefix, "hidden_projection"),
        )
        self.codec_head = nn.Linear(
            talker_config.text_config.hidden_size,
            talker_config.text_config.vocab_size,
            bias=False,
        )

        # Initialize code_predictor using HuggingFace transformers directly
        self.code_predictor = Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration._from_config(
            config=talker_config.code_predictor_config
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.model.make_empty_intermediate_tensors
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def prepare_inputs_from_past_hidden_states(
        self,
        trailing_text_hidden_states:torch.Tensor,
        input_ids: list[int],
        past_hidden_states: torch.Tensor,
        generation_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare inputs for TTS generation during decode using code_predictor for a single request.

        This implements the logic from HF's prepare_inputs_for_generation.

        Args:
            trailing_text_hidden_states: Hidden states from thinker's output
            should be [1, seq_len, hidden_size].
            input_ids: Input token IDs (only last token is used)
            past_hidden_states: Hidden states from previous forward pass
            generation_step: Current generation step for trailing text indexing

        Returns:
            Dictionary containing:
                - inputs_embeds: Aggregated embeddings for next forward pass
                - residual_codes: Generated codec token IDs (input_ids + generated sequences)
        """
        if trailing_text_hidden_states.dim() == 2:
            # [seq_len, hidden_size] -> [1, seq_len, hidden_size]
            trailing_text_hidden_states = trailing_text_hidden_states.unsqueeze(0)

        # the new input is now thinker_embeds
        trailing_text_hidden_states = self.text_projection(trailing_text_hidden_states)

        # Extract last token from input_ids
        # input_ids is list[int] in vLLM, need to convert to tensor
        last_token_id = input_ids[-1]
        # Create tensor directly on the correct device (get device from model weights)
        device = next(self.thinker_text_embed.parameters()).device
        last_token_ids = torch.tensor([[last_token_id]], dtype=torch.long, device=device)  # [1, 1]

        # Get embedding for last token
        # During decode, input_ids contains sampled codec tokens (0-3071) from compute_logits
        # Use talker's codec embedding layer, matching HF: self.get_input_embeddings()(input_ids)
        last_id_hidden = self.language_model.model.embed_tokens(last_token_ids)  # [1, 1, talker_hidden_size]

        # Extract past_hidden: last token from past_hidden_states
        # past_hidden_states may be on CPU (cached by model runner), move to GPU
        past_hidden_states = past_hidden_states.to(device)

        # past_hidden_states shape: [seq_len, hidden_size] in vLLM
        if past_hidden_states.dim() == 2:
            # [seq_len, hidden_size] -> [1, 1, hidden_size]
            past_hidden = past_hidden_states[-1:, :].unsqueeze(0)
        else:
            # [batch_size, seq_len, hidden_size] -> [batch_size, 1, hidden_size]
            past_hidden = past_hidden_states[:, -1:, :]

        # Concatenate for code_predictor input
        predictor_input = torch.cat((past_hidden, last_id_hidden), dim=1)  # [1, 2, hidden_size]

        # Generate codec tokens using code_predictor.generate()
        max_new_tokens = self.config.code_predictor_config.num_code_groups - 1  # 15

        predictor_result = self.code_predictor.generate(
            inputs_embeds=predictor_input,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.8,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        # Create residual_codes: input_ids + generated sequences
        residual_codes = torch.cat((last_token_ids, predictor_result.sequences.to(device)), dim=-1)  # [1, 16]

        # Extract hidden states from generation
        # We want the last layer output from each intermediate step
        mid_residual_hiddens = [hid[0].to(device) for hid in predictor_result.hidden_states[1:]]  # Skip first step

        # Get embedding for last generated token
        last_residual_hidden = self.code_predictor.get_input_embeddings()[-1](
            predictor_result.sequences[..., -1:]
        ).to(device)

        # Aggregate embeddings
        # Concatenate: [last_id_hidden] + mid_hiddens + [last_residual_hidden]
        codec_hiddens = torch.cat(
            [last_id_hidden] + mid_residual_hiddens + [last_residual_hidden],
            dim=1,
        )  # [1, 17, hidden_size] (1 initial + 15 generated + 1 final)

        inputs_embeds = codec_hiddens.sum(1, keepdim=True)  # [1, 1, hidden_size]

        # Add trailing_text_hidden or tts_pad_embed
        # trailing_text_hidden_states should be 3D: [1(bs), seq_len, hidden_size]
        if generation_step < trailing_text_hidden_states.shape[1]:
            # Ensure trailing_text_hidden_states is on the correct device
            trailing_text_hidden_states = trailing_text_hidden_states.to(device)
            # Index the specific generation_step and unsqueeze to match HF implementation
            inputs_embeds = inputs_embeds + trailing_text_hidden_states[:, generation_step].unsqueeze(1).to(device)
        else:
            # tts_pad_embed: constant zero padding
            tts_pad_token_id = 151671
            tts_pad_thinker_embed = self.thinker_text_embed(torch.tensor([[tts_pad_token_id]], device=device))  # [1, 1, thinker_hidden_size]
            tts_pad_embed = self.text_projection(tts_pad_thinker_embed)  # [1, 1, talker_hidden_size]
            inputs_embeds = inputs_embeds + tts_pad_embed

        # Squeeze to match expected shape [1, hidden_size]
        inputs_embeds = inputs_embeds.squeeze(0)  # [1, hidden_size]

        return inputs_embeds, residual_codes

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
        thinker_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Apply token embeddings to input_ids.

        Thinker => Talker only needs thinker 24'th layer's hidden states for MM data,
        the rest are all through test_projection after thinker text embedding.
        """
        assert is_multimodal is not None, "is_multimodal tensor must be provided"
        # Embed using thinker's text vocabulary bc tokenzied through thinker
        thinker_embeds = self.thinker_text_embed(input_ids)
        talker_embeds = torch.empty(
            thinker_embeds.shape[0],
            self.config.text_config.hidden_size,
            device=thinker_embeds.device,
            dtype=thinker_embeds.dtype  # Match dtype to avoid mismatch
        )
        device = talker_embeds.device

        if is_multimodal is not None and is_multimodal.any().item():
            num_mm_tokens = is_multimodal.sum().item()
            thinker_hidden = thinker_hidden_states.to(device)
            thinker_hidden = thinker_hidden[:is_multimodal.shape[0], :]
            thinker_hidden_mm = thinker_hidden[is_multimodal]
            # use thinker hidden for mm tokens
            projected_hidden_mm = self.hidden_projection(thinker_hidden_mm)
            talker_embeds[is_multimodal] = projected_hidden_mm

        # for non-mm tokens, directly use thinker embeds
        text_thinker_embeds = thinker_embeds[~is_multimodal]
        projected_text = self.text_projection(text_thinker_embeds)
        talker_embeds[~is_multimodal] = projected_text

        # Add codec control embeddings for the 9-token assistant section
        # This matches HF's _get_talker_assistant_parts
        # Structure: [3 control tokens][4 tts_pad][1 tts_bos][1 first_text]
        # the last token is \n, which will be replaced by the first thinker's output text token
        # prompt trailing pattern = [151644, 77091, 198, 151671, 151645, 198, 151644, 77091, 198]
        replace_pattern = [151644, 77091, 198, 151671, 151645, 198, 151644, 77091]
        replace_indices = find_pattern_in_tensor(input_ids, replace_pattern)
        for idx in replace_indices:
            talker_embeds = self._replace_codec_control_embeddings_at_index(talker_embeds, idx)
        return talker_embeds

    def _replace_codec_control_embeddings_at_index(
        self,
        talker_embeddings: torch.Tensor,
        index: int,
    ) -> torch.Tensor:
        """
        Replace 9 token embeddings starting at index with codec control embeddings.

        This is a helper function that applies the same logic as _add_codec_control_embeddings
        but for a specific location in the embedding tensor, enabling batch processing.

        Args:
            talker_embeddings: The full embeddings tensor [num_tokens, hidden_size]
            index: Starting index where to replace the 9 tokens

        Returns:
            Modified embeddings tensor with replacements at the specified index
        """
        device = talker_embeddings.device
        dtype = talker_embeddings.dtype
        hidden_size = talker_embeddings.shape[-1]

        # Extract the first 3 tokens' embeddings at the pattern location
        first_3_text_embeds = talker_embeddings[index:index+3].clone()  # [3, hidden_size]

        # Create TTS control embeddings
        tts_bos_token_id = 151672
        tts_eos_token_id = 151673
        tts_pad_token_id = 151671

        talker_special_tokens = torch.tensor(
            [[tts_bos_token_id, tts_eos_token_id, tts_pad_token_id]],
            device=device,
            dtype=torch.long,
        )  # [1, 3]

        # Embed and project in batch
        tts_embeds_thinker = self.thinker_text_embed(talker_special_tokens)  # [1, 3, thinker_hidden_size]
        tts_embeds_projected = self.text_projection(tts_embeds_thinker)  # [1, 3, talker_hidden_size]

        # Chunk along dim=1
        tts_bos_embed, tts_eos_embed, tts_pad_embed = tts_embeds_projected.chunk(3, dim=1)

        # Reshape and expand
        tts_bos_embed = tts_bos_embed.squeeze(1)  # [1, talker_hidden_size]
        tts_pad_embed = tts_pad_embed.squeeze(1)  # [1, talker_hidden_size]
        tts_pad_embed = tts_pad_embed.expand(4, -1)  # [4, talker_hidden_size]

        # For the last position, we need the to use the first text from thinker's output
        # For batching, we'll we the embedding of index+8:+9, relying on the last token id being replaced by the first thinker's output token already
        last_text_embed = talker_embeddings[index+8:index+9]  # [1, hidden_size]

        # Build assistant_text_hidden: [3 first tokens][4 tts_pad][1 tts_bos][1 last token]
        assistant_text_hidden = torch.cat(
            [first_3_text_embeds, tts_pad_embed, tts_bos_embed, last_text_embed],
            dim=0,
        )  # [9, hidden_size]

        # Create codec_special_tokens
        speaker_id_map = getattr(self.config, 'speaker_id', {})
        speaker_id = speaker_id_map.get('ethan', 2302)  # Default to Ethan
        codec_special_tokens = torch.tensor(
            [
                self.config.codec_nothink_id,      # Position 3
                self.config.codec_think_bos_id,    # Position 4
                self.config.codec_think_eos_id,    # Position 5
                speaker_id,                         # Position 6
                self.config.codec_pad_id,          # Position 7
                self.config.codec_bos_id,          # Position 8
            ],
            device=device,
            dtype=torch.long,
        )

        # Get codec embeddings
        codec_embeds = self.language_model.model.embed_tokens(codec_special_tokens)  # [6, hidden_size]

        # Build assistant_codec_hidden: [3 zeros][6 codec embeddings]
        zeros_for_control = torch.zeros(3, hidden_size, device=device, dtype=dtype)
        assistant_codec_hidden = torch.cat([zeros_for_control, codec_embeds], dim=0)  # [9, hidden_size]

        # Final embeddings
        final_assistant_embeds = assistant_text_hidden + assistant_codec_hidden  # [9, hidden_size]

        # Replace the 9 token embeddings at the specified index
        talker_embeddings = talker_embeddings.clone()
        talker_embeddings[index:index+9] = final_assistant_embeds

        logger.debug(f"Replaced codec control embeddings at index {index}")
        return talker_embeddings

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor | None:
        """Compute logits for talker model.

        Returns logits over talker's codec vocabulary (vocab_size=3072).
        This matches HF implementation where codec_head projects to talker's vocab space.
        The EOS token (codec_eos_token_id=2150) is in this talker vocab space.
        """
        # Project hidden states to codec vocabulary logits using codec_head
        return self.codec_head(hidden_states)  # [batch_size, vocab_size=3072]

    @staticmethod
    def _squeeze_batch_dims(tensor: torch.Tensor, target_ndim: int) -> torch.Tensor:
        """Remove leading batch dimensions until target ndim is reached."""
        while tensor.ndim > target_ndim and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        return tensor

    def _normalize_mm_kwargs(self, kwargs: dict) -> dict:
        """Normalize all multimodal tensor shapes for upstream validation."""
        kwargs = dict(kwargs)  # Make mutable copy

        # Image: pixel_values [1,np,cps]->[np,cps], image_grid_thw [1,1,3]->[1,3]
        if "pixel_values" in kwargs and isinstance(kwargs["pixel_values"], torch.Tensor):
            kwargs["pixel_values"] = self._squeeze_batch_dims(kwargs["pixel_values"], 2)
        if "image_grid_thw" in kwargs and isinstance(kwargs["image_grid_thw"], torch.Tensor):
            ig = self._squeeze_batch_dims(kwargs["image_grid_thw"], 2)
            kwargs["image_grid_thw"] = ig.unsqueeze(0) if ig.ndim == 1 else ig

        # Video: similar pattern
        if "pixel_values_videos" in kwargs and isinstance(kwargs["pixel_values_videos"], torch.Tensor):
            kwargs["pixel_values_videos"] = self._squeeze_batch_dims(kwargs["pixel_values_videos"], 2)
        if "video_grid_thw" in kwargs and isinstance(kwargs["video_grid_thw"], torch.Tensor):
            vg = self._squeeze_batch_dims(kwargs["video_grid_thw"], 2)
            kwargs["video_grid_thw"] = vg.unsqueeze(0) if vg.ndim == 1 else vg

        # Audio: squeeze features and mask, flatten lengths
        for key in ("input_audio_features", "input_features"):
            if key in kwargs and isinstance(kwargs[key], torch.Tensor):
                kwargs[key] = self._squeeze_batch_dims(kwargs[key], 2)
        if "feature_attention_mask" in kwargs and isinstance(kwargs["feature_attention_mask"], torch.Tensor):
            fam = self._squeeze_batch_dims(kwargs["feature_attention_mask"], 2)
            kwargs["feature_attention_mask"] = fam.unsqueeze(0) if fam.ndim == 1 else fam
        if "audio_feature_lengths" in kwargs and isinstance(kwargs["audio_feature_lengths"], torch.Tensor):
            afl = kwargs["audio_feature_lengths"]
            kwargs["audio_feature_lengths"] = afl.reshape(-1) if afl.ndim > 0 else afl.reshape(1)

        return kwargs

    def get_multimodal_embeddings(
        self, **kwargs: object
    ) -> MultiModalEmbeddings | None:
        """Return empty embeddings with correct shapes matching thinker's output.

        This dummy implementation doesn't require self.visual or self.audio_tower.
        """
        kwargs = self._normalize_mm_kwargs(kwargs)

        # Parse and validate each modality (parent class has individual methods, not wrapper)
        mm_input_by_modality = {}
        for input_key in kwargs:
            if input_key in ("pixel_values", "image_embeds") and "image" not in mm_input_by_modality:
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(**kwargs)
            if input_key in ("pixel_values_videos", "video_embeds") and "video" not in mm_input_by_modality:
                mm_input_by_modality["video"] = self._parse_and_validate_video_input(**kwargs)
            if input_key in ("input_audio_features",) and "audio" not in mm_input_by_modality:
                mm_input_by_modality["audio"] = self._parse_and_validate_audio_input(**kwargs)

        if not mm_input_by_modality:
            return []

        # Constants from thinker config
        VISION_EMBED_DIM = self.thinker_config.text_config.hidden_size  # 2048
        AUDIO_EMBED_DIM = self.thinker_config.text_config.hidden_size  # 2048
        SPATIAL_MERGE_SIZE = self.thinker_config.vision_config.spatial_merge_size  # 2

        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        for modality in mm_input_by_modality:
            mm_input = mm_input_by_modality[modality]
            if modality == "image":
                if mm_input["type"] == "image_embeds":
                    # Return empty tensor(s) with same shape as input embeds
                    image_embeds = mm_input["image_embeds"]
                    multimodal_embeddings += (
                        (torch.empty_like(image_embeds),)
                        if isinstance(image_embeds, torch.Tensor)
                        else tuple(torch.empty_like(t) for t in image_embeds)
                    )
                else:
                    # Compute sizes from grid and create empty tensors
                    image_grid_thw = mm_input["image_grid_thw"]
                    pixel_values = mm_input["pixel_values"]
                    sizes = (image_grid_thw.prod(-1) // SPATIAL_MERGE_SIZE ** 2).tolist()
                    multimodal_embeddings += tuple(
                        torch.empty(sz, VISION_EMBED_DIM, dtype=pixel_values.dtype, device=pixel_values.device)
                        for sz in sizes
                    )
            elif modality == "video":
                if mm_input["type"] == "video_embeds":
                    video_embeds = mm_input["video_embeds"]
                    multimodal_embeddings += (
                        (torch.empty_like(video_embeds),)
                        if isinstance(video_embeds, torch.Tensor)
                        else tuple(torch.empty_like(t) for t in video_embeds)
                    )
                else:
                    video_grid_thw = mm_input["video_grid_thw"]
                    pixel_values_videos = mm_input["pixel_values_videos"]
                    sizes = (video_grid_thw.prod(-1) // SPATIAL_MERGE_SIZE ** 2).tolist()
                    multimodal_embeddings += tuple(
                        torch.empty(sz, VISION_EMBED_DIM, dtype=pixel_values_videos.dtype, device=pixel_values_videos.device)
                        for sz in sizes
                    )
            elif modality == "audio":
                # Compute audio output lengths using helper function
                audio_feature_lengths = mm_input["audio_feature_lengths"]
                _, audio_output_lengths = _get_feat_extract_output_lengths(audio_feature_lengths)
                input_features = mm_input["input_features"]
                multimodal_embeddings += tuple(
                    torch.empty(n, AUDIO_EMBED_DIM, dtype=input_features.dtype, device=input_features.device)
                    for n in audio_output_lengths.tolist()
                )

        return multimodal_embeddings

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["thinker.", "code2wav."],
        )
        loaded_weights = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        return loaded_weights

    @classmethod
    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        hf_config: PretrainedConfig,
        image_grid_thw: list[list[int]] | torch.Tensor | None,
        video_grid_thw: list[list[int]] | torch.Tensor | None,
        second_per_grid_ts: list[float] | None = None,
        context_len: int = 0,
        seq_len: int | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
        use_audio_in_video: bool = False,
    ) -> tuple[torch.Tensor, int]:
        """Changed to use self.thinker_config from thinker's implementation."""
        config = hf_config.thinker_config
        if isinstance(image_grid_thw, list):
            image_grid_thw = torch.tensor(image_grid_thw)
        if isinstance(video_grid_thw, list):
            video_grid_thw = torch.tensor(video_grid_thw)
        input_ids = torch.tensor(input_tokens)
        if input_ids is None or input_ids.ndim != 1:
            raise ValueError("_omni3_get_input_positions_tensor expects 1D input_ids")

        seq_len = input_ids.shape[0]
        if audio_feature_lengths is not None and not isinstance(
            audio_feature_lengths, torch.Tensor
        ):
            audio_feature_lengths = torch.as_tensor(
                audio_feature_lengths, dtype=torch.long
            )
        if second_per_grid_ts is None:
            if video_grid_thw is not None and video_grid_thw.numel() > 0:
                second_per_grids = torch.ones(
                    video_grid_thw.shape[0], dtype=torch.float32
                )
            else:
                second_per_grids = torch.tensor([], dtype=torch.float32)
        else:
            second_per_grids = torch.tensor(second_per_grid_ts, dtype=torch.float32)

        spatial_merge_size = config.vision_config.spatial_merge_size
        image_token_id = config.image_token_id
        video_token_id = config.video_token_id
        audio_token_id = config.audio_token_id
        vision_start_token_id = config.vision_start_token_id
        audio_start_token_id = config.audio_start_token_id
        position_id_per_seconds = config.position_id_per_seconds

        vision_start_indices = torch.argwhere(
            input_ids == vision_start_token_id
        ).squeeze(1)
        if vision_start_indices.numel() > 0:
            vision_tokens = input_ids[vision_start_indices + 1]
        else:
            vision_tokens = input_ids.new_empty((0,), dtype=input_ids.dtype)
        audio_nums = torch.sum(input_ids == audio_start_token_id)
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (
            (vision_tokens == audio_start_token_id).sum()
            if use_audio_in_video
            else (vision_tokens == video_token_id).sum()
        )

        llm_pos_ids_list: list[torch.Tensor] = []
        st = 0
        image_idx = 0
        video_idx = 0
        audio_idx = 0
        remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums  # noqa: E501
        multimodal_nums = (
            image_nums + audio_nums
            if use_audio_in_video
            else image_nums + video_nums + audio_nums
        )  # noqa: E501

        for _ in range(multimodal_nums):
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            if (image_token_id in input_tokens or video_token_id in input_tokens) and (
                remain_videos > 0 or remain_images > 0
            ):
                ed_vision_start = input_tokens.index(vision_start_token_id, st)
            else:
                ed_vision_start = len(input_tokens) + 1
            if audio_token_id in input_tokens and remain_audios > 0:
                ed_audio_start = input_tokens.index(audio_start_token_id, st)
            else:
                ed_audio_start = len(input_tokens) + 1
            min_ed = min(ed_vision_start, ed_audio_start)

            if min_ed == ed_audio_start:
                text_len = min_ed - st
                if text_len != 0:
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len, dtype=torch.long)
                        .view(1, -1)
                        .expand(3, -1)
                        + st_idx
                    )
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                bos_len = 1
                llm_pos_ids_list.append(
                    torch.arange(bos_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                _, audio_len = _get_feat_extract_output_lengths(
                    audio_feature_lengths[audio_idx]
                )
                llm_pos_ids = (
                    torch.arange(audio_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                llm_pos_ids_list.append(llm_pos_ids)
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                eos_len = 1
                llm_pos_ids_list.append(
                    torch.arange(eos_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                st += text_len + bos_len + audio_len + eos_len
                audio_idx += 1
                remain_audios -= 1
            elif (
                min_ed == ed_vision_start
                and input_ids[ed_vision_start + 1] == image_token_id
            ):
                text_len = min_ed - st
                if text_len != 0:
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len, dtype=torch.long)
                        .view(1, -1)
                        .expand(3, -1)
                        + st_idx
                    )
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                bos_len = 1
                llm_pos_ids_list.append(
                    torch.arange(bos_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                grid_t = image_grid_thw[image_idx][0]
                grid_hs = image_grid_thw[:, 1]
                grid_ws = image_grid_thw[:, 2]
                t_index = torch.arange(grid_t) * position_id_per_seconds
                llm_pos_ids = get_llm_pos_ids_for_vision(
                    st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                )
                image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                llm_pos_ids_list.append(llm_pos_ids)
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                eos_len = 1
                llm_pos_ids_list.append(
                    torch.arange(eos_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                st += text_len + bos_len + image_len + eos_len
                image_idx += 1
                remain_images -= 1
            elif (
                min_ed == ed_vision_start
                and input_ids[ed_vision_start + 1] == video_token_id
                and not use_audio_in_video
            ):
                text_len = min_ed - st
                if text_len != 0:
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len, dtype=torch.long)
                        .view(1, -1)
                        .expand(3, -1)
                        + st_idx
                    )
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                bos_len = 1
                llm_pos_ids_list.append(
                    torch.arange(bos_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                grid_t = video_grid_thw[video_idx][0]
                grid_hs = video_grid_thw[:, 1]
                grid_ws = video_grid_thw[:, 2]
                t_index = (
                    torch.arange(grid_t)
                    * float(second_per_grids[video_idx].item())
                    * position_id_per_seconds
                )
                llm_pos_ids = get_llm_pos_ids_for_vision(
                    st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                )
                video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                llm_pos_ids_list.append(llm_pos_ids)
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                eos_len = 1
                llm_pos_ids_list.append(
                    torch.arange(eos_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                st += text_len + bos_len + video_len + eos_len
                video_idx += 1
                remain_videos -= 1
            elif (
                min_ed == ed_vision_start
                and ed_vision_start + 1 == ed_audio_start
                and use_audio_in_video
            ):
                text_len = min_ed - st
                if text_len != 0:
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len, dtype=torch.long)
                        .view(1, -1)
                        .expand(3, -1)
                        + st_idx
                    )
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                bos_len = 1
                bos_block = (
                    torch.arange(bos_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                llm_pos_ids_list.append(bos_block)
                llm_pos_ids_list.append(bos_block)
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                _, audio_len = _get_feat_extract_output_lengths(
                    audio_feature_lengths[audio_idx]
                )
                audio_llm_pos_ids = (
                    torch.arange(audio_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                grid_t = video_grid_thw[video_idx][0]
                grid_hs = video_grid_thw[:, 1]
                grid_ws = video_grid_thw[:, 2]
                t_index = (
                    torch.arange(grid_t)
                    * float(second_per_grids[video_idx].item())
                    * position_id_per_seconds
                )
                video_llm_pos_ids = get_llm_pos_ids_for_vision(
                    st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                )
                video_data_index, audio_data_index = 0, 0
                while (
                    video_data_index < video_llm_pos_ids.shape[-1]
                    and audio_data_index < audio_llm_pos_ids.shape[-1]
                ):
                    if (
                        video_llm_pos_ids[0][video_data_index]
                        <= audio_llm_pos_ids[0][audio_data_index]
                    ):
                        llm_pos_ids_list.append(
                            video_llm_pos_ids[
                                :, video_data_index : video_data_index + 1
                            ]
                        )
                        video_data_index += 1
                    else:
                        llm_pos_ids_list.append(
                            audio_llm_pos_ids[
                                :, audio_data_index : audio_data_index + 1
                            ]
                        )
                        audio_data_index += 1
                if video_data_index < video_llm_pos_ids.shape[-1]:
                    llm_pos_ids_list.append(
                        video_llm_pos_ids[
                            :, video_data_index : video_llm_pos_ids.shape[-1]
                        ]
                    )
                if audio_data_index < audio_llm_pos_ids.shape[-1]:
                    llm_pos_ids_list.append(
                        audio_llm_pos_ids[
                            :, audio_data_index : audio_llm_pos_ids.shape[-1]
                        ]
                    )
                video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                eos_len = 1
                eos_block = (
                    torch.arange(eos_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                llm_pos_ids_list.append(eos_block)
                llm_pos_ids_list.append(eos_block)
                st += text_len + bos_len * 2 + audio_len + video_len + eos_len * 2  # noqa: E501
                audio_idx += 1
                video_idx += 1
                remain_videos -= 1
                remain_audios -= 1

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len, dtype=torch.long).view(1, -1).expand(3, -1)
                + st_idx
            )

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        if llm_positions.shape[1] != seq_len:
            raise RuntimeError("Position ids length mismatch with input ids length")

        mrope_position_delta = llm_positions.max() + 1 - seq_len
        return llm_positions, mrope_position_delta

@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class Qwen3OmniMoeTalkerVocoderForConditionalGeneration(
    Qwen3OmniMoeTalkerForConditionalGeneration
):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "thinker.model.embed_tokens": "thinker_text_embed",
            "talker.model.codec_embedding": "language_model.model.embed_tokens",
            "talker.model.": "language_model.model.",
            "talker.code_predictor.": "code_predictor.",
            "talker.text_projection.": "text_projection.",
            "talker.hidden_projection.": "hidden_projection.",
            "talker.codec_head.": "codec_head.",
            "talker.": "",
        }
    )
    is_audio_generator = True
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        hf_config = vllm_config.model_config.hf_config
        self.code2wav = Qwen3OmniMoeCode2Wav._from_config(hf_config.code2wav_config)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["thinker."],
        )
        loaded_weights = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        return loaded_weights
    
    def generate_audio(
        self,
        residual_codes: list[torch.Tensor],
        *,
        chunk_size: int = 300,
    ) -> torch.Tensor:
        input_residual_codes = torch.stack(residual_codes, dim=1)
        input_residual_codes = input_residual_codes.transpose(1, 2).to(self.code2wav.device)
        wavs = self.code2wav.chunked_decode(input_residual_codes, chunk_size=chunk_size)
        wavs = wavs.float()
        return wavs

    def generate_audio_chunk(
        self,
        residual_codes: list[torch.Tensor],
        *,
        chunk_size: int = 100,
        left_context_size: int = 25,
        finished: bool = False,
    ) -> torch.Tensor | None:
        """
        Generate the newest audio chunk given all residual codes so far.

        This method uses left context when generating audio to ensure smooth
        transitions between chunks, then slices off the context portion from
        the output to avoid repetition (following HF chunked_decode pattern).

        Args:
            residual_codes: List of residual code tensors accumulated so far
            chunk_size: Number of codes per chunk (default: 100)
            left_context_size: Number of codes from previous chunk to use as context (default: 25)
            finished: Whether this is the final generation call

        Returns:
            Audio tensor with shape [..., num_samples] or None if not ready to generate
        """
        num_codes = len(residual_codes)
        if num_codes == 0:
            return None

        total_upsample = self.code2wav.total_upsample
        is_final_partial = finished and num_codes % chunk_size != 0

        # Determine if we should generate a chunk
        if not is_final_partial and num_codes % chunk_size != 0:
            # Not enough codes yet for a full chunk
            return None

        # Calculate indices for code slicing
        if is_final_partial:
            # Final partial chunk: includes leftover codes after last full chunk
            last_full_chunk_end = (num_codes // chunk_size) * chunk_size
            start_index = max(0, last_full_chunk_end - left_context_size)
            context_size = last_full_chunk_end - start_index
            chunk_type = "final_partial"
        else:
            # Regular full chunk
            start_index = max(0, num_codes - chunk_size - left_context_size)
            context_size = min(left_context_size, num_codes - chunk_size)
            chunk_type = "regular"

        logger.debug(
            "Generating %s chunk: num_codes=%d, start_index=%d, "
            "context_size=%d, chunk_size=%d",
            chunk_type, num_codes, start_index, context_size, chunk_size
        )

        # Prepare codes and generate audio
        input_codes = self._prepare_residual_codes(residual_codes[start_index:num_codes])
        wav_chunk = self.code2wav(input_codes)

        # Slice off context portion to avoid repetition at chunk boundaries
        # Matches HF implementation: wavs.append(wav_chunk[..., context_size * self.total_upsample :])
        if context_size > 0:
            slice_samples = context_size * total_upsample
            wav_chunk = wav_chunk[..., slice_samples:]
            logger.debug(
                "Sliced %d context samples from chunk (context_size=%d * total_upsample=%d)",
                slice_samples, context_size, total_upsample
            )

        return wav_chunk

    def _prepare_residual_codes(self, residual_codes: list[torch.Tensor]) -> torch.Tensor:
        """
        Prepare residual codes tensor for code2wav generation.

        Args:
            residual_codes: List of tensors of residual codes

        Returns:
            Tensor of shape [1, num_codes, code_length]
        """
        input_residual_codes = torch.stack(residual_codes, dim=1)
        input_residual_codes = input_residual_codes.transpose(1, 2).to(self.code2wav.device)
        return input_residual_codes

def find_pattern_in_tensor(input_ids: torch.Tensor, pattern: list[int]) -> list[int]:
    """
    Find all starting indices where pattern occurs in a 1D torch.Tensor.

    Args:
        input_ids: 1D tensor of token IDs to search in
        pattern: List of token IDs to find

    Returns:
        List of starting indices where pattern is found
    """
    if not pattern or len(pattern) > len(input_ids):
        return []
    indices = []
    pattern_len = len(pattern)
    pattern_tensor = torch.tensor(pattern, device=input_ids.device, dtype=input_ids.dtype)
    # Sliding window approach on tensors
    for i in range(len(input_ids) - pattern_len + 1):
        # Check if pattern matches starting at position i
        if torch.equal(input_ids[i:i + pattern_len], pattern_tensor):
            indices.append(i)
    return indices

