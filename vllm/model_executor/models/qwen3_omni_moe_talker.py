"""Inference-only Qwen3-Omni-Moe model (talker part)."""

import os
from collections.abc import Iterable
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
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


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch_size, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size,
        num_kv_heads,
        n_rep,
        seq_len,
        head_dim,
    )
    return hidden_states.reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)


class Qwen3OmniMoeCodePredictorRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position = position_ids[:, None, :].float()
        freqs = (inv_freq @ position).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)
        return cos, sin


class Qwen3OmniMoeTalkerCodePredictorAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        self.hidden_size = config.hidden_size

        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )

        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

        self.q_proj = nn.Linear(self.hidden_size, self.q_size, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.kv_size,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.kv_size,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self._qkv_proj_weight_cache: dict[
            tuple[torch.device, torch.dtype], torch.Tensor
        ] = {}

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        cache_len: int,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        cache_key = (hidden_states.device, hidden_states.dtype)
        qkv_proj_weight = self._qkv_proj_weight_cache.get(cache_key)
        if qkv_proj_weight is None:
            qkv_proj_weight = torch.cat(
                (
                    self.q_proj.weight,
                    self.k_proj.weight,
                    self.v_proj.weight,
                ),
                dim=0,
            )
            if qkv_proj_weight.device != hidden_states.device:
                qkv_proj_weight = qkv_proj_weight.to(hidden_states.device)
            if qkv_proj_weight.dtype != hidden_states.dtype:
                qkv_proj_weight = qkv_proj_weight.to(hidden_states.dtype)
            self._qkv_proj_weight_cache[cache_key] = qkv_proj_weight

        qkv = F.linear(hidden_states, qkv_proj_weight)
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        )
        v = v.view(
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        )

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        cached_k, cached_v = kv_cache
        write_end = cache_len + seq_len
        cached_k[:, :, cache_len:write_end, :] = k
        cached_v[:, :, cache_len:write_end, :] = v

        k_all = cached_k[:, :, :write_end, :]
        v_all = cached_v[:, :, :write_end, :]

        try:
            attn_output = F.scaled_dot_product_attention(
                q,
                k_all,
                v_all,
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=self.num_key_value_groups > 1,
            )
        except RuntimeError:
            k_rep = repeat_kv(k_all, self.num_key_value_groups)
            v_rep = repeat_kv(v_all, self.num_key_value_groups)
            attn_output = F.scaled_dot_product_attention(
                q,
                k_rep,
                v_rep,
                dropout_p=0.0,
                is_causal=False,
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen3OmniMoeTalkerCodePredictorMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(
            config.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            self.intermediate_size, config.hidden_size, bias=False
        )
        self._gate_up_proj_weight_cache: dict[
            tuple[torch.device, torch.dtype], torch.Tensor
        ] = {}

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        cache_key = (hidden_states.device, hidden_states.dtype)
        gate_up_proj_weight = self._gate_up_proj_weight_cache.get(cache_key)
        if gate_up_proj_weight is None:
            gate_up_proj_weight = torch.cat(
                (
                    self.gate_proj.weight,
                    self.up_proj.weight,
                ),
                dim=0,
            )
            if gate_up_proj_weight.device != hidden_states.device:
                gate_up_proj_weight = gate_up_proj_weight.to(hidden_states.device)
            if gate_up_proj_weight.dtype != hidden_states.dtype:
                gate_up_proj_weight = gate_up_proj_weight.to(hidden_states.dtype)
            self._gate_up_proj_weight_cache[cache_key] = gate_up_proj_weight

        gate_up = F.linear(hidden_states, gate_up_proj_weight)
        gate, up = torch.split(gate_up, self.intermediate_size, dim=-1)
        gate = F.silu(gate)
        return self.down_proj(gate * up)


class Qwen3OmniMoeTalkerCodePredictorDecoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.self_attn = Qwen3OmniMoeTalkerCodePredictorAttention(config)
        self.mlp = Qwen3OmniMoeTalkerCodePredictorMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        cache_len: int,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(hidden_states, cos, sin, kv_cache, cache_len)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3OmniMoeTalkerCodePredictorModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_code_groups = config.num_code_groups

        self.layers = nn.ModuleList(
            [
                Qwen3OmniMoeTalkerCodePredictorDecoderLayer(config)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.codec_embedding = nn.ModuleList(
            [
                nn.Embedding(config.vocab_size, config.hidden_size)
                for _ in range(config.num_code_groups - 1)
            ]
        )

        self.rotary_emb = Qwen3OmniMoeCodePredictorRotaryEmbedding(
            dim=getattr(
                config, "head_dim", config.hidden_size // config.num_attention_heads
            ),
            max_position_embeddings=config.max_position_embeddings,
            base=getattr(config, "rope_theta", 10000.0),
        )
        self._kv_cache_pool: dict[
            tuple[torch.device, torch.dtype, int, int],
            list[tuple[torch.Tensor, torch.Tensor]],
        ] = {}
        self._position_ids_cache: dict[tuple[torch.device, int], torch.Tensor] = {}
        self._rotary_cache: dict[
            tuple[torch.device, torch.dtype, int],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}
        self._step_graph_pool: dict[
            tuple[torch.device, torch.dtype, int, int],
            tuple[
                list[tuple[torch.Tensor, torch.Tensor]],
                dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]],
            ],
        ] = {}
        use_predictor_cudagraph = (
            os.environ.get("CORNSERVE_TALKER_CODE_PREDICTOR_CUDAGRAPH", "")
            .strip()
            .lower()
        )
        self._use_predictor_cudagraph = use_predictor_cudagraph not in {
            "0",
            "false",
            "no",
            "off",
        }

    def _get_position_ids(
        self,
        device: torch.device,
        max_cache_len: int,
    ) -> torch.Tensor:
        cache_key = (device, max_cache_len)
        position_ids = self._position_ids_cache.get(cache_key)
        if position_ids is None:
            position_ids = torch.arange(
                max_cache_len,
                device=device,
                dtype=torch.long,
            )
            self._position_ids_cache[cache_key] = position_ids
        return position_ids

    def _get_rotary_embeds(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        max_cache_len: int,
        cache_len: int,
        seq_len: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key = (device, dtype, max_cache_len)
        cached = self._rotary_cache.get(cache_key)
        if cached is None:
            position_ids = self._get_position_ids(device, max_cache_len).unsqueeze(0)
            dummy = torch.empty((1, max_cache_len, 1), dtype=dtype, device=device)
            cached = self.rotary_emb(dummy, position_ids)
            self._rotary_cache[cache_key] = cached

        cos_cache, sin_cache = cached
        end = cache_len + seq_len
        cos = cos_cache[:, cache_len:end, :]
        sin = sin_cache[:, cache_len:end, :]
        if batch_size > 1:
            cos = cos.expand(batch_size, -1, -1)
            sin = sin.expand(batch_size, -1, -1)
        return cos, sin

    def _get_or_create_kv_caches(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        max_cache_len: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        cache_key = (device, dtype, batch_size, max_cache_len)
        kv_caches = self._kv_cache_pool.get(cache_key)
        if kv_caches is None:
            kv_caches = []
            for layer in self.layers:
                predictor_layer = cast(
                    Qwen3OmniMoeTalkerCodePredictorDecoderLayer,
                    layer,
                )
                num_kv_heads = predictor_layer.self_attn.num_key_value_heads
                head_dim = predictor_layer.self_attn.head_dim
                cache_shape = (batch_size, num_kv_heads, max_cache_len, head_dim)
                k_cache = torch.empty(cache_shape, dtype=dtype, device=device)
                v_cache = torch.empty(cache_shape, dtype=dtype, device=device)
                kv_caches.append((k_cache, v_cache))
            self._kv_cache_pool[cache_key] = kv_caches
        return kv_caches

    def _forward_tokens_impl(
        self,
        inputs_embeds: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
        cache_len: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        for layer_idx, layer in enumerate(self.layers):
            predictor_layer = cast(Qwen3OmniMoeTalkerCodePredictorDecoderLayer, layer)
            layer_cache = kv_caches[layer_idx]
            hidden_states = predictor_layer(
                hidden_states,
                cos,
                sin,
                layer_cache,
                cache_len,
            )
        return self.norm(hidden_states)

    def _get_or_create_step_graphs(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        max_cache_len: int,
    ) -> tuple[
        list[tuple[torch.Tensor, torch.Tensor]],
        dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]],
    ]:
        cache_key = (device, dtype, batch_size, max_cache_len)
        cached = self._step_graph_pool.get(cache_key)
        if cached is not None:
            return cached

        kv_caches = self._get_or_create_kv_caches(
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
        )
        step_graphs: dict[
            int,
            tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor],
        ] = {}

        capture_stream = torch.cuda.Stream(device=device)
        current_stream = torch.cuda.current_stream(device=device)
        capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(capture_stream):
            for step_cache_len in range(max_cache_len):
                static_input = torch.zeros(
                    (batch_size, 1, self.config.hidden_size),
                    dtype=dtype,
                    device=device,
                )
                cos, sin = self._get_rotary_embeds(
                    device=device,
                    dtype=dtype,
                    max_cache_len=max_cache_len,
                    cache_len=step_cache_len,
                    seq_len=1,
                    batch_size=batch_size,
                )
                self._forward_tokens_impl(
                    static_input,
                    kv_caches,
                    step_cache_len,
                    cos,
                    sin,
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static_output = self._forward_tokens_impl(
                        static_input,
                        kv_caches,
                        step_cache_len,
                        cos,
                        sin,
                    )
                step_graphs[step_cache_len] = (graph, static_input, static_output)
        current_stream.wait_stream(capture_stream)

        cached = (kv_caches, step_graphs)
        self._step_graph_pool[cache_key] = cached
        return cached

    def forward_tokens_cudagraph(
        self,
        inputs_embeds: torch.Tensor,
        cache_len: int,
        max_cache_len: int,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]], int]:
        batch_size, seq_len, _ = inputs_embeds.shape
        if (
            not self._use_predictor_cudagraph
            or inputs_embeds.device.type != "cuda"
            or seq_len != 1
        ):
            return self.forward_tokens(
                inputs_embeds,
                kv_caches=None,
                cache_len=cache_len,
                max_cache_len=max_cache_len,
            )

        kv_caches, step_graphs = self._get_or_create_step_graphs(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
        )
        graph, static_input, static_output = step_graphs[cache_len]
        static_input.copy_(inputs_embeds)
        graph.replay()
        return static_output, kv_caches, cache_len + 1

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.codec_embedding

    def forward_tokens(
        self,
        inputs_embeds: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None,
        cache_len: int,
        max_cache_len: int,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]], int]:
        batch_size, seq_len, _ = inputs_embeds.shape

        if kv_caches is None:
            kv_caches = self._get_or_create_kv_caches(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
                batch_size=batch_size,
                max_cache_len=max_cache_len,
            )

        next_cache_len = cache_len + seq_len
        cos, sin = self._get_rotary_embeds(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
            max_cache_len=max_cache_len,
            cache_len=cache_len,
            seq_len=seq_len,
            batch_size=batch_size,
        )

        hidden_states = self._forward_tokens_impl(
            inputs_embeds,
            kv_caches,
            cache_len,
            cos,
            sin,
        )
        return hidden_states, kv_caches, next_cache_len


class Qwen3OmniMoeTalkerCodePredictor(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.model = Qwen3OmniMoeTalkerCodePredictorModel(config)
        self.vocab_size = config.vocab_size
        self.num_code_groups = config.num_code_groups
        self.lm_head = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                for _ in range(config.num_code_groups - 1)
            ]
        )

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.model.get_input_embeddings()


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
            lambda prefix: Qwen3MoeTalkerDecoderLayer(
                vllm_config=vllm_config, prefix=prefix
            ),
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

    merge_by_field_config = True

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
        talker_vllm_config = talker_vllm_config.with_hf_text_config(
            talker_config.text_config
        )

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

        self.code_predictor = Qwen3OmniMoeTalkerCodePredictor(
            config=talker_config.code_predictor_config
        )
        self._tts_pad_embed_cache: dict[
            tuple[torch.device, torch.dtype], torch.Tensor
        ] = {}
        self._code_predictor_top_k = 50
        self._code_predictor_top_p = 0.8

        compile_predictor = os.environ.get("CORNSERVE_TALKER_CODE_PREDICTOR_COMPILE")
        if compile_predictor is not None and compile_predictor.lower() not in {
            "",
            "0",
            "false",
            "no",
        }:
            try:
                self.code_predictor.model.forward_tokens = torch.compile(  # type: ignore[method-assign]
                    self.code_predictor.model.forward_tokens,
                    mode="default",
                    dynamic=False,
                    fullgraph=False,
                )
                logger.info("Enabled torch.compile for talker code predictor")
            except Exception:
                logger.exception(
                    "Failed to enable torch.compile for talker code predictor"
                )
        self.make_empty_intermediate_tensors = (
            self.language_model.model.make_empty_intermediate_tensors
        )

    def _get_projected_tts_pad_embed(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        cache_key = (device, dtype)
        if cache_key in self._tts_pad_embed_cache:
            return self._tts_pad_embed_cache[cache_key]

        tts_pad_token_id = 151671
        tts_pad_thinker_embed = self.thinker_text_embed(
            torch.tensor([[tts_pad_token_id]], dtype=torch.long, device=device)
        )
        tts_pad_embed = (
            self.text_projection(tts_pad_thinker_embed).to(dtype=dtype).detach()
        )
        self._tts_pad_embed_cache[cache_key] = tts_pad_embed
        return tts_pad_embed

    def _sample_codec_token(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.clamp(logits.to(dtype=torch.float32), min=-100.0, max=100.0)
        scores = torch.nan_to_num(scores, nan=-100.0, posinf=100.0, neginf=-100.0)

        top_k = min(self._code_predictor_top_k, scores.shape[-1])
        if top_k < scores.shape[-1]:
            topk_scores, topk_indices = torch.topk(scores, k=top_k, dim=-1)
        else:
            topk_scores = scores
            topk_indices = torch.arange(
                scores.shape[-1], device=scores.device, dtype=torch.long
            ).expand(scores.shape[0], -1)

        probs = torch.softmax(topk_scores, dim=-1)
        if self._code_predictor_top_p < 1.0:
            cumulative_probs = torch.cumsum(probs, dim=-1)
            nucleus_mask = cumulative_probs > self._code_predictor_top_p
            nucleus_mask[..., 0] = False
            probs = probs.masked_fill(nucleus_mask, 0.0)

        prob_mass = probs.sum(dim=-1, keepdim=True)
        zero_mass_mask = prob_mass <= 0
        safe_prob_mass = torch.where(
            zero_mass_mask, torch.ones_like(prob_mass), prob_mass
        )
        probs = probs / safe_prob_mass

        fallback_probs = torch.zeros_like(probs)
        fallback_probs[..., 0] = 1.0
        probs = torch.where(zero_mass_mask, fallback_probs, probs)

        sampled_idx = torch.multinomial(probs, num_samples=1)
        best_idx = topk_scores.argmax(dim=-1, keepdim=True)
        final_idx = torch.where(zero_mass_mask, best_idx, sampled_idx)
        return topk_indices.gather(-1, final_idx)

    def _predict_codec_hiddens_and_codes(
        self,
        predictor_input: torch.Tensor,
        last_token_ids: torch.Tensor,
        last_id_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_new_codes = self.config.code_predictor_config.num_code_groups - 1
        if num_new_codes <= 0:
            return last_id_hidden, last_token_ids

        batch_size = last_token_ids.shape[0]
        hidden_size = last_id_hidden.shape[-1]
        residual_codes = torch.zeros(
            (batch_size, num_new_codes + 1),
            dtype=torch.long,
            device=last_token_ids.device,
        )
        residual_codes[:, :1] = last_token_ids
        codec_hiddens = torch.zeros(
            (batch_size, num_new_codes + 1, hidden_size),
            dtype=last_id_hidden.dtype,
            device=last_id_hidden.device,
        )
        codec_hiddens[:, :1] = last_id_hidden

        step_inputs_embeds = predictor_input
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        cache_len = 0
        max_cache_len = num_new_codes + 1
        last_sampled_embed: torch.Tensor | None = None

        for layer_idx in range(num_new_codes):
            if self.code_predictor.model._use_predictor_cudagraph:
                step_hidden_states, past_key_values, cache_len = (
                    self.code_predictor.model.forward_tokens_cudagraph(
                        step_inputs_embeds,
                        cache_len,
                        max_cache_len,
                    )
                )
            else:
                step_hidden_states, past_key_values, cache_len = (
                    self.code_predictor.model.forward_tokens(
                        step_inputs_embeds,
                        past_key_values,
                        cache_len,
                        max_cache_len,
                    )
                )
            step_hidden = step_hidden_states[:, -1:, :]
            logits = self.code_predictor.lm_head[layer_idx](step_hidden).squeeze(1)
            sampled_code = self._sample_codec_token(logits)

            residual_codes[:, layer_idx + 1 : layer_idx + 2] = sampled_code
            sampled_embed = self.code_predictor.model.codec_embedding[layer_idx](
                sampled_code
            )
            last_sampled_embed = sampled_embed
            if layer_idx < num_new_codes - 1:
                codec_hiddens[:, layer_idx + 1 : layer_idx + 2] = sampled_embed

            step_inputs_embeds = sampled_embed

        if last_sampled_embed is not None:
            codec_hiddens[:, -1:] = last_sampled_embed

        return codec_hiddens, residual_codes

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
        trailing_text_hidden_states: torch.Tensor,
        input_ids: list[int] | int,
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
            trailing_text_hidden_states = trailing_text_hidden_states.unsqueeze(0)

        device = next(self.thinker_text_embed.parameters()).device

        if trailing_text_hidden_states.shape[-1] == self.config.text_config.hidden_size:
            trailing_text_hidden_states = trailing_text_hidden_states.to(device)
        else:
            trailing_text_hidden_states = self.text_projection(
                trailing_text_hidden_states.to(device)
            )

        last_token_id = input_ids if isinstance(input_ids, int) else input_ids[-1]
        last_token_ids = torch.tensor(
            [[last_token_id]], dtype=torch.long, device=device
        )

        last_id_hidden = self.language_model.model.embed_tokens(last_token_ids)
        past_hidden_states = past_hidden_states.to(device)

        if past_hidden_states.dim() == 2:
            past_hidden = past_hidden_states[-1:, :].unsqueeze(0)
        else:
            past_hidden = past_hidden_states[:, -1:, :]

        predictor_input = torch.cat((past_hidden, last_id_hidden), dim=1)
        codec_hiddens, residual_codes = self._predict_codec_hiddens_and_codes(
            predictor_input,
            last_token_ids,
            last_id_hidden,
        )

        inputs_embeds = codec_hiddens.sum(1, keepdim=True)
        if inputs_embeds.dtype != last_id_hidden.dtype:
            inputs_embeds = inputs_embeds.to(dtype=last_id_hidden.dtype)

        if generation_step < trailing_text_hidden_states.shape[1]:
            inputs_embeds = inputs_embeds + trailing_text_hidden_states[
                :, generation_step
            ].unsqueeze(1)
        else:
            tts_pad_embed = self._get_projected_tts_pad_embed(
                device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds + tts_pad_embed

        inputs_embeds = inputs_embeds.squeeze(0)
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
            dtype=thinker_embeds.dtype,  # Match dtype to avoid mismatch
        )
        device = talker_embeds.device

        if is_multimodal is not None and is_multimodal.any().item():
            num_mm_tokens = is_multimodal.sum().item()
            thinker_hidden = thinker_hidden_states.to(device)
            thinker_hidden = thinker_hidden[: is_multimodal.shape[0], :]
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
            talker_embeds = self._replace_codec_control_embeddings_at_index(
                talker_embeds, idx
            )
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
        first_3_text_embeds = talker_embeddings[
            index : index + 3
        ].clone()  # [3, hidden_size]

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
        tts_embeds_thinker = self.thinker_text_embed(
            talker_special_tokens
        )  # [1, 3, thinker_hidden_size]
        tts_embeds_projected = self.text_projection(
            tts_embeds_thinker
        )  # [1, 3, talker_hidden_size]

        # Chunk along dim=1
        tts_bos_embed, tts_eos_embed, tts_pad_embed = tts_embeds_projected.chunk(
            3, dim=1
        )

        # Reshape and expand
        tts_bos_embed = tts_bos_embed.squeeze(1)  # [1, talker_hidden_size]
        tts_pad_embed = tts_pad_embed.squeeze(1)  # [1, talker_hidden_size]
        tts_pad_embed = tts_pad_embed.expand(4, -1)  # [4, talker_hidden_size]

        # For the last position, we need the to use the first text from thinker's output
        # For batching, we'll we the embedding of index+8:+9, relying on the last token id being replaced by the first thinker's output token already
        last_text_embed = talker_embeddings[index + 8 : index + 9]  # [1, hidden_size]

        # Build assistant_text_hidden: [3 first tokens][4 tts_pad][1 tts_bos][1 last token]
        assistant_text_hidden = torch.cat(
            [first_3_text_embeds, tts_pad_embed, tts_bos_embed, last_text_embed],
            dim=0,
        )  # [9, hidden_size]

        # Create codec_special_tokens
        speaker_id_map = getattr(self.config, "speaker_id", {})
        speaker_id = speaker_id_map.get("ethan", 2302)  # Default to Ethan
        codec_special_tokens = torch.tensor(
            [
                self.config.codec_nothink_id,  # Position 3
                self.config.codec_think_bos_id,  # Position 4
                self.config.codec_think_eos_id,  # Position 5
                speaker_id,  # Position 6
                self.config.codec_pad_id,  # Position 7
                self.config.codec_bos_id,  # Position 8
            ],
            device=device,
            dtype=torch.long,
        )

        # Get codec embeddings
        codec_embeds = self.language_model.model.embed_tokens(
            codec_special_tokens
        )  # [6, hidden_size]

        # Build assistant_codec_hidden: [3 zeros][6 codec embeddings]
        zeros_for_control = torch.zeros(3, hidden_size, device=device, dtype=dtype)
        assistant_codec_hidden = torch.cat(
            [zeros_for_control, codec_embeds], dim=0
        )  # [9, hidden_size]

        # Final embeddings
        final_assistant_embeds = (
            assistant_text_hidden + assistant_codec_hidden
        )  # [9, hidden_size]

        # Replace the 9 token embeddings at the specified index
        talker_embeddings = talker_embeddings.clone()
        talker_embeddings[index : index + 9] = final_assistant_embeds

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
        if "pixel_values" in kwargs and isinstance(
            kwargs["pixel_values"], torch.Tensor
        ):
            kwargs["pixel_values"] = self._squeeze_batch_dims(kwargs["pixel_values"], 2)
        if "image_grid_thw" in kwargs and isinstance(
            kwargs["image_grid_thw"], torch.Tensor
        ):
            ig = self._squeeze_batch_dims(kwargs["image_grid_thw"], 2)
            kwargs["image_grid_thw"] = ig.unsqueeze(0) if ig.ndim == 1 else ig

        # Video: similar pattern
        if "pixel_values_videos" in kwargs and isinstance(
            kwargs["pixel_values_videos"], torch.Tensor
        ):
            kwargs["pixel_values_videos"] = self._squeeze_batch_dims(
                kwargs["pixel_values_videos"], 2
            )
        if "video_grid_thw" in kwargs and isinstance(
            kwargs["video_grid_thw"], torch.Tensor
        ):
            vg = self._squeeze_batch_dims(kwargs["video_grid_thw"], 2)
            kwargs["video_grid_thw"] = vg.unsqueeze(0) if vg.ndim == 1 else vg

        # Audio: squeeze features and mask, flatten lengths
        for key in ("input_audio_features", "input_features"):
            if key in kwargs and isinstance(kwargs[key], torch.Tensor):
                kwargs[key] = self._squeeze_batch_dims(kwargs[key], 2)
        if "feature_attention_mask" in kwargs and isinstance(
            kwargs["feature_attention_mask"], torch.Tensor
        ):
            fam = self._squeeze_batch_dims(kwargs["feature_attention_mask"], 2)
            kwargs["feature_attention_mask"] = (
                fam.unsqueeze(0) if fam.ndim == 1 else fam
            )
        if "audio_feature_lengths" in kwargs and isinstance(
            kwargs["audio_feature_lengths"], torch.Tensor
        ):
            afl = kwargs["audio_feature_lengths"]
            kwargs["audio_feature_lengths"] = (
                afl.reshape(-1) if afl.ndim > 0 else afl.reshape(1)
            )

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
            if (
                input_key in ("pixel_values", "image_embeds")
                and "image" not in mm_input_by_modality
            ):
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(
                    **kwargs
                )
            if (
                input_key in ("pixel_values_videos", "video_embeds")
                and "video" not in mm_input_by_modality
            ):
                mm_input_by_modality["video"] = self._parse_and_validate_video_input(
                    **kwargs
                )
            if (
                input_key in ("input_audio_features",)
                and "audio" not in mm_input_by_modality
            ):
                mm_input_by_modality["audio"] = self._parse_and_validate_audio_input(
                    **kwargs
                )

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
                    sizes = (image_grid_thw.prod(-1) // SPATIAL_MERGE_SIZE**2).tolist()
                    multimodal_embeddings += tuple(
                        torch.empty(
                            sz,
                            VISION_EMBED_DIM,
                            dtype=pixel_values.dtype,
                            device=pixel_values.device,
                        )
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
                    sizes = (video_grid_thw.prod(-1) // SPATIAL_MERGE_SIZE**2).tolist()
                    multimodal_embeddings += tuple(
                        torch.empty(
                            sz,
                            VISION_EMBED_DIM,
                            dtype=pixel_values_videos.dtype,
                            device=pixel_values_videos.device,
                        )
                        for sz in sizes
                    )
            elif modality == "audio":
                # Compute audio output lengths using helper function
                audio_feature_lengths = mm_input["audio_feature_lengths"]
                _, audio_output_lengths = _get_feat_extract_output_lengths(
                    audio_feature_lengths
                )
                input_features = mm_input["input_features"]
                multimodal_embeddings += tuple(
                    torch.empty(
                        n,
                        AUDIO_EMBED_DIM,
                        dtype=input_features.dtype,
                        device=input_features.device,
                    )
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
        input_residual_codes = input_residual_codes.transpose(1, 2).to(
            self.code2wav.device
        )
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
            chunk_type,
            num_codes,
            start_index,
            context_size,
            chunk_size,
        )

        # Prepare codes and generate audio
        input_codes = self._prepare_residual_codes(
            residual_codes[start_index:num_codes]
        )
        wav_chunk = self.code2wav(input_codes)

        # Slice off context portion to avoid repetition at chunk boundaries
        # Matches HF implementation: wavs.append(wav_chunk[..., context_size * self.total_upsample :])
        if context_size > 0:
            slice_samples = context_size * total_upsample
            wav_chunk = wav_chunk[..., slice_samples:]
            logger.debug(
                "Sliced %d context samples from chunk (context_size=%d * total_upsample=%d)",
                slice_samples,
                context_size,
                total_upsample,
            )

        return wav_chunk

    def _prepare_residual_codes(
        self, residual_codes: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Prepare residual codes tensor for code2wav generation.

        Args:
            residual_codes: List of tensors of residual codes

        Returns:
            Tensor of shape [1, num_codes, code_length]
        """
        input_residual_codes = torch.stack(residual_codes, dim=1)
        input_residual_codes = input_residual_codes.transpose(1, 2).to(
            self.code2wav.device
        )
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
    pattern_len = len(pattern)
    pattern_tensor = torch.tensor(
        pattern, device=input_ids.device, dtype=input_ids.dtype
    )
    windows = input_ids.unfold(0, pattern_len, 1)
    matches = (windows == pattern_tensor).all(dim=1)
    return matches.nonzero(as_tuple=False).flatten().tolist()
