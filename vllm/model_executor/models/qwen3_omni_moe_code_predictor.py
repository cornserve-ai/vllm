"""Code predictor module for Qwen3 Omni Moe talker."""

import os
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerConfig,
)

from vllm.model_executor.layers.layernorm import RMSNorm


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

        # Use causal attention during prefill (seq_len > 1) to match HF's
        # causal mask. For single-token decode steps the KV cache already
        # restricts the context so is_causal is not needed.
        use_causal = seq_len > 1

        try:
            attn_output = F.scaled_dot_product_attention(
                q,
                k_all,
                v_all,
                dropout_p=0.0,
                is_causal=use_causal,
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
                is_causal=use_causal,
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
    def __init__(self, config, *, text_hidden_size: int) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3OmniMoeTalkerCodePredictorModel(config)
        self.vocab_size = config.vocab_size
        self.num_code_groups = config.num_code_groups
        self.lm_head = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                for _ in range(config.num_code_groups - 1)
            ]
        )
        self._text_hidden_size = text_hidden_size
        self._tts_pad_embed_cache: dict[
            tuple[torch.device, torch.dtype], torch.Tensor
        ] = {}
        self._code_predictor_top_k = 50
        self._code_predictor_top_p = 0.8

    @classmethod
    def from_talker_config(
        cls,
        talker_config: Qwen3OmniMoeTalkerConfig,
    ) -> "Qwen3OmniMoeTalkerCodePredictor":
        return cls(
            config=talker_config.code_predictor_config,
            text_hidden_size=talker_config.text_config.hidden_size,
        )

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.model.get_input_embeddings()

    def _get_projected_tts_pad_embed(
        self,
        thinker_text_embed: nn.Module,
        text_projection: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        cache_key = (device, dtype)
        if cache_key in self._tts_pad_embed_cache:
            return self._tts_pad_embed_cache[cache_key]

        tts_pad_token_id = 151671
        tts_pad_thinker_embed = thinker_text_embed(
            torch.tensor([[tts_pad_token_id]], dtype=torch.long, device=device)
        )
        tts_pad_embed = text_projection(tts_pad_thinker_embed).to(dtype=dtype).detach()
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
        num_new_codes = self.num_code_groups - 1
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
            if self.model._use_predictor_cudagraph:
                step_hidden_states, past_key_values, cache_len = (
                    self.model.forward_tokens_cudagraph(
                        step_inputs_embeds,
                        cache_len,
                        max_cache_len,
                    )
                )
            else:
                step_hidden_states, past_key_values, cache_len = (
                    self.model.forward_tokens(
                        step_inputs_embeds,
                        past_key_values,
                        cache_len,
                        max_cache_len,
                    )
                )
            step_hidden = step_hidden_states[:, -1:, :]
            logits = self.lm_head[layer_idx](step_hidden).squeeze(1)
            sampled_code = self._sample_codec_token(logits)

            residual_codes[:, layer_idx + 1 : layer_idx + 2] = sampled_code
            sampled_embed = self.model.codec_embedding[layer_idx](sampled_code)
            last_sampled_embed = sampled_embed
            if layer_idx < num_new_codes - 1:
                codec_hiddens[:, layer_idx + 1 : layer_idx + 2] = sampled_embed

            step_inputs_embeds = sampled_embed

        if last_sampled_embed is not None:
            codec_hiddens[:, -1:] = last_sampled_embed

        return codec_hiddens, residual_codes

    def prepare_inputs_from_past_hidden_states(
        self,
        trailing_text_hidden_states: torch.Tensor,
        input_ids: list[int] | int,
        past_hidden_states: torch.Tensor,
        generation_step: int,
        thinker_text_embed: nn.Module,
        text_projection: nn.Module,
        codec_embed_tokens: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if trailing_text_hidden_states.dim() == 2:
            trailing_text_hidden_states = trailing_text_hidden_states.unsqueeze(0)

        device = next(thinker_text_embed.parameters()).device

        if trailing_text_hidden_states.shape[-1] == self._text_hidden_size:
            trailing_text_hidden_states = trailing_text_hidden_states.to(device)
        else:
            trailing_text_hidden_states = text_projection(
                trailing_text_hidden_states.to(device)
            )

        last_token_id = input_ids if isinstance(input_ids, int) else input_ids[-1]
        last_token_ids = torch.tensor(
            [[last_token_id]], dtype=torch.long, device=device
        )

        last_id_hidden = codec_embed_tokens(last_token_ids)
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
                thinker_text_embed,
                text_projection,
                device,
                inputs_embeds.dtype,
            )
            inputs_embeds = inputs_embeds + tts_pad_embed

        inputs_embeds = inputs_embeds.squeeze(0)
        return inputs_embeds, residual_codes
