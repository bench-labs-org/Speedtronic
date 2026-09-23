"""Reference decoder-only transformer used by the examples and smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .registry import register_model


@dataclass
class GPTConfig:
    vocab_size: int = 512
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 8
    n_kv_head: int | None = None
    d_model: int = 256
    d_ff: int | None = None
    dropout: float = 0.0
    tie_weights: bool = True
    rope_base: float = 10_000.0

    def __post_init__(self) -> None:
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if (
            min(
                self.vocab_size,
                self.block_size,
                self.n_layer,
                self.n_head,
                self.n_kv_head,
                self.d_model,
                self.d_ff,
            )
            <= 0
        ):
            raise ValueError("all GPT dimensions must be positive")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps).to(x.dtype)) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to ``x`` with shape ``(B, H, T, D)``."""

    # cos/sin are generated as (T, D/2), then doubled for the rotate-half
    # formulation.  Keeping dimensions explicit also handles odd head sizes.
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        self.head_dim = head_dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_len = 0
        self._cached_device: torch.device | None = None
        self._cached_dtype: torch.dtype | None = None
        self._cached_cos: torch.Tensor | None = None
        self._cached_sin: torch.Tensor | None = None

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._cached_cos is not None
            and self._cached_sin is not None
            and self._cached_len >= seq_len
            and self._cached_device == device
            and self._cached_dtype == dtype
        ):
            return self._cached_cos[:seq_len], self._cached_sin[:seq_len]
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        frequencies = torch.outer(positions, self.inv_freq.to(device=device, dtype=torch.float32))
        emb = torch.cat((frequencies, frequencies), dim=-1)
        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)
        self._cached_len = seq_len
        self._cached_device = device
        self._cached_dtype = dtype
        self._cached_cos = cos
        self._cached_sin = sin
        return cos, sin


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head or config.n_head
        self.head_dim = config.d_model // config.n_head
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.d_model, config.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_head * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_head * self.head_dim, config.d_model, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        cos, sin = rope(seq_len, x.device, q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if self.n_kv_head != self.n_head:
            repeats = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        # SDPA dispatches to flash, memory-efficient, or math kernels based on
        # the device and tensor shapes.  No hardware-specific package is needed.
        attn_mask = None
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape[:2] != (batch, seq_len):
                raise ValueError("attention_mask must have shape (batch, sequence)")
            attn_mask = torch.zeros((batch, 1, seq_len, seq_len), device=x.device, dtype=q.dtype)
            key_valid = attention_mask.to(device=x.device, dtype=torch.bool)
            attn_mask.masked_fill_(~key_valid[:, None, None, :], float("-inf"))
            causal = torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool).triu(1)
            attn_mask.masked_fill_(causal[None, None, :, :], float("-inf"))
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=attn_mask is None,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.resid_dropout(self.out_proj(y))


class SwiGLU(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden_dim = int(config.d_ff * (2 / 3))
        # Round to a multiple of the head dimension for tensor-core friendly
        # shapes while preserving the requested approximate width.
        hidden_dim = ((hidden_dim + config.n_head - 1) // config.n_head) * config.n_head
        self.gate_proj = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)
        self.gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            try:
                return checkpoint(self._forward, x, rope, attention_mask, use_reentrant=False)
            except TypeError:  # pragma: no cover - PyTorch < 2.0 compatibility
                return checkpoint(self._forward, x, rope, attention_mask)
        return self._forward(x, rope, attention_mask)

    def _forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x), rope, attention_mask))
        return x + self.mlp(self.ln2(x))


@register_model("reference_transformer")
@register_model("gpt")
class ReferenceTransformer(nn.Module):
    """A compact GPT-style model with RoPE, GQA, SwiGLU, and weight tying."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        if "max_seq_len" in kwargs and "block_size" not in kwargs:
            kwargs["block_size"] = kwargs.pop("max_seq_len")
        config_values = {
            "vocab_size": kwargs.pop("vocab_size", 512),
            "block_size": kwargs.pop("block_size", 128),
            "n_layer": kwargs.pop("n_layer", 4),
            "n_head": kwargs.pop("n_head", 8),
            "n_kv_head": kwargs.pop("n_kv_head", None),
            "d_model": kwargs.pop("d_model", 256),
            "d_ff": kwargs.pop("d_ff", None),
            "dropout": kwargs.pop("dropout", 0.0),
            "tie_weights": kwargs.pop("tie_weights", True),
            "rope_base": kwargs.pop("rope_base", 10_000.0),
        }
        if kwargs:
            unknown = ", ".join(kwargs)
            raise TypeError(f"unknown model arguments: {unknown}")
        self.config = GPTConfig(**config_values)
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(self.config.vocab_size, self.config.d_model),
                "drop": nn.Dropout(self.config.dropout),
                "h": nn.ModuleList(
                    TransformerBlock(self.config) for _ in range(self.config.n_layer)
                ),
                "ln_f": RMSNorm(self.config.d_model),
            }
        )
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        self.rope = RotaryEmbedding(
            self.config.d_model // self.config.n_head, self.config.rope_base
        )
        if self.config.tie_weights:
            self.lm_head.weight = self.transformer["wte"].weight
        self.apply(self._init_weights)
        self.gradient_checkpointing = False

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)
        for block in self.transformer["h"]:
            block.gradient_checkpointing = self.gradient_checkpointing

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        seq_len = input_ids.shape[1]
        if seq_len > self.config.block_size:
            raise ValueError(
                f"sequence length {seq_len} exceeds model block size {self.config.block_size}"
            )
        x = self.transformer["wte"](input_ids)
        x = self.transformer["drop"](x)
        for block in self.transformer["h"]:
            x = block(x, self.rope, attention_mask)
        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)
        output = {"logits": logits}
        if labels is not None:
            if labels.ndim != 2 or labels.shape[0] != input_ids.shape[0]:
                raise ValueError("labels must have shape (batch, sequence)")
            if labels.shape[1] == seq_len:
                shift_labels = labels[:, 1:].contiguous()
            elif labels.shape[1] == seq_len - 1:
                shift_labels = labels
            else:
                raise ValueError("labels must have sequence length or sequence length - 1")
            shift_logits = logits[:, :-1, :].contiguous()
            if shift_labels.numel() == 0 or not torch.any(shift_labels != -100):
                output["loss"] = logits.sum() * 0.0
            else:
                output["loss"] = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                )
        return output


# Friendly aliases used by users and older examples.
GPT = ReferenceTransformer
Transformer = ReferenceTransformer
ReferenceModel = ReferenceTransformer
GPTModel = ReferenceTransformer
TransformerConfig = GPTConfig


__all__ = [
    "CausalSelfAttention",
    "GPT",
    "GPTConfig",
    "GPTModel",
    "ReferenceModel",
    "ReferenceTransformer",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLU",
    "Transformer",
    "TransformerBlock",
    "TransformerConfig",
    "apply_rope",
]
