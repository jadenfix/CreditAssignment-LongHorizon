"""``LocalModelAPI`` — the smoke / unit-test backend abstraction.

Kept deliberately rich so every method and loss can be exercised end-to-end on
CPU. Scale backends implement :class:`cts.backends.scale_adapter.ScaleBackendAdapter`
instead; they share no protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp


@dataclass
class ForwardOut:
    logits: jax.Array  # [B, T, V]
    hidden: jax.Array  # [B, T, D]


@dataclass
class GenerateOut:
    tokens: jax.Array  # [B, T_out]
    hidden: jax.Array  # [B, T_out, D]
    logprobs: jax.Array  # [B, T_out]


@dataclass
class DecodeCfg:
    max_new_tokens: int = 32
    temperature: float = 1.0
    greedy: bool = False
    rng: jax.Array | None = None


@runtime_checkable
class LocalModelAPI(Protocol):
    """Small-model interface for tests and smoke runs."""

    vocab_size: int
    hidden_size: int

    def forward(self, tokens: jax.Array) -> ForwardOut: ...

    def generate(self, prompt_tokens: jax.Array, cfg: DecodeCfg) -> GenerateOut: ...

    def params(self) -> Any: ...

    def apply_grads(self, grads: Any, lr: float) -> None: ...


def pad_to(arr: jax.Array, length: int, axis: int = -1, pad_value: int = 0) -> jax.Array:
    cur = arr.shape[axis]
    if cur >= length:
        return arr
    pad_shape = list(arr.shape)
    pad_shape[axis] = length - cur
    return jnp.concatenate([arr, jnp.full(pad_shape, pad_value, dtype=arr.dtype)], axis=axis)
