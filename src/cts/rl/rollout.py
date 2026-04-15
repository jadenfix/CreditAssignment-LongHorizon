"""Rollout with hidden-state capture, for the ``LocalModelAPI`` path.

For the scale path, use :meth:`ScaleBackendAdapter.rollout` instead — this
module is deliberately local-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..backends.local_api import DecodeCfg, LocalModelAPI


@dataclass
class Rollout:
    tokens: jax.Array    # [B, T_out]
    hidden: jax.Array    # [B, T_out, D]
    logprobs: jax.Array  # [B, T_out]


def rollout_local(
    model: LocalModelAPI,
    prompt_tokens: jax.Array,  # [B, T_in]
    *,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    greedy: bool = False,
    rng: jax.Array | None = None,
) -> Rollout:
    out = model.generate(
        prompt_tokens,
        DecodeCfg(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            greedy=greedy,
            rng=rng if rng is not None else jax.random.key(0),
        ),
    )
    return Rollout(tokens=out.tokens, hidden=out.hidden, logprobs=out.logprobs)


def pad_and_stack(arrs: list[jax.Array], pad_value: int = 0) -> jax.Array:
    T = max(a.shape[-1] for a in arrs)
    padded = []
    for a in arrs:
        if a.shape[-1] < T:
            pad = jnp.full(a.shape[:-1] + (T - a.shape[-1],), pad_value, dtype=a.dtype)
            a = jnp.concatenate([a, pad], axis=-1)
        padded.append(a)
    return jnp.stack(padded, axis=0)
