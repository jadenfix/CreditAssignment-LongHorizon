"""Decode anchor: ``-log pi_theta(y1 | x, y0, f)``.

This is the grounding term that keeps CTS from wandering off into latent space.
Hidden-state optimization works, but only as an auxiliary — the decode path
still needs explicit supervision on the corrected revision.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def decode_anchor_loss(
    logits: jax.Array,       # [B, T, V]  logits for teacher-forced targets
    target_tokens: jax.Array, # [B, T]     next-token targets
    mask: jax.Array | None = None,  # [B, T] 1 where we supervise y1 tokens, 0 on prompt
) -> jax.Array:
    """Mean token-level NLL over the masked positions (typically y1 tokens)."""
    log_p = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_p, target_tokens[..., None], axis=-1)[..., 0]  # [B, T]
    if mask is None:
        return nll.mean()
    mask = mask.astype(nll.dtype)
    denom = jnp.maximum(mask.sum(), 1.0)
    return (nll * mask).sum() / denom
