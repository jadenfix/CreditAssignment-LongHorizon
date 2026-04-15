"""A very thin GRPO step, local variant.

This is **not** a replacement for Tunix's GRPO trainer — it exists so nano-LM
smoke tests can exercise the RL path end-to-end without pulling Tunix. The
scale path routes through :class:`cts.backends.tunix_adapter.TunixAdapter`.

Reference: DeepSeekMath, Shao et al. 2024. Given a group of ``G`` rollouts per
prompt with rewards ``r_i``, the group-relative advantage is
``A_i = (r_i - mean(r)) / (std(r) + eps)``, and the policy-gradient surrogate
is ``-E[A * logpi(y|x)]`` with a clipped-ratio PPO term optional.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class GRPOCfg:
    kl_coef: float = 0.01
    clip: float = 0.2
    eps: float = 1e-6


def group_relative_advantages(rewards: jax.Array, eps: float = 1e-6) -> jax.Array:
    """``rewards: [B, G]`` -> ``A: [B, G]`` (per-group standardization)."""
    mu = rewards.mean(axis=-1, keepdims=True)
    sd = rewards.std(axis=-1, keepdims=True)
    return (rewards - mu) / (sd + eps)


def grpo_surrogate_loss(
    logprobs_new: jax.Array,  # [B, G, T]
    logprobs_old: jax.Array,  # [B, G, T] (stop-grad)
    mask: jax.Array,          # [B, G, T]
    advantages: jax.Array,    # [B, G]
    cfg: GRPOCfg,
) -> jax.Array:
    logprobs_old = jax.lax.stop_gradient(logprobs_old)
    ratio = jnp.exp((logprobs_new - logprobs_old) * mask)
    A = advantages[..., None]
    unclipped = ratio * A
    clipped = jnp.clip(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * A
    surrogate = jnp.minimum(unclipped, clipped)
    denom = jnp.maximum(mask.sum(axis=-1), 1.0)
    per_seq = (surrogate * mask).sum(axis=-1) / denom
    pg_loss = -per_seq.mean()
    if cfg.kl_coef != 0.0:
        pg_loss = pg_loss + cfg.kl_coef * approx_kl(logprobs_new, logprobs_old, mask)
    return pg_loss


def approx_kl(logprobs_new: jax.Array, logprobs_old: jax.Array, mask: jax.Array) -> jax.Array:
    """Schulman k3 estimator of KL(new || old): always ≥ 0, low variance.

    kl ≈ exp(logp_old - logp_new) - (logp_old - logp_new) - 1, averaged over
    unmasked tokens. Equals 0 iff the two distributions agree on the sample.
    """
    r = (logprobs_old - logprobs_new) * mask
    kl = (jnp.exp(r) - r - 1.0) * mask
    denom = jnp.maximum(mask.sum(), 1.0)
    return kl.sum() / denom
