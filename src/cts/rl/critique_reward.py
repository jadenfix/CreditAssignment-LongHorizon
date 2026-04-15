"""Critique reward channel — the Critique-GRPO style signal.

Given the outcome reward ``r_outcome`` and a scalar ``critique_uptake`` score
(higher = revision addressed the critique better, e.g. via a learned or rule
classifier), combine into a single reward with configurable weight. This is
the reward fed into the GRPO group-relative baseline for method B4.
"""

from __future__ import annotations

import jax.numpy as jnp


def critique_reward(
    r_outcome: jnp.ndarray,
    critique_uptake: jnp.ndarray,
    *,
    lam: float = 0.3,
) -> jnp.ndarray:
    """``r = r_outcome + lam * critique_uptake``."""
    return r_outcome + lam * critique_uptake
