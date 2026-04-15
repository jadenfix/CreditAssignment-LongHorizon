"""Alignment-loss dispatcher.

Routes ``kind = {"sinkhorn", "soft_dtw", "l2"}`` to the right backend. This is
the single seam for ablation A2 (swap in L2) and A3 (swap in soft-DTW).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp

from .sinkhorn import sinkhorn_divergence_loss
from .soft_dtw import soft_dtw_loss

AlignmentKind = Literal["sinkhorn", "soft_dtw", "l2"]


@dataclass
class AlignmentCfg:
    kind: AlignmentKind = "sinkhorn"
    weight: float = 1.0
    # sinkhorn
    epsilon: float = 0.1
    time_alpha: float = 0.2
    num_iters: int = 50
    unbalanced_tau: float | None = None
    use_ott: bool = True
    # soft_dtw
    gamma: float = 1.0
    # shared
    extra: dict = field(default_factory=dict)


def _l2_seq_loss(R_a: jax.Array, R_b: jax.Array) -> jax.Array:
    """Plain per-timestep MSE; requires equal T_a == T_b (we assume truncation upstream)."""
    if R_a.shape != R_b.shape:
        T = min(R_a.shape[-2], R_b.shape[-2])
        R_a = R_a[..., :T, :]
        R_b = R_b[..., :T, :]
    return jnp.mean((R_a - R_b) ** 2)


def alignment_loss(R_a: jax.Array, R_b: jax.Array, cfg: AlignmentCfg) -> jax.Array:
    if cfg.weight == 0.0:
        return jnp.zeros((), dtype=R_a.dtype)
    if cfg.kind == "l2":
        base = _l2_seq_loss(R_a, R_b)
    elif cfg.kind == "soft_dtw":
        base = soft_dtw_loss(R_a, R_b, gamma=cfg.gamma)
    elif cfg.kind == "sinkhorn":
        base = sinkhorn_divergence_loss(
            R_a,
            R_b,
            epsilon=cfg.epsilon,
            time_alpha=cfg.time_alpha,
            num_iters=cfg.num_iters,
            unbalanced_tau=cfg.unbalanced_tau,
            use_ott=cfg.use_ott,
        )
    else:
        raise ValueError(f"Unknown alignment kind: {cfg.kind}")
    return cfg.weight * base
