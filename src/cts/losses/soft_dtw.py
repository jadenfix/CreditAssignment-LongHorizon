"""Differentiable soft-DTW (Cuturi & Blondel, 2017) in JAX.

Used as the ablation-A3 comparator to the OT-based alignment loss. Handles
variable-length sequences under monotone temporal warping, which Sinkhorn does
not enforce (Sinkhorn allows any transport, possibly non-monotone).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _softmin(a: jax.Array, b: jax.Array, c: jax.Array, gamma: float) -> jax.Array:
    m = jnp.minimum(jnp.minimum(a, b), c)
    z = jnp.exp(-(a - m) / gamma) + jnp.exp(-(b - m) / gamma) + jnp.exp(-(c - m) / gamma)
    return m - gamma * jnp.log(z)


def soft_dtw_pair(
    x: jax.Array,  # [T_x, D]
    y: jax.Array,  # [T_y, D]
    gamma: float = 1.0,
) -> jax.Array:
    """Soft-DTW distance between two sequences under squared-Euclidean cost."""
    Tx, Ty = x.shape[0], y.shape[0]
    # Pairwise squared distances
    C = jnp.sum(x * x, axis=-1, keepdims=True) + jnp.sum(y * y, axis=-1) - 2.0 * x @ y.T
    C = jnp.maximum(C, 0.0)

    R = jnp.full((Tx + 2, Ty + 2), jnp.inf)
    R = R.at[0, 0].set(0.0)

    def row_step(R, i):
        def col_step(R, j):
            cost = C[i - 1, j - 1]
            val = cost + _softmin(R[i - 1, j - 1], R[i - 1, j], R[i, j - 1], gamma)
            R = R.at[i, j].set(val)
            return R, None

        R, _ = jax.lax.scan(col_step, R, jnp.arange(1, Ty + 1))
        return R, None

    R, _ = jax.lax.scan(row_step, R, jnp.arange(1, Tx + 1))
    return R[Tx, Ty]


def soft_dtw_loss(
    R_a: jax.Array,  # [B, T_a, D]
    R_b: jax.Array,  # [B, T_b, D]
    *,
    gamma: float = 1.0,
) -> jax.Array:
    """Batch mean of pairwise soft-DTW distances."""
    return jax.vmap(lambda x, y: soft_dtw_pair(x, y, gamma=gamma))(R_a, R_b).mean()
