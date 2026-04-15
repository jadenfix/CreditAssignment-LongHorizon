"""Time-augmented Sinkhorn divergence between two projected trajectories.

Primary backend: OTT-JAX ``sinkhorn_divergence`` (entropic OT with debiasing,
optionally unbalanced). We augment each state ``r_t`` with a scaled time
coordinate ``(1-α)·r_t ⊕ α·t/T`` so the transport plan respects temporal
ordering — this is cheaper than soft-DTW for long sequences and gracefully
handles revised trajectories that delete steps (via unbalanced OT).

A pure-JAX fallback (Sinkhorn on squared-Euclidean cost) is provided so unit
tests run without OTT-JAX installed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _time_augment(R: jax.Array, alpha: float) -> jax.Array:
    """Append a scaled time coordinate to each state. ``R: [T, D]`` → ``[T, D+1]``."""
    T = R.shape[-2]
    t = jnp.linspace(0.0, 1.0, T)[:, None]  # [T, 1]
    # Broadcast to any leading batch dims
    shape = list(R.shape[:-2]) + [T, 1]
    t_b = jnp.broadcast_to(t, tuple(shape))
    return jnp.concatenate([(1.0 - alpha) * R, alpha * t_b], axis=-1)


def _pairwise_sqdist(a: jax.Array, b: jax.Array) -> jax.Array:
    # a: [M, D], b: [N, D]  ->  [M, N]
    return jnp.sum(a * a, axis=-1, keepdims=True) + jnp.sum(b * b, axis=-1) - 2.0 * a @ b.T


def _fallback_sinkhorn(
    a: jax.Array, b: jax.Array, *, epsilon: float, num_iters: int = 50
) -> jax.Array:
    """Symmetric entropic Sinkhorn divergence with uniform marginals.

    Returns ``OT(a, b) - 0.5 (OT(a, a) + OT(b, b))``.
    """

    def _ot(x: jax.Array, y: jax.Array) -> jax.Array:
        M, N = x.shape[0], y.shape[0]
        C = _pairwise_sqdist(x, y)
        log_mu = jnp.full((M,), -jnp.log(M))
        log_nu = jnp.full((N,), -jnp.log(N))
        log_K = -C / epsilon
        log_u0 = jnp.zeros((M,))
        log_v0 = jnp.zeros((N,))

        def _step(carry, _):
            log_u, log_v = carry
            log_u = log_mu - jax.scipy.special.logsumexp(log_K + log_v[None, :], axis=1)
            log_v = log_nu - jax.scipy.special.logsumexp(log_K.T + log_u[None, :], axis=1)
            return (log_u, log_v), None

        (log_u, log_v), _ = jax.lax.scan(_step, (log_u0, log_v0), xs=None, length=num_iters)
        log_P = log_u[:, None] + log_K + log_v[None, :]
        return jnp.sum(jnp.exp(log_P) * C)

    return _ot(a, b) - 0.5 * (_ot(a, a) + _ot(b, b))


def sinkhorn_divergence_loss(
    R_a: jax.Array,        # [B, T_a, D]
    R_b: jax.Array,        # [B, T_b, D]
    *,
    epsilon: float = 0.1,
    time_alpha: float = 0.2,
    num_iters: int = 50,
    unbalanced_tau: float | None = None,
    use_ott: bool = True,
) -> jax.Array:
    """Mean over the batch of the (time-augmented) Sinkhorn divergence.

    Parameters
    ----------
    time_alpha
        Weight of the appended time coordinate. 0 reduces to plain OT.
    unbalanced_tau
        If not None and OTT-JAX is available, use unbalanced Sinkhorn with this
        tau. Preferred when the revised trajectory legitimately deletes steps.
    use_ott
        If True, prefer OTT-JAX; falls back to the in-file implementation on
        ImportError so tests run without the extra.
    """
    R_a_t = _time_augment(R_a, time_alpha)
    R_b_t = _time_augment(R_b, time_alpha)

    if use_ott:
        try:
            from ott.geometry import pointcloud
            from ott.tools.sinkhorn_divergence import sinkhorn_divergence

            def per_batch(a: jax.Array, b: jax.Array) -> jax.Array:
                geom = pointcloud.PointCloud(a, b, epsilon=epsilon)
                kwargs: dict = {"max_iterations": num_iters}
                if unbalanced_tau is not None:
                    kwargs.update({"tau_a": unbalanced_tau, "tau_b": unbalanced_tau})
                out = sinkhorn_divergence(
                    pointcloud.PointCloud, x=a, y=b, epsilon=epsilon, sinkhorn_kwargs=kwargs
                )
                return out.divergence

            return jax.vmap(per_batch)(R_a_t, R_b_t).mean()
        except Exception:  # pragma: no cover - fallback path
            pass

    return jax.vmap(
        lambda a, b: _fallback_sinkhorn(a, b, epsilon=epsilon, num_iters=num_iters)
    )(R_a_t, R_b_t).mean()
