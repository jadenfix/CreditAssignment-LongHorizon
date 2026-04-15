"""Energy critic ``E_psi(r_t, e_f)`` + critique-localized edit ``R* = R - η ∇_R U_psi``.

Given a projected trajectory ``R = [r_1, ..., r_T]`` and a critique embedding
``e_f``, ``E_psi`` scores "badness" of each step against the critique. The
edit step performs one gradient descent move in latent space to produce a
pseudo-target ``R*`` that CTS aligns toward. The critic also supplies a
contrastive training signal (see :mod:`cts.losses.energy_loss`).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx


@dataclass
class EnergyCfg:
    r_dim: int = 32
    f_dim: int = 32
    hidden: int = 64
    edit_eta: float = 0.1
    freeze: bool = False  # ablation A5


class EnergyCritic(nnx.Module):
    def __init__(self, cfg: EnergyCfg, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.fc1 = nnx.Linear(cfg.r_dim + cfg.f_dim, cfg.hidden, rngs=rngs)
        self.fc2 = nnx.Linear(cfg.hidden, cfg.hidden, rngs=rngs)
        self.fc_out = nnx.Linear(cfg.hidden, 1, rngs=rngs)

    def score(self, r_t: jax.Array, e_f: jax.Array) -> jax.Array:
        # r_t: [..., r_dim], e_f: [..., f_dim] — broadcastable over a time axis
        x = jnp.concatenate([r_t, e_f], axis=-1)
        x = jax.nn.gelu(self.fc1(x))
        x = jax.nn.gelu(self.fc2(x))
        return self.fc_out(x)[..., 0]  # scalar per step

    def U(self, R: jax.Array, e_f: jax.Array, alpha: jax.Array | None = None) -> jax.Array:
        """``U(R, f) = Σ_t α_t E_psi(r_t, e_f)``.

        ``R: [B, T, D_r]``, ``e_f: [B, D_f]`` broadcast to each step.
        """
        T = R.shape[-2]
        ef = jnp.broadcast_to(e_f[..., None, :], R.shape[:-1] + (e_f.shape[-1],))
        e_per_step = self.score(R, ef)  # [B, T]
        if alpha is None:
            alpha = jnp.ones_like(e_per_step) / T
        return jnp.sum(alpha * e_per_step, axis=-1)


def edit_trajectory(
    critic: EnergyCritic,
    R: jax.Array,
    e_f: jax.Array,
    *,
    eta: float | None = None,
    alpha: jax.Array | None = None,
) -> jax.Array:
    """Return ``R* = R - η ∇_R U(R, f)`` — one step of latent editing.

    Gradient is taken *through R only*. The A5 freeze ablation is implemented
    at the call site (see ``cts.methods.b5_cts._critic_view``): callers pass
    in a view of the critic whose parameters have ``jax.lax.stop_gradient``
    applied, so the critic's params are still used in the forward pass but
    receive no gradient. Toggling ``EnergyCfg.freeze`` alone has no effect —
    the flag is read by the B5 step, which builds the frozen view.
    """
    step = critic.cfg.edit_eta if eta is None else eta

    def U_of_R(R_in: jax.Array) -> jax.Array:
        return jnp.sum(critic.U(R_in, e_f, alpha))

    grad_R = jax.grad(U_of_R)(R)
    return R - step * grad_R
