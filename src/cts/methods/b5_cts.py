"""B5 — CTS: GRPO + verifier  +  λ₁·alignment  +  λ₂·decode_anchor  +  λ₃·energy.

This is the paper's method. The step function composes:

1. The B3 GRPO surrogate (RL outer loop).
2. A decode anchor on ``y¹`` so latent edits stay grounded in text space.
3. A time-aware latent alignment loss between the model's projected trajectory
   ``R_theta(x)`` and a critique-localized pseudo-target
   ``R* = R - η ∇_R U_psi(R, f)`` (optionally averaged with a reference
   trajectory ``R_ref`` from ``y¹``).
4. A contrastive energy loss that trains ``E_psi`` to score bad critique / trajectory
   pairs higher than good ones.

All ablations A1–A7 are config toggles; see :mod:`cts.configs`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import nnx

from ..backends.local_api import LocalModelAPI
from ..losses.alignment import AlignmentCfg, alignment_loss
from ..losses.decode_anchor import decode_anchor_loss
from ..losses.energy_loss import energy_contrastive_loss
from ..models.energy import EnergyCfg, EnergyCritic, edit_trajectory
from ..models.projection import Projection, ProjectionCfg
from ..rl.grpo import GRPOCfg, approx_kl, group_relative_advantages, grpo_surrogate_loss
from .b2_grpo_outcome import GRPOBatch

NAME = "cts"


@dataclass
class CTSCfg:
    lam_align: float = 1.0   # λ₁
    lam_decode: float = 1.0  # λ₂
    lam_energy: float = 0.1  # λ₃
    edit_eta: float = 0.1
    # Ablations
    use_ref_trajectory: bool = True     # if False, align only to R*
    online_critique: bool = False       # A6: refresh critique from current policy each step
    # Sub-configs
    alignment: AlignmentCfg = field(default_factory=AlignmentCfg)
    projection: ProjectionCfg = field(default_factory=ProjectionCfg)
    energy: EnergyCfg = field(default_factory=EnergyCfg)
    grpo: GRPOCfg = field(default_factory=GRPOCfg)


class CTSModules(nnx.Module):
    """Holds the extra CTS-only parameters (projection + energy critic + critique embed)."""

    def __init__(self, cfg: CTSCfg, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.projection = Projection(cfg.projection, rngs=rngs)
        self.energy = EnergyCritic(cfg.energy, rngs=rngs)
        # Critique is embedded by a small encoder over its token IDs. For the
        # local smoke path we use a learned mean-pool over token embeddings.
        self.crit_emb = nnx.Embed(256, cfg.energy.f_dim, rngs=rngs)

    def embed_critique(self, f_token_ids: jnp.ndarray) -> jnp.ndarray:
        e = self.crit_emb(f_token_ids)  # [B, T_f, f_dim]
        return jnp.mean(e, axis=-2)


@dataclass
class CTSBatch:
    """Everything B5 needs. ``grpo`` comes from B3 upstream; the rest is CTS-specific.

    For alignment, prefer supplying ``bad_input_ids`` / ``ref_input_ids`` so the
    step function recomputes hidden states through the current model — that's
    what gives the backbone a gradient through the transport term. The raw
    ``bad_hidden`` / ``ref_hidden`` tensors are kept as a fallback for tests and
    for backends that materialize hidden states outside the JAX graph (e.g.
    frozen teacher traces); when both are provided, the ``_input_ids`` path
    wins.
    """

    grpo: GRPOBatch
    input_ids: jnp.ndarray                 # [B, T] prompt + y1 (decode anchor)
    target_ids: jnp.ndarray                # [B, T]
    y1_mask: jnp.ndarray                   # [B, T]
    critique_ids: jnp.ndarray              # [B, T_f]
    ref_hidden: jnp.ndarray | None = None  # [B, T_ref, D] teacher-forced y1 hidden
    bad_hidden: jnp.ndarray | None = None  # [B, T_bad, D] teacher-forced y0 hidden
    bad_input_ids: jnp.ndarray | None = None   # [B, T_bad] y0 path token ids
    ref_input_ids: jnp.ndarray | None = None   # [B, T_ref] y1 path token ids


def _critic_view(critic: EnergyCritic, *, freeze: bool) -> EnergyCritic:
    """Return a view of ``critic`` with stop_gradient applied to its parameters
    iff ``freeze=True`` (A5 ablation). Gradients still flow through the inputs,
    so ``edit_trajectory`` keeps working; but no gradient reaches psi itself,
    making the contrastive energy loss a no-op on the critic and the transport
    term stop shaping E_psi.
    """
    if not freeze:
        return critic
    gdef, params, rest = nnx.split(critic, nnx.Param, ...)
    params = jax.lax.stop_gradient(params)
    return nnx.merge(gdef, params, rest)


def step(
    model: LocalModelAPI,
    modules: CTSModules,
    batch: CTSBatch,
    cfg: CTSCfg | None = None,
) -> tuple[jnp.ndarray, dict]:
    cfg = cfg or CTSCfg()

    # (1) GRPO surrogate on rollouts (same path as B3).
    g = batch.grpo
    B, G, T = g.completion_ids.shape
    full = jnp.concatenate(
        [
            jnp.broadcast_to(g.prompt_ids[:, None, :], (B, G, g.prompt_ids.shape[-1])),
            g.completion_ids,
        ],
        axis=-1,
    ).reshape(B * G, -1)
    out_grpo = model.forward(full)
    logp = jax.nn.log_softmax(out_grpo.logits, axis=-1)
    # Align logits→next-token correctly: logit[i] predicts token i+1.
    tgt = full[:, 1:]                       # [B*G, L-1]
    logp_next = logp[:, :-1, :]             # [B*G, L-1, V]
    tok_lp = jnp.take_along_axis(logp_next, tgt[..., None], axis=-1)[..., 0]
    comp_slice = tok_lp[:, -T:].reshape(B, G, T)
    A = group_relative_advantages(g.rewards, cfg.grpo.eps)
    grpo_loss = grpo_surrogate_loss(comp_slice, g.logprobs_old, g.mask, A, cfg.grpo)
    kl = approx_kl(comp_slice, g.logprobs_old, g.mask)

    # (2) Decode anchor on y¹.
    out_decode = model.forward(batch.input_ids)
    anchor = decode_anchor_loss(out_decode.logits, batch.target_ids, batch.y1_mask)

    # (3) Alignment. Project hidden states of the model's current trajectory onto R,
    #     build the critique-localized pseudo-target R*, and align toward it.
    #
    # IMPORTANT: to give the *backbone* a gradient through the alignment term,
    # the bad-path hidden must come from `model.forward(...)` inside this step.
    # We prefer the `bad_input_ids` path when present; otherwise fall back to
    # the pre-materialized `bad_hidden` tensor (which carries no backbone grad).
    e_f = modules.embed_critique(batch.critique_ids)  # [B, f_dim]
    if batch.bad_input_ids is not None:
        bad_hidden = model.forward(batch.bad_input_ids).hidden
    else:
        assert batch.bad_hidden is not None, "CTSBatch needs bad_input_ids or bad_hidden"
        bad_hidden = batch.bad_hidden
    R_theta = modules.projection(bad_hidden)          # [B, T_bad, r_dim]

    # Build R*. Freeze the critic parameters when A5 ablation is on — we still
    # need gradient w.r.t. R (inputs) for the edit step, so we sg the critic's
    # params only, not its output.
    frozen_critic = _critic_view(modules.energy, freeze=cfg.energy.freeze)
    R_star = edit_trajectory(frozen_critic, R_theta, e_f, eta=cfg.edit_eta)
    align_a = alignment_loss(R_theta, jax.lax.stop_gradient(R_star), cfg.alignment)
    if cfg.use_ref_trajectory:
        if batch.ref_input_ids is not None:
            # Ref trajectory is a teacher-forced target; no backbone grad from it.
            ref_hidden = jax.lax.stop_gradient(model.forward(batch.ref_input_ids).hidden)
        else:
            assert batch.ref_hidden is not None, "CTSBatch needs ref_input_ids or ref_hidden"
            ref_hidden = batch.ref_hidden
        R_ref = modules.projection(ref_hidden)
        align_b = alignment_loss(R_theta, jax.lax.stop_gradient(R_ref), cfg.alignment)
        align = 0.5 * (align_a + align_b)
    else:
        align = align_a

    # (4) Energy contrastive loss: bad > good (good = ref trajectory).
    # The energy term is about training E_psi, so detach hidden → no backbone
    # grad here; and when A5 freeze is set, also detach the critic's params so
    # this loss is a no-op on psi.
    R_good_in = jax.lax.stop_gradient(ref_hidden) if cfg.use_ref_trajectory and batch.ref_input_ids is not None \
                else (jax.lax.stop_gradient(batch.ref_hidden) if batch.ref_hidden is not None else None)
    if R_good_in is None:
        # No ref available (cfg.use_ref_trajectory=False and no raw ref_hidden) — skip energy loss.
        e_loss = jnp.asarray(0.0)
    else:
        R_good = modules.projection(R_good_in)
        R_bad = modules.projection(jax.lax.stop_gradient(bad_hidden))
        U_bad = frozen_critic.U(R_bad, e_f)
        U_good = frozen_critic.U(R_good, e_f)
        e_loss = energy_contrastive_loss(U_bad, U_good)

    total = (
        grpo_loss
        + cfg.grpo.kl_coef * kl
        + cfg.lam_decode * anchor
        + cfg.lam_align * align
        + cfg.lam_energy * e_loss
    )
    metrics = {
        "total": total,
        "grpo_loss": grpo_loss,
        "approx_kl": kl,
        "decode_anchor": anchor,
        "alignment": align,
        "energy_loss": e_loss,
        "reward_mean": g.rewards.mean(),
    }
    return total, metrics
