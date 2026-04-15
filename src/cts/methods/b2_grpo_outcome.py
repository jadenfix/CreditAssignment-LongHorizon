"""B2 — GRPO with outcome-only reward (pass/fail from verifier).

Local smoke implementation: G rollouts per prompt, group-relative advantage,
clipped PPO-style surrogate with a cheap approx-KL regularizer. The real
training path uses Tunix's GRPO trainer; this file exists so we can verify
gradients, shapes, and Δ_critique end-to-end on nano-LM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from ..backends.local_api import DecodeCfg, LocalModelAPI
from ..rl.grpo import GRPOCfg, approx_kl, group_relative_advantages, grpo_surrogate_loss

NAME = "grpo_outcome"


@dataclass
class GRPOBatch:
    prompt_ids: jnp.ndarray      # [B, T_in]
    completion_ids: jnp.ndarray  # [B, G, T_out]
    mask: jnp.ndarray            # [B, G, T_out]
    logprobs_old: jnp.ndarray    # [B, G, T_out]
    rewards: jnp.ndarray         # [B, G]


def step(
    model: LocalModelAPI, batch: GRPOBatch, cfg: GRPOCfg | None = None
) -> tuple[jnp.ndarray, dict]:
    cfg = cfg or GRPOCfg()
    B, G, T = batch.completion_ids.shape
    full = jnp.concatenate(
        [
            jnp.broadcast_to(batch.prompt_ids[:, None, :], (B, G, batch.prompt_ids.shape[-1])),
            batch.completion_ids,
        ],
        axis=-1,
    ).reshape(B * G, -1)
    out = model.forward(full)
    logp = jax.nn.log_softmax(out.logits, axis=-1)
    # logits at position i predict token i+1. Completion tokens occupy positions
    # [T_in, T_in+T-1] in `full`; their generation logprobs are therefore at
    # prediction positions [T_in-1, T_in+T-2]. Slice those directly.
    tgt = full[:, 1:]                               # [B*G, L-1] — next-token targets
    logp_next = logp[:, :-1, :]                     # [B*G, L-1, V] — aligned predictions
    tok_lp = jnp.take_along_axis(logp_next, tgt[..., None], axis=-1)[..., 0]  # [B*G, L-1]
    comp_slice = tok_lp[:, -T:].reshape(B, G, T)

    A = group_relative_advantages(batch.rewards, cfg.eps)
    loss = grpo_surrogate_loss(comp_slice, batch.logprobs_old, batch.mask, A, cfg)
    kl = approx_kl(comp_slice, batch.logprobs_old, batch.mask)
    total = loss + cfg.kl_coef * kl
    return total, {"grpo_loss": loss, "approx_kl": kl, "reward_mean": batch.rewards.mean()}
