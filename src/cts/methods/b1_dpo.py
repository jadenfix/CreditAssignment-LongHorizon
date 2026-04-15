"""B1 — DPO on (preferred = y1, dispreferred = y0) pairs.

Reference is the current model with parameters stopped (a fixed snapshot in
real runs; for the local smoke path we approximate by using ``stop_gradient``
on a cloned forward — adequate for shape/gradient tests, upgrade in M8 when
the Tunix DPO trainer is wired in).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from ..backends.local_api import LocalModelAPI
from ..data.schema import Quadruple
from ..utils.tokenizer import BOS, EOS, PAD, SEP, ByteTokenizer
from ._batch import render_prompt

NAME = "dpo"


@dataclass
class DPOCfg:
    beta: float = 0.1
    max_len: int = 128


def _encode_pref(
    quads: list[Quadruple], tokenizer: ByteTokenizer, max_len: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns (ids_chosen, mask_chosen, ids_rejected, mask_rejected) [B, T]."""
    import numpy as np

    B = len(quads)
    ids_c = np.full((B, max_len), PAD, dtype=np.int32)
    ids_r = np.full((B, max_len), PAD, dtype=np.int32)
    m_c = np.zeros((B, max_len), dtype=np.int32)
    m_r = np.zeros((B, max_len), dtype=np.int32)
    for i, q in enumerate(quads):
        prompt = render_prompt(q)
        p_ids = [BOS] + tokenizer.encode(prompt) + [SEP]
        for arr, mask, y in ((ids_c, m_c, q.y1), (ids_r, m_r, q.y0)):
            y_ids = tokenizer.encode(y) + [EOS]
            ids = (p_ids + y_ids)[:max_len]
            arr[i, : len(ids)] = ids
            y_start = max(0, len(p_ids) - 1)
            y_end = min(len(ids) - 1, y_start + len(y_ids))
            if y_end > y_start:
                mask[i, y_start:y_end] = 1
    return jnp.asarray(ids_c), jnp.asarray(m_c), jnp.asarray(ids_r), jnp.asarray(m_r)


def _seq_logprob(model: LocalModelAPI, input_ids: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    out = model.forward(input_ids)
    logp = jax.nn.log_softmax(out.logits, axis=-1)
    # next-token targets are shift of input_ids
    tgt = jnp.concatenate([input_ids[:, 1:], jnp.zeros_like(input_ids[:, :1])], axis=-1)
    tok_lp = jnp.take_along_axis(logp, tgt[..., None], axis=-1)[..., 0]
    mask_f = mask.astype(tok_lp.dtype)
    return (tok_lp * mask_f).sum(axis=-1)


def prepare_batch(quads: list[Quadruple], tokenizer, max_len: int = 128):
    return _encode_pref(quads, tokenizer, max_len)


def step(
    model: LocalModelAPI,
    batch: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    cfg: DPOCfg | None = None,
) -> tuple[jnp.ndarray, dict]:
    cfg = cfg or DPOCfg()
    ids_c, m_c, ids_r, m_r = batch
    lp_c = _seq_logprob(model, ids_c, m_c)
    lp_r = _seq_logprob(model, ids_r, m_r)
    # Reference log-probs (stop-grad approximation for local smoke).
    lp_c_ref = jax.lax.stop_gradient(lp_c)
    lp_r_ref = jax.lax.stop_gradient(lp_r)
    logits = cfg.beta * ((lp_c - lp_c_ref) - (lp_r - lp_r_ref))
    loss = -jax.nn.log_sigmoid(logits).mean()
    return loss, {"dpo_loss": loss, "lp_chosen": lp_c.mean(), "lp_rejected": lp_r.mean()}
