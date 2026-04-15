"""B0 — SFT on the corrected revision. ``-log pi(y1 | x, y0, f)``.

Simplest critique-consumption baseline; everything else must beat this.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..backends.local_api import LocalModelAPI
from ..data.schema import Quadruple
from ..losses.decode_anchor import decode_anchor_loss
from ._batch import TokenizedBatch, encode_batch

NAME = "sft_revision"


def prepare_batch(quads: list[Quadruple], tokenizer, max_len: int = 128) -> TokenizedBatch:
    return encode_batch(quads, tokenizer, max_len=max_len)


def step(model: LocalModelAPI, batch: TokenizedBatch, cfg: Any = None) -> tuple[jnp.ndarray, dict]:
    out = model.forward(batch.input_ids)
    loss = decode_anchor_loss(out.logits, batch.target_ids, batch.y1_mask)
    return loss, {"sft_nll": loss}
