"""Shared tokenized-batch helper for method step functions.

Keeps every method honest about the prompt template (fairness control
``prompt_template_hash`` depends on this function being the single source of
truth).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ..data.schema import Quadruple
from ..utils.tokenizer import BOS, EOS, PAD, SEP, ByteTokenizer

_PROMPT_TEMPLATE = (
    "PROB: {x}\n"
    "ATT0: {y0}\n"
    "CRIT: {f}\n"
    "REV1:"
)


def prompt_template_hash() -> str:
    return hashlib.sha256(_PROMPT_TEMPLATE.encode()).hexdigest()[:12]


def render_prompt(q: Quadruple) -> str:
    return _PROMPT_TEMPLATE.format(x=q.x, y0=q.y0, f=q.f)


@dataclass
class TokenizedBatch:
    input_ids: jnp.ndarray     # [B, T]  prompt + y1 tokens
    target_ids: jnp.ndarray    # [B, T]  next-token targets (shift of input_ids)
    y1_mask: jnp.ndarray       # [B, T]  1 on y1 positions, 0 elsewhere
    prompt_lens: jnp.ndarray   # [B]     length of prompt region in input_ids
    total_lens: jnp.ndarray    # [B]


def encode_batch(
    quads: list[Quadruple],
    tokenizer: ByteTokenizer,
    *,
    max_len: int = 320,
) -> TokenizedBatch:
    B = len(quads)
    input_ids = np.full((B, max_len), PAD, dtype=np.int32)
    target_ids = np.full((B, max_len), PAD, dtype=np.int32)
    y1_mask = np.zeros((B, max_len), dtype=np.int32)
    prompt_lens = np.zeros((B,), dtype=np.int32)
    total_lens = np.zeros((B,), dtype=np.int32)

    for i, q in enumerate(quads):
        prompt = render_prompt(q)
        p_ids = [BOS] + tokenizer.encode(prompt) + [SEP]
        y_ids = tokenizer.encode(q.y1) + [EOS]
        ids = p_ids + y_ids
        ids = ids[:max_len]
        n = len(ids)
        input_ids[i, :n] = ids
        # next-token targets
        if n >= 2:
            target_ids[i, : n - 1] = ids[1:n]
        # mask positions whose *target* lives inside y_ids (i.e. positions at and after len(p_ids)-1)
        y_start = max(0, len(p_ids) - 1)
        y_end = min(n - 1, len(p_ids) - 1 + len(y_ids))
        if y_end > y_start:
            y1_mask[i, y_start:y_end] = 1
        prompt_lens[i] = min(len(p_ids), n)
        total_lens[i] = n

    return TokenizedBatch(
        input_ids=jnp.asarray(input_ids),
        target_ids=jnp.asarray(target_ids),
        y1_mask=jnp.asarray(y1_mask),
        prompt_lens=jnp.asarray(prompt_lens),
        total_lens=jnp.asarray(total_lens),
    )
