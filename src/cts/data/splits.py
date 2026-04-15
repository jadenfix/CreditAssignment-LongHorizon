"""Deterministic train/val/test/transfer splits keyed on a frozen hash of the prompt.

The test-set hash is expected to be committed into the repo once generated, so
any accidental redistribution of examples across splits fails loudly.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from .schema import Quadruple

_SPLIT_SALT = "cts-v1"


def _bucket(q: Quadruple, mod: int) -> int:
    h = hashlib.sha256(f"{_SPLIT_SALT}:{q.x}".encode()).hexdigest()
    return int(h[:8], 16) % mod


def assign_splits(
    quads: Sequence[Quadruple],
    *,
    train: float = 0.70,
    val: float = 0.10,
    test: float = 0.20,
) -> list[Quadruple]:
    """Return a new list with ``meta.split`` deterministically assigned.

    Transfer examples are expected to carry ``meta.split='transfer'`` upstream;
    this function preserves them.
    """
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError("train+val+test must sum to 1.0")
    out: list[Quadruple] = []
    t_cut = int(round(train * 1000))
    v_cut = t_cut + int(round(val * 1000))
    for q in quads:
        if q.meta.split == "transfer":
            out.append(q)
            continue
        b = _bucket(q, 1000)
        if b < t_cut:
            split = "train"
        elif b < v_cut:
            split = "val"
        else:
            split = "test"
        new_meta = q.meta.model_copy(update={"split": split})
        out.append(q.model_copy(update={"meta": new_meta}))
    return out
