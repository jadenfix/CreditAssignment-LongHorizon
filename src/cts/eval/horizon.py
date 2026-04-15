"""Horizon and structural-size bucketizers."""

from __future__ import annotations

from collections.abc import Sequence

from ..data.schema import Quadruple


def bucketize(value: float | int | None, edges: Sequence[float]) -> int:
    if value is None:
        return -1
    for i, e in enumerate(edges):
        if value < e:
            return i
    return len(edges)


def bucket_horizon(q: Quadruple, edges: Sequence[int] = (2, 4, 8)) -> int:
    return bucketize(q.meta.horizon, edges)


def bucket_edit_distance(q: Quadruple, edges: Sequence[int] = (2, 5, 10)) -> int:
    return bucketize(q.meta.edit_distance, edges)
