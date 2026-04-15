"""Error-taxonomy utilities. The taxonomy itself lives in :mod:`cts.data.schema`."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from .schema import ErrorTag, Quadruple

STRUCTURAL_TAGS: frozenset[ErrorTag] = frozenset(
    {
        ErrorTag.MISSING_CASE,
        ErrorTag.WRONG_DECOMPOSITION,
        ErrorTag.STATE_TRACKING,
        ErrorTag.INVALID_ASSUMPTION,
    }
)
LOCAL_TAGS: frozenset[ErrorTag] = frozenset(
    {ErrorTag.LOCAL_ARITHMETIC, ErrorTag.TOOL_MISUSE, ErrorTag.STYLE_ONLY}
)


def is_structural(q: Quadruple) -> bool:
    return any(t in STRUCTURAL_TAGS for t in q.meta.error_tags)


def is_local(q: Quadruple) -> bool:
    return bool(q.meta.error_tags) and all(t in LOCAL_TAGS for t in q.meta.error_tags)


def tag_histogram(quads: Iterable[Quadruple]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for q in quads:
        for t in q.meta.error_tags:
            c[t.value] += 1
    return dict(c)
