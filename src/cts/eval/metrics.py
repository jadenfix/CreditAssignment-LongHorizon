"""Primary metric: Δ_critique = m(y¹) − m(y⁰). Plus the usual outcome metrics."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ..data.schema import Quadruple

ScoreFn = Callable[[str, str], float]  # (task_id, output) -> 0/1


def exact_match(pred: str, gold: str) -> float:
    return float(pred.strip() == gold.strip())


def pass_at_1(results: Sequence[float]) -> float:
    if not results:
        return 0.0
    return sum(results) / len(results)


def delta_critique(
    quads: Sequence[Quadruple],
    y0_scores: Sequence[float],
    y1_scores: Sequence[float],
) -> dict[str, float]:
    """Per-example Δ = score(y1) − score(y0), aggregated to mean and std."""
    assert len(quads) == len(y0_scores) == len(y1_scores)
    deltas = [float(b) - float(a) for a, b in zip(y0_scores, y1_scores, strict=True)]
    n = max(len(deltas), 1)
    mean = sum(deltas) / n
    var = sum((d - mean) ** 2 for d in deltas) / max(n - 1, 1)
    return {
        "delta_mean": mean,
        "delta_std": var**0.5,
        "n": n,
        "y0_pass1": pass_at_1(y0_scores),
        "y1_pass1": pass_at_1(y1_scores),
    }
