"""Statistical tooling — paired bootstrap, Wilcoxon, Holm, Cohen's κ.

Kept dependency-light (numpy only) so CI doesn't need SciPy. Exact-tie
handling on Wilcoxon uses the mid-rank convention.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap the mean of ``b - a`` with paired resampling."""
    x = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = x[idx].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return {"mean": float(x.mean()), "lo": lo, "hi": hi, "n": n}


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    """Two-sided Wilcoxon signed-rank test, large-sample normal approximation."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return {"W": 0.0, "z": 0.0, "p": 1.0, "n": 0}
    abs_d = np.abs(d)
    order = np.argsort(abs_d, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    # mid-rank for ties
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_d[order[j + 1]] == abs_d[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        ranks[order[i : j + 1]] = avg
        i = j + 1
    signed = np.where(d > 0, ranks, -ranks)
    W = float(signed.sum())
    mean_W = 0.0
    var_W = n * (n + 1) * (2 * n + 1) / 6.0
    z = W / (var_W**0.5) if var_W > 0 else 0.0
    # two-sided p via standard normal
    from math import erf, sqrt

    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2))))
    return {"W": W, "z": float(z), "p": float(p), "n": n}


def holm_bonferroni(pvals: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Return a list of reject/accept decisions under Holm correction."""
    m = len(pvals)
    order = np.argsort(pvals)
    reject = [False] * m
    for rank, idx in enumerate(order):
        thresh = alpha / (m - rank)
        if pvals[idx] <= thresh:
            reject[idx] = True
        else:
            break
    return reject


def cohens_kappa(rater_a: Sequence[int], rater_b: Sequence[int]) -> float:
    """Cohen's κ for agreement between two raters over discrete labels."""
    a = np.asarray(rater_a)
    b = np.asarray(rater_b)
    if a.size == 0:
        return 0.0
    labels = np.unique(np.concatenate([a, b]))
    n = a.size
    po = float((a == b).mean())
    pe = 0.0
    for lab in labels:
        pe += (a == lab).mean() * (b == lab).mean()
    if pe == 1.0:
        return 1.0
    return float((po - pe) / (1.0 - pe))
