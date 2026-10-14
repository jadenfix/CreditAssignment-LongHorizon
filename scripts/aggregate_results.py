"""Aggregate per-cell eval JSONs into the paper's main results table.

Reads a directory produced by ``run_paper_matrix.py`` (one ``B<idx>_<task>_seed<N>.json``
per cell, plus ``manifest.json``) and emits a markdown table:

| Method | Task | Δ_critique mean | 95% CI | Wilcoxon p vs B3 (Holm) |

Per-bucket horizon breakdown is appended below the headline table.

Uses the existing ``cts.eval.stats`` primitives — does not reimplement.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from cts.eval.stats import holm_bonferroni, paired_bootstrap_ci, wilcoxon_signed_rank

CELL_RE = re.compile(r"^(B\d)_([a-z0-9_]+)_seed(\d+)\.json$")


def _load_cells(root: Path) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in sorted(root.glob("*.json")):
        m = CELL_RE.match(p.name)
        if not m:
            continue
        method, task, seed = m.groups()
        cell = json.loads(p.read_text())
        cell["_seed"] = int(seed)
        out[(method, task)].append(cell)
    for runs in out.values():
        runs.sort(key=lambda cell: cell["_seed"])
    return out


def _delta_means(cells: list[dict]) -> list[float]:
    return [float(c["headline"]["delta_mean"]) for c in cells]


def _format_table(rows: list[dict]) -> str:
    lines = [
        "| Method | Task | Δ_critique mean ± std (seeds) | 95% paired-bootstrap CI (seeds) | Wilcoxon p vs B3 | Holm reject |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['task']} | {r['mean']:+.4f} ± {r['std']:.4f} (n={r['n_seeds']}) "
            f"| [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] (n={r['n_seeds']}) "
            f"| {r['p']:.3g} | {'yes' if r['reject'] else 'no'} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--base", default="B3", help="Comparator method label")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=None, help="Markdown output path; default stdout")
    args = ap.parse_args()

    root = Path(args.results_dir)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        print(
            f"[aggregate] manifest fingerprint: {manifest['fairness_fingerprint']}", file=sys.stderr
        )

    cells = _load_cells(root)
    if not cells:
        raise SystemExit(f"No cell JSONs found under {root}")

    # Build per-(method,task) seed-mean stats and bootstrap CI on instance-level
    # deltas pooled across seeds.
    by_task: dict[str, dict[str, list[float]]] = defaultdict(dict)
    rows: list[dict] = []
    for (method, task), runs in cells.items():
        seed_means = _delta_means(runs)
        # Resample actual independent seed summaries once each. Repeating them
        # invents sample size and understates uncertainty.
        ci = paired_bootstrap_ci([0.0] * len(seed_means), seed_means, n_resamples=2000)
        rows.append(
            {
                "method": method,
                "task": task,
                "mean": mean(seed_means),
                "std": pstdev(seed_means) if len(seed_means) > 1 else 0.0,
                "n_seeds": len(seed_means),
                "ci_lo": ci["lo"],
                "ci_hi": ci["hi"],
                "_seed_means": seed_means,
            }
        )
        by_task[task][method] = seed_means

    # Wilcoxon vs base, Holm-corrected within each task family.
    for task, per_method in by_task.items():
        base = per_method.get(args.base)
        if base is None:
            raise SystemExit(f"Comparator {args.base} missing for task {task}")
        contenders = [m for m in per_method if m != args.base]
        pvals: list[float] = []
        for m in contenders:
            other = per_method[m]
            base_seeds = [r["_seed"] for r in cells[(args.base, task)]]
            other_seeds = [r["_seed"] for r in cells[(m, task)]]
            if base_seeds != other_seeds:
                raise SystemExit(
                    f"Paired comparison requires identical seed IDs: {args.base} and {m} for {task}"
                )
            wx = wilcoxon_signed_rank(base, other)
            pvals.append(wx["p"])
        rejects = holm_bonferroni(pvals, alpha=args.alpha)
        for m, p, rej in zip(contenders, pvals, rejects, strict=True):
            for r in rows:
                if r["method"] == m and r["task"] == task:
                    r["p"] = p
                    r["reject"] = rej
        for r in rows:
            if r["method"] == args.base and r["task"] == task:
                r["p"] = 1.0
                r["reject"] = False

    rows.sort(key=lambda r: (r["task"], r["method"]))
    table = _format_table(rows)

    out_text = (
        f"# CTS replay-pair diagnostics\n\n"
        "These scores describe archived y0/y1 replay pairs, not trained-model performance. "
        "Repeated fixture evaluations are not independent research evidence.\n\n"
        f"Comparator: **{args.base}**, α = {args.alpha} (Holm-corrected within task).\n\n"
        f"{table}\n"
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out_text)
        print(f"[aggregate] wrote {args.out}")
    else:
        print(out_text)


if __name__ == "__main__":
    main()
