"""Blinded human-eval CSV export.

Produces a CSV of ``(row_id, prompt, critique, candidate_A, candidate_B)``
rows in a deterministically shuffled order; the mapping back to (method_a,
method_b) lives in a sibling ``*.key.json`` file that is NOT shown to raters.
"""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Sequence
from pathlib import Path

from ..data.schema import Quadruple


def export_blinded(
    quads: Sequence[Quadruple],
    candidates_a: Sequence[str],
    candidates_b: Sequence[str],
    *,
    label_a: str,
    label_b: str,
    out_dir: str | Path,
    seed: int = 0,
) -> dict[str, Path]:
    assert len(quads) == len(candidates_a) == len(candidates_b)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows = []
    key = []
    for i, (q, ca, cb) in enumerate(zip(quads, candidates_a, candidates_b, strict=True)):
        swap = rng.random() < 0.5
        left, right = (cb, ca) if swap else (ca, cb)
        rows.append(
            {
                "row_id": i,
                "prompt": q.x,
                "critique": q.f,
                "candidate_left": left,
                "candidate_right": right,
            }
        )
        key.append(
            {
                "row_id": i,
                "left": label_b if swap else label_a,
                "right": label_a if swap else label_b,
            }
        )
    csv_path = out_dir / "human_eval.csv"
    key_path = out_dir / "human_eval.key.json"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")
    return {"csv": csv_path, "key": key_path}
