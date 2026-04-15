"""GSM8K loader. Pulls raw math problems via HF ``datasets``; emits *problem
shells* only — i.e. x plus gold answer. Use ``teacher.cli`` to generate (y⁰, f,
y¹) from a fixed base model + teacher, writing into a replay shard.
"""

from __future__ import annotations

from collections.abc import Iterator

from .schema import Quadruple, QuadrupleMeta


def load_gsm8k_problems(split: str = "train", limit: int | None = None) -> Iterator[dict]:
    try:
        from datasets import load_dataset
    except Exception as e:  # pragma: no cover
        raise ImportError("GSM8K loader needs `cts[hf]`.") from e
    ds = load_dataset("gsm8k", "main", split=split)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield {
            "task_id": f"gsm8k_{split}_{i}",
            "question": row["question"],
            "answer": row["answer"],
        }


def gold_answer_map(split: str = "test", limit: int | None = None) -> dict[str, str]:
    """Return ``{task_id: final_number}`` for the math verifier."""
    import re

    out: dict[str, str] = {}
    for p in load_gsm8k_problems(split, limit):
        m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", p["answer"])
        if m:
            out[p["task_id"]] = m.group(1)
    return out


def empty_quadruple_from_problem(p: dict) -> Quadruple:
    """Shell Quadruple with empty y0/f/y1 — to be filled by the teacher pass."""
    return Quadruple(
        x=p["question"],
        y0="",
        f="",
        y1="",
        meta=QuadrupleMeta(domain="math", task_id=p["task_id"]),
    )
