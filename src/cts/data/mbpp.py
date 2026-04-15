"""MBPP — smoke-only loader.

APPS is the paper's primary code benchmark (see plan). MBPP lives here as a
fast local smoke task: its problems are shorter and tend toward local fixes,
which is useful for unit-testing the training loop but not the core claim.
"""

from __future__ import annotations

from collections.abc import Iterator

from .schema import Quadruple, QuadrupleMeta


def load_mbpp_problems(split: str = "train", limit: int | None = None) -> Iterator[dict]:
    try:
        from datasets import load_dataset
    except Exception as e:  # pragma: no cover
        raise ImportError("MBPP loader needs `cts[hf]`.") from e
    ds = load_dataset("mbpp", split=split)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield {
            "task_id": f"mbpp_{split}_{row['task_id']}",
            "question": row["text"],
            "tests": row.get("test_list") or [],
            "code": row.get("code"),
        }


def tests_map(split: str = "test", limit: int | None = None) -> dict[str, list[str]]:
    return {p["task_id"]: p["tests"] for p in load_mbpp_problems(split, limit) if p["tests"]}


def empty_quadruple_from_problem(p: dict) -> Quadruple:
    return Quadruple(
        x=p["question"],
        y0="",
        f="",
        y1="",
        meta=QuadrupleMeta(domain="code", task_id=p["task_id"]),
    )
