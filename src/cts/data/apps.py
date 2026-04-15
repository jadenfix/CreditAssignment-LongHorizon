"""APPS loader (primary code domain).

Reads problem descriptions + unit tests from the HF APPS dataset. Tests are
stored per-task for :class:`cts.rl.verifier.CodeVerifier`. As with GSM8K, this
module emits *shells*; (y⁰, f, y¹) come from the teacher CLI.
"""

from __future__ import annotations

from collections.abc import Iterator

from .schema import Quadruple, QuadrupleMeta


def load_apps_problems(split: str = "train", limit: int | None = None) -> Iterator[dict]:
    try:
        from datasets import load_dataset
    except Exception as e:  # pragma: no cover
        raise ImportError("APPS loader needs `cts[hf]`.") from e
    ds = load_dataset("codeparrot/apps", split=split)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield {
            "task_id": f"apps_{split}_{i}",
            "question": row["question"],
            "input_output": row.get("input_output"),
            "difficulty": row.get("difficulty"),
        }


def tests_map(split: str = "test", limit: int | None = None) -> dict[str, list[str]]:
    """Return ``{task_id: [assertion1, assertion2, ...]}`` for the code verifier.

    APPS' ``input_output`` gives ``{inputs, outputs}`` lists per task. We shape
    them into Python ``assert`` strings that exercise a candidate ``solve(...)``
    entry point. Callers may need to adjust per problem format.
    """
    import json

    out: dict[str, list[str]] = {}
    for p in load_apps_problems(split, limit):
        io = p.get("input_output")
        if not io:
            continue
        try:
            io = json.loads(io) if isinstance(io, str) else io
        except Exception:
            continue
        inputs = io.get("inputs", [])
        outputs = io.get("outputs", [])
        asserts: list[str] = []
        for x, y in zip(inputs, outputs, strict=False):
            asserts.append(f"assert str(solve({x!r})).strip() == str({y!r}).strip()")
        if asserts:
            out[p["task_id"]] = asserts
    return out


def empty_quadruple_from_problem(p: dict) -> Quadruple:
    return Quadruple(
        x=p["question"],
        y0="",
        f="",
        y1="",
        meta=QuadrupleMeta(domain="code", task_id=p["task_id"]),
    )
