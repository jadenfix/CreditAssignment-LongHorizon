"""CLI: stream problem shells → teacher → replay shard.

Example::

    python -m cts.data.teacher.cli --task gsm8k --split train \
        --teacher anthropic --out artifacts/replay/gsm8k/train.jsonl --limit 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..replay import write_jsonl
from ..schema import Quadruple, QuadrupleMeta


def _load_problems(task: str, split: str, limit: int):
    if task == "gsm8k":
        from ..gsm8k import load_gsm8k_problems

        return load_gsm8k_problems(split, limit), "math"
    if task == "apps":
        from ..apps import load_apps_problems

        return load_apps_problems(split, limit), "code"
    if task == "mbpp":
        from ..mbpp import load_mbpp_problems

        return load_mbpp_problems(split, limit), "code"
    raise ValueError(f"Unknown task {task}")


def _make_teacher(name: str):
    if name == "anthropic":
        from .anthropic import AnthropicTeacher

        return AnthropicTeacher()
    if name == "openai":
        from .openai import OpenAITeacher

        return OpenAITeacher()
    raise ValueError(f"Unknown teacher {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["gsm8k", "apps", "mbpp"], required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--teacher", choices=["anthropic", "openai"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    problems, domain = _load_problems(args.task, args.split, args.limit)
    teacher = _make_teacher(args.teacher)

    out_quads = []
    for p in problems:
        q = teacher.generate(p["question"], domain=domain)
        out_quads.append(
            Quadruple(
                x=p["question"],
                y0=q.y0,
                f=q.f,
                y1=q.y1,
                meta=QuadrupleMeta(
                    domain=domain,
                    task_id=p["task_id"],
                    split=args.split,
                    teacher=args.teacher,
                ),
            )
        )
    n = write_jsonl(Path(args.out), out_quads)
    print(f"Wrote {n} quadruples to {args.out}")


if __name__ == "__main__":
    main()
