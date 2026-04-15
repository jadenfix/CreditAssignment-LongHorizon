"""Drive the paper's experiment matrix: methods × tasks × seeds.

Produces one results JSONL per cell under ``--out``, with a sibling
``manifest.json`` recording the fairness fingerprint (which must be identical
across cells, per ``cts.train.fairness.assert_consistent``).

Examples
--------
Local smoke (CPU, fixtures, 2 seeds, math only)::

    uv run python scripts/run_paper_matrix.py --backend local_nano \\
        --tasks math --seeds 2 --out artifacts/results/smoke

Full paper run (Tunix, both tasks, 5 seeds)::

    uv run python scripts/run_paper_matrix.py --backend tunix \\
        --tasks both --seeds 5 --out gs://my-bucket/runs/$(date +%F)

The script *itself* does no training; it shells out to ``cts.train.main`` and
``cts.eval.run_eval``. This keeps the matrix orchestrator decoupled from any
algorithmic code.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shlex
import subprocess
import sys
from pathlib import Path

# Method × task matrix. Keep ordering stable — the paper table reads top-down.
METHODS = ["sft", "dpo", "grpo_outcome", "grpo_verifier", "grpo_critique", "cts"]
METHOD_LABEL = {
    "sft": "B0",
    "dpo": "B1",
    "grpo_outcome": "B2",
    "grpo_verifier": "B3",
    "grpo_critique": "B4",
    "cts": "B5",
}
TASK_GROUPS = {
    "math": [("gsm8k", "math")],
    "code": [("apps", "code")],
    "both": [("gsm8k", "math"), ("apps", "code")],
    "smoke": [("fixtures_math", "math")],
}


def _run(cmd: list[str], dry: bool) -> int:
    print("$", " ".join(shlex.quote(c) for c in cmd), flush=True)
    if dry:
        return 0
    return subprocess.run(cmd, check=False).returncode


def _train_cell(method: str, task: str, seed: int, backend: str, max_steps: int, dry: bool) -> int:
    cmd = [
        sys.executable,
        "-m",
        "cts.train.main",
        f"method={method}",
        f"task={task}",
        f"backend={backend}",
        f"seed={seed}",
        f"trainer.max_steps={max_steps}",
    ]
    return _run(cmd, dry)


def _eval_cell(task: str, domain: str, replay: str, out_path: Path, dry: bool) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "cts.eval.run_eval",
        "--replay",
        replay,
        "--split",
        "test",
        "--domain",
        domain,
        "--out",
        str(out_path),
    ]
    return _run(cmd, dry)


def _replay_path_for(task: str) -> str:
    # Mirror the defaults baked into configs/task/*.yaml; keep this list tight.
    table = {
        "fixtures_math": "src/cts/data/fixtures/gsm8k_tiny.jsonl",
        "gsm8k": "artifacts/replay/gsm8k/test.jsonl",
        "apps": "artifacts/replay/apps/test.jsonl",
        "mbpp_smoke": "src/cts/data/fixtures/mbpp_tiny.jsonl",
    }
    return table[task]


def _fingerprint_for(method: str, task: str, backend: str, seed: int) -> dict:
    """Compute the fairness fingerprint a cell *would* use without launching it.

    Imported lazily so the script runs even if cts isn't fully installed (e.g.
    on a controller node that only has the orchestrator).
    """
    from cts.train.fairness import fingerprint_from_cfg
    from cts.train.main import _load_cfg  # type: ignore[attr-defined]

    cfg = _load_cfg([f"method={method}", f"task={task}", f"backend={backend}", f"seed={seed}"])
    return dataclasses.asdict(fingerprint_from_cfg(cfg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--backend", default="local_nano", choices=["local_nano", "local_hf", "tunix", "maxtext"]
    )
    ap.add_argument("--tasks", default="math", choices=list(TASK_GROUPS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument(
        "--methods",
        default=",".join(METHODS),
        help="Comma-separated subset of " + ",".join(METHODS),
    )
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--out", required=True, help="Local dir or gs:// prefix")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--skip-train",
        action="store_true",
        help="Run only the eval pass (assumes training already happened)",
    )
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHODS:
            raise SystemExit(f"Unknown method {m!r}; choose from {METHODS}")
    tasks = TASK_GROUPS[args.tasks]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Fairness check: every cell should share the same fingerprint.
    fps: list[tuple[str, dict]] = []
    for method in methods:
        for task, _ in tasks:
            fp = _fingerprint_for(method, task, args.backend, seed=0)
            fps.append((f"{method}@{task}", fp))
    ref_name, ref_fp = fps[0]
    for name, fp in fps[1:]:
        if fp != ref_fp:
            raise SystemExit(
                f"Fairness fingerprint mismatch:\n  {ref_name}: {ref_fp}\n  {name}: {fp}"
            )
    (out_root / "manifest.json").write_text(
        json.dumps(
            {
                "backend": args.backend,
                "tasks": tasks,
                "seeds": args.seeds,
                "methods": methods,
                "max_steps": args.max_steps,
                "fairness_fingerprint": ref_fp,
            },
            indent=2,
        )
    )

    failures: list[str] = []
    for method in methods:
        for task, domain in tasks:
            for seed in range(args.seeds):
                tag = f"{METHOD_LABEL[method]}_{task}_seed{seed}"
                if not args.skip_train:
                    rc = _train_cell(method, task, seed, args.backend, args.max_steps, args.dry_run)
                    if rc != 0:
                        failures.append(f"train:{tag}")
                        continue
                out_path = out_root / f"{tag}.json"
                replay = _replay_path_for(task)
                rc = _eval_cell(task, domain, replay, out_path, args.dry_run)
                if rc != 0:
                    failures.append(f"eval:{tag}")

    if failures:
        print(f"\n[matrix] {len(failures)} cell(s) failed: {failures}", file=sys.stderr)
        sys.exit(1)
    print(f"\n[matrix] OK — {len(methods) * len(tasks) * args.seeds} cells under {out_root}")


if __name__ == "__main__":
    main()
