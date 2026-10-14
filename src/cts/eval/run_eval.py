"""End-to-end eval CLI: load a replay test split, score y⁰ and y¹ with a verifier,
emit Δ_critique + CIs + horizon-bucket breakdown to ``results.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data.replay import ReplayShard
from ..rl.verifier import CodeVerifier, MathVerifier
from .horizon import bucket_horizon
from .metrics import delta_critique
from .stats import paired_bootstrap_ci, wilcoxon_signed_rank


def _score_all(quads, verifier, field: str) -> list[float]:
    return [verifier.score(q.meta.task_id, getattr(q, field)) for q in quads]


def evaluate(
    replay_path: str,
    split: str,
    domain: str,
    gold: dict[str, str] | None = None,
    tests: dict[str, list[str]] | None = None,
    audited_labels: bool = False,
) -> dict:
    quads = ReplayShard(replay_path).load_split(split)
    if not quads:
        raise ValueError("Replay split contains no examples")
    if audited_labels:
        if any(not q.meta.is_human_audited or q.meta.y1_correct is None for q in quads):
            raise ValueError("Audit-label diagnostics require every pair to be human audited")
        y0 = [float(q.meta.y0_correct) for q in quads]
        y1 = [float(q.meta.y1_correct is True) for q in quads]
    else:
        reference = gold if domain == "math" else tests if domain == "code" else None
        if reference is None or any(q.meta.task_id not in reference for q in quads):
            raise ValueError("A complete gold-answer or test-bank map is required for verification")
        verifier = (
            MathVerifier(gold=gold or {}) if domain == "math" else CodeVerifier(tests=tests or {})
        )
        y0 = _score_all(quads, verifier, "y0")
        y1 = _score_all(quads, verifier, "y1")
    headline = delta_critique(quads, y0, y1)
    boot = paired_bootstrap_ci(y0, y1)
    wx = wilcoxon_signed_rank(y0, y1)
    # Horizon buckets
    buckets: dict[int, dict] = {}
    for q, a, b in zip(quads, y0, y1, strict=True):
        k = bucket_horizon(q)
        buckets.setdefault(k, {"y0": [], "y1": []})
        buckets[k]["y0"].append(a)
        buckets[k]["y1"].append(b)
    per_bucket = {
        str(k): delta_critique([q for q in quads if bucket_horizon(q) == k], v["y0"], v["y1"])
        for k, v in buckets.items()
    }
    return {
        "evaluation_kind": "archived_audit_labels" if audited_labels else "verified_archived_pairs",
        "trained_model_evaluated": False,
        "headline": headline,
        "bootstrap": boot,
        "wilcoxon": wx,
        "buckets": per_bucket,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--domain", choices=["math", "code"], required=True)
    ap.add_argument("--gold-json", default=None)
    ap.add_argument("--tests-json", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--audited-labels",
        action="store_true",
        help="Report archived human audit labels; does not verify outputs or evaluate a model",
    )
    args = ap.parse_args()
    gold = json.loads(Path(args.gold_json).read_text()) if args.gold_json else None
    tests = json.loads(Path(args.tests_json).read_text()) if args.tests_json else None
    result = evaluate(args.replay, args.split, args.domain, gold, tests, args.audited_labels)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
