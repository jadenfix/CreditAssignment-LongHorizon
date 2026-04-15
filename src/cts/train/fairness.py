"""Fairness-control guard. A sweep across methods must match on all of these
fields or the loader refuses to launch — this is the single biggest reviewer
attack surface for any CTS-vs-baseline claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FairnessFingerprint:
    base_ckpt_hash: str | None
    total_train_tokens: int | None
    total_opt_steps: int | None
    prompt_template_hash: str | None
    decode_temperature: float
    verifier_id: str | None
    revision_budget: int
    test_compute_budget: int | None


def fingerprint_from_cfg(cfg: Any) -> FairnessFingerprint:
    f = cfg.fairness
    return FairnessFingerprint(
        base_ckpt_hash=f.get("base_ckpt_hash"),
        total_train_tokens=f.get("total_train_tokens"),
        total_opt_steps=f.get("total_opt_steps"),
        prompt_template_hash=f.get("prompt_template_hash"),
        decode_temperature=float(f.get("decode_temperature", 1.0)),
        verifier_id=f.get("verifier_id"),
        revision_budget=int(f.get("revision_budget", 1)),
        test_compute_budget=f.get("test_compute_budget"),
    )


def assert_consistent(fingerprints: list[tuple[str, FairnessFingerprint]]) -> None:
    """Raise unless every method in a sweep has the same fingerprint."""
    if not fingerprints:
        return
    ref_name, ref = fingerprints[0]
    for name, fp in fingerprints[1:]:
        if fp != ref:
            raise RuntimeError(
                "Fairness guard tripped.\n"
                f"  {ref_name}: {ref}\n"
                f"  {name}: {fp}\n"
                "All methods in a sweep must share compute / prompt / verifier settings."
            )
