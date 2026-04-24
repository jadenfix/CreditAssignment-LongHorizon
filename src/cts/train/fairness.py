"""Fairness-control guard. A sweep across methods must match on all of these
fields or the loader refuses to launch — this is the single biggest reviewer
attack surface for any CTS-vs-baseline claim.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
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
    # Hash of the replay shards backing this cell. None if the task path is
    # missing on disk (smoke runs) — otherwise a dict keyed by relative path.
    # Without this, swapping a test.jsonl mid-matrix is invisible to the gate.
    replay_shard_hashes: tuple[tuple[str, str], ...] | None = None


def _hash_replay_path(replay_path: str | Path) -> tuple[tuple[str, str], ...] | None:
    """Return a stable tuple of (relpath, sha256-hex) for the shard(s) at ``replay_path``.

    Priority: if a sibling ``MANIFEST.json`` exists (produced by the teacher
    upload step), use it verbatim so the fingerprint matches what was uploaded
    to GCS. Otherwise hash the files on disk. Returns ``None`` if the path is
    absent (smoke / missing-data case — treat as a non-gate rather than an
    error so matrix pre-flight works on controller nodes without data).
    """
    p = Path(replay_path)
    if not p.exists():
        return None

    # Prefer a manifest if present — matches the uploaded GCS artifact.
    manifest_candidates: list[Path] = []
    if p.is_dir():
        manifest_candidates.append(p / "MANIFEST.json")
        manifest_candidates.append(p.parent / "MANIFEST.json")
    else:
        manifest_candidates.append(p.parent / "MANIFEST.json")
    for mp in manifest_candidates:
        if mp.exists():
            try:
                data = json.loads(mp.read_text())
                if isinstance(data, dict) and data:
                    return tuple(sorted((k, str(v)) for k, v in data.items()))
            except (OSError, json.JSONDecodeError):
                pass

    # Fall back to live file hashing.
    files: list[Path] = []
    if p.is_dir():
        files = sorted(p.rglob("*.jsonl"))
    elif p.is_file():
        files = [p]
    if not files:
        return None
    root = p if p.is_dir() else p.parent
    out: list[tuple[str, str]] = []
    for f in files:
        h = hashlib.sha256()
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
        try:
            rel = str(f.relative_to(root))
        except ValueError:
            rel = f.name
        out.append((rel, h.hexdigest()))
    return tuple(sorted(out))


def fingerprint_from_cfg(cfg: Any) -> FairnessFingerprint:
    f = cfg.fairness
    replay_path = None
    try:
        replay_path = cfg.task.replay_path
    except Exception:
        pass
    replay_hashes = _hash_replay_path(replay_path) if replay_path else None
    return FairnessFingerprint(
        base_ckpt_hash=f.get("base_ckpt_hash"),
        total_train_tokens=f.get("total_train_tokens"),
        total_opt_steps=f.get("total_opt_steps"),
        prompt_template_hash=f.get("prompt_template_hash"),
        decode_temperature=float(f.get("decode_temperature", 1.0)),
        verifier_id=f.get("verifier_id"),
        revision_budget=int(f.get("revision_budget", 1)),
        test_compute_budget=f.get("test_compute_budget"),
        replay_shard_hashes=replay_hashes,
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
