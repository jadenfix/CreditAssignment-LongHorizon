#!/usr/bin/env python3
"""Verify CTS fixture provenance, receipts, and published media."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/demo/source-manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob(path: str) -> str:
    return subprocess.run(
        ["git", "hash-object", path],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _probe(path: Path) -> dict:
    output = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-show_entries",
            "stream=codec_name,width,height,nb_frames,pix_fmt",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(output)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    for path, expected in manifest["source_blobs"].items():
        actual = _git_blob(path)
        if actual != expected:
            raise SystemExit(f"source drift: {path}: expected {expected}, got {actual}")
    for path, expected in manifest["artifact_sha256"].items():
        actual = _sha256(ROOT / path)
        if actual != expected:
            raise SystemExit(f"artifact drift: {path}: expected {expected}, got {actual}")

    receipt = json.loads((ROOT / "docs/demo/training-receipt.json").read_text())
    assert receipt["steps"] == 2
    assert math.isfinite(receipt["loss"])
    assert receipt["parameter_delta_l1"] > 0
    assert receipt["generated_token_shape"] == [2, 2]
    assert receipt["fixture_rollout"]["candidate_group_shape"] == [2, 3, 8]
    assert receipt["fixture_rollout"]["sampled_rollout_shape"] == [2, 1, 8]
    assert receipt["fixture_rollout"]["reward_min"] == 0.0
    assert receipt["fixture_rollout"]["reward_max"] == 1.0
    assert "no policy-quality or scale claim" in receipt["scope"]

    logs = [
        json.loads(line)
        for line in (ROOT / "docs/demo/train.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [item["step"] for item in logs] == [0, 1]
    assert all(math.isfinite(item["loss"]) for item in logs)

    gif = _probe(ROOT / "docs/demo/cts-fixture.gif")
    mp4 = _probe(ROOT / "docs/demo/cts-fixture.mp4")
    gif_stream, mp4_stream = gif["streams"][0], mp4["streams"][0]
    assert (gif_stream["codec_name"], gif_stream["width"], gif_stream["height"]) == (
        "gif",
        960,
        540,
    )
    assert (mp4_stream["codec_name"], mp4_stream["width"], mp4_stream["height"]) == (
        "h264",
        1280,
        720,
    )
    assert gif_stream["nb_frames"] == mp4_stream["nb_frames"] == "60"
    assert math.isclose(float(gif["format"]["duration"]), 10.0)
    assert math.isclose(float(mp4["format"]["duration"]), 10.0)
    print("verified: source blobs, receipts, 60-frame GIF, and 60-frame MP4")


if __name__ == "__main__":
    main()
