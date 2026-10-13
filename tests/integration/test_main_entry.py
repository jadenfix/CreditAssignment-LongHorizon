"""Smoke the train/main.py entry: SFT + DPO on nano-LM + fixture replay."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest


@pytest.mark.parametrize("method", ["sft", "dpo"])
def test_main_entry_runs(method: str, tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cts.train.main",
            f"method={method}",
            "task=fixtures_math",
            "backend=local_nano",
            "trainer.max_steps=1",
            "trainer.batch_size=2",
            f"out_dir={tmp_path}",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "loss=" in proc.stdout or "step0" in proc.stdout

    receipt = json.loads((tmp_path / "training-receipt.json").read_text())
    assert receipt["steps"] == 1
    assert receipt["parameter_delta_l1"] > 0
    assert receipt["generated_token_shape"] == [2, 2]
    assert (tmp_path / "train.jsonl").is_file()


def test_rollout_stub_is_not_success():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cts.train.main",
            "method=cts",
            "task=fixtures_math",
            "backend=local_nano",
            "trainer.max_steps=1",
            "trainer.batch_size=2",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode != 0
    assert "task-specific rollout/reward" in proc.stderr
