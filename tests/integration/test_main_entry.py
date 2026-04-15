"""Smoke the train/main.py entry: SFT + DPO on nano-LM + fixture replay."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize("method", ["sft", "dpo"])
def test_main_entry_runs(method: str):
    proc = subprocess.run(
        [sys.executable, "-m", "cts.train.main",
         f"method={method}", "task=fixtures_math", "backend=local_nano",
         "trainer.max_steps=1", "trainer.batch_size=2"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "loss=" in proc.stdout or "step0" in proc.stdout
