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


def test_cts_fixture_rollout_trains(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cts.train.main",
            "method=cts",
            "task=fixtures_math",
            "backend=local_nano",
            "trainer.max_steps=1",
            "trainer.warmup_steps=0",
            "trainer.batch_size=2",
            "method_cfg.group_size=3",
            "method_cfg.max_new_tokens=8",
            "method_cfg.max_len=48",
            "method_cfg.alignment.num_iters=5",
            "method_cfg.alignment.use_ott=false",
            f"out_dir={tmp_path}",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    receipt = json.loads((tmp_path / "training-receipt.json").read_text())
    assert receipt["method"] == "cts"
    assert receipt["steps"] == 1
    assert receipt["parameter_delta_l1"] > 0
    assert receipt["fixture_rollout"]["reward_id"] == "fixture_math_final_number_v1"
    assert receipt["fixture_rollout"]["sampled_rollout_shape"] == [2, 1, 8]
    assert receipt["fixture_rollout"]["candidate_group_shape"] == [2, 3, 8]
    assert receipt["fixture_rollout"]["reward_min"] == 0.0
    assert receipt["fixture_rollout"]["reward_max"] == 1.0
    assert receipt["scope"] == "tiny random fixture on CPU; no policy-quality or scale claim"
