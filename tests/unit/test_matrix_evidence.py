import json
import subprocess
import sys


def test_aggregation_uses_actual_seeds_and_labels_archive(tmp_path):
    for label in ["B0", "B3"]:
        for seed in [0, 1]:
            (tmp_path / f"{label}_fixture_seed{seed}.json").write_text(
                json.dumps({"headline": {"delta_mean": float(seed)}})
            )
    result = subprocess.run(
        [sys.executable, "scripts/aggregate_results.py", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "CI (seeds)" in result.stdout
    assert "(n=2)" in result.stdout
    assert "(n=100)" not in result.stdout
    assert "not trained-model performance" in result.stdout


def test_paired_comparison_rejects_different_seed_ids(tmp_path):
    for label, seed in [("B0", 0), ("B3", 1)]:
        (tmp_path / f"{label}_fixture_seed{seed}.json").write_text(
            json.dumps({"headline": {"delta_mean": 0.0}})
        )
    result = subprocess.run(
        [sys.executable, "scripts/aggregate_results.py", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "identical seed IDs" in result.stderr
