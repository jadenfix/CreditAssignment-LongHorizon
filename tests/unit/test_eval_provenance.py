import pytest

from cts.eval.run_eval import evaluate

FIXTURE = "src/cts/data/fixtures/gsm8k_tiny.jsonl"


def test_missing_references_cannot_be_reported_as_failures():
    with pytest.raises(ValueError, match="complete gold"):
        evaluate(FIXTURE, "test", "math")


def test_empty_split_rejected():
    with pytest.raises(ValueError, match="no examples"):
        evaluate(FIXTURE, "nonexistent", "math", audited_labels=True)


def test_explicit_audit_diagnostics_keep_real_count():
    result = evaluate(FIXTURE, "test", "math", audited_labels=True)
    assert result["headline"]["n"] == 2
    assert result["headline"]["y1_pass1"] == 1.0
    assert result["bootstrap"]["n"] == 2
