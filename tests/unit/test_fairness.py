import pytest
from omegaconf import OmegaConf

from cts.train.fairness import assert_consistent, fingerprint_from_cfg


def _mk(steps: int):
    return OmegaConf.create(
        {
            "fairness": {
                "base_ckpt_hash": "x",
                "total_train_tokens": 1000,
                "total_opt_steps": steps,
                "prompt_template_hash": "h",
                "decode_temperature": 1.0,
                "verifier_id": "v",
                "revision_budget": 1,
                "test_compute_budget": 64,
            }
        }
    )


def test_consistent_ok():
    a = fingerprint_from_cfg(_mk(100))
    b = fingerprint_from_cfg(_mk(100))
    assert_consistent([("a", a), ("b", b)])


def test_inconsistent_raises():
    a = fingerprint_from_cfg(_mk(100))
    b = fingerprint_from_cfg(_mk(200))
    with pytest.raises(RuntimeError):
        assert_consistent([("a", a), ("b", b)])
