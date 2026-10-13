from __future__ import annotations

import jax.numpy as jnp
import pytest

from cts.backends.nano_lm import NanoLM, NanoLMConfig
from cts.data.replay import ReplayShard
from cts.train.fixture_rl import FIXTURE_REWARD_ID, build_fixture_math_grpo_batch
from cts.utils.tokenizer import ByteTokenizer

FIX = "src/cts/data/fixtures/gsm8k_tiny.jsonl"


def test_fixture_group_contains_replay_pair_and_real_samples():
    model = NanoLM(
        NanoLMConfig(
            vocab_size=128,
            hidden_size=16,
            num_layers=1,
            num_heads=2,
            max_seq_len=256,
        ),
        seed=0,
    )
    tokenizer = ByteTokenizer()
    quads = ReplayShard(FIX).load_split("train")[:2]

    batch, receipt = build_fixture_math_grpo_batch(
        model,
        quads,
        tokenizer,
        group_size=3,
        max_new_tokens=8,
        max_prompt_tokens=248,
        temperature=1.0,
        seed=7,
        reward_kind="verifier",
    )

    assert batch.completion_ids.shape == (2, 3, 8)
    assert batch.logprobs_old.shape == batch.mask.shape == (2, 3, 8)
    assert jnp.all(jnp.isfinite(batch.logprobs_old))
    assert jnp.array_equal(batch.rewards[:, 0], jnp.zeros((2,)))
    assert jnp.array_equal(batch.rewards[:, 1], jnp.ones((2,)))
    assert receipt.reward_id == FIXTURE_REWARD_ID
    assert receipt.sampled_rollouts_per_prompt == 1
    assert receipt.sampled_rollout_shape == (2, 1, 8)


def test_fixture_group_rejects_missing_sample_slot():
    model = NanoLM(NanoLMConfig(), seed=0)
    quads = ReplayShard(FIX).load_split("train")[:1]
    with pytest.raises(ValueError, match="y0, y1, and at least one"):
        build_fixture_math_grpo_batch(
            model,
            quads,
            ByteTokenizer(),
            group_size=2,
            max_new_tokens=8,
            max_prompt_tokens=128,
            temperature=1.0,
            seed=0,
            reward_kind="verifier",
        )
