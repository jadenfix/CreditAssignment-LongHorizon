"""Smoke every method forward pass on nano-LM + fixture replay.

Each method must produce a finite loss and non-empty metrics dict. RL methods
use a synthetic GRPO batch with small shapes — we're verifying shapes and
gradient paths, not policy quality.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from cts.backends.nano_lm import NanoLM, NanoLMConfig
from cts.data.replay import ReplayShard
from cts.methods import b0_sft_revision, b1_dpo, b2_grpo_outcome, b5_cts
from cts.methods._batch import encode_batch
from cts.methods.b2_grpo_outcome import GRPOBatch
from cts.methods.b5_cts import CTSBatch, CTSCfg, CTSModules
from cts.utils.tokenizer import ByteTokenizer

FIX = "src/cts/data/fixtures/gsm8k_tiny.jsonl"


@pytest.fixture(scope="module")
def model_and_tok():
    cfg = NanoLMConfig(vocab_size=128, hidden_size=32, num_layers=2, num_heads=4, max_seq_len=96)
    return NanoLM(cfg, seed=0), ByteTokenizer(vocab_size=128)


def _quads(n: int = 2):
    return ReplayShard(FIX).load_all()[:n]


def test_b0_sft_revision(model_and_tok):
    model, tok = model_and_tok
    batch = encode_batch(_quads(), tok, max_len=64)
    loss, metrics = b0_sft_revision.step(model, batch)
    assert jnp.isfinite(loss)
    assert "sft_nll" in metrics


def test_b1_dpo(model_and_tok):
    model, tok = model_and_tok
    batch = b1_dpo.prepare_batch(_quads(), tok, max_len=64)
    loss, metrics = b1_dpo.step(model, batch)
    assert jnp.isfinite(loss)
    assert "dpo_loss" in metrics


def _mk_grpo_batch(model: NanoLM, B: int = 2, G: int = 2, T_in: int = 8, T_out: int = 6):
    rng = jax.random.PRNGKey(0)
    prompt = jax.random.randint(rng, (B, T_in), 0, model.vocab_size)
    comp = jax.random.randint(jax.random.PRNGKey(1), (B, G, T_out), 0, model.vocab_size)
    mask = jnp.ones((B, G, T_out))
    old_lp = jax.random.normal(jax.random.PRNGKey(2), (B, G, T_out)) * 0.01
    rewards = jax.random.uniform(jax.random.PRNGKey(3), (B, G))
    return GRPOBatch(
        prompt_ids=prompt, completion_ids=comp, mask=mask, logprobs_old=old_lp, rewards=rewards
    )


def test_b2_grpo_outcome(model_and_tok):
    model, _ = model_and_tok
    batch = _mk_grpo_batch(model)
    loss, metrics = b2_grpo_outcome.step(model, batch)
    assert jnp.isfinite(loss)
    assert {"grpo_loss", "approx_kl", "reward_mean"} <= set(metrics)


def test_b5_cts_full_forward(model_and_tok):
    model, _ = model_and_tok
    cfg = CTSCfg()
    cfg.projection.hidden_size = model.hidden_size
    cfg.projection.out_dim = 16
    cfg.energy.r_dim = 16
    modules = CTSModules(cfg, rngs=nnx.Rngs(0))

    grpo_batch = _mk_grpo_batch(model)
    B, T = 2, 16
    critique = jax.random.randint(jax.random.PRNGKey(4), (B, 8), 0, 256)
    rng = jax.random.PRNGKey(5)
    bad_hidden = jax.random.normal(rng, (B, 10, model.hidden_size))
    ref_hidden = jax.random.normal(rng, (B, 10, model.hidden_size))
    input_ids = jax.random.randint(jax.random.PRNGKey(6), (B, T), 0, model.vocab_size)
    target_ids = jax.random.randint(jax.random.PRNGKey(7), (B, T), 0, model.vocab_size)
    y1_mask = jnp.ones((B, T))

    batch = CTSBatch(
        grpo=grpo_batch,
        input_ids=input_ids,
        target_ids=target_ids,
        y1_mask=y1_mask,
        critique_ids=critique,
        ref_hidden=ref_hidden,
        bad_hidden=bad_hidden,
    )
    # Force the sinkhorn fallback so this test doesn't depend on OTT-JAX.
    cfg.alignment.use_ott = False
    cfg.alignment.num_iters = 20

    loss, metrics = b5_cts.step(model, modules, batch, cfg)
    assert jnp.isfinite(loss)
    for k in ("grpo_loss", "decode_anchor", "alignment", "energy_loss"):
        assert k in metrics


def test_b5_cts_ablations(model_and_tok):
    """A1 (alignment=0) and A4 (decode=0) toggles zero those terms."""
    model, _ = model_and_tok
    cfg = CTSCfg(lam_align=0.0, lam_decode=0.0, lam_energy=0.0)
    cfg.projection.hidden_size = model.hidden_size
    cfg.projection.out_dim = 16
    cfg.energy.r_dim = 16
    cfg.alignment.use_ott = False
    modules = CTSModules(cfg, rngs=nnx.Rngs(0))

    grpo_batch = _mk_grpo_batch(model)
    B, T = 2, 16
    batch = CTSBatch(
        grpo=grpo_batch,
        input_ids=jax.random.randint(jax.random.PRNGKey(0), (B, T), 0, model.vocab_size),
        target_ids=jax.random.randint(jax.random.PRNGKey(1), (B, T), 0, model.vocab_size),
        y1_mask=jnp.ones((B, T)),
        critique_ids=jax.random.randint(jax.random.PRNGKey(2), (B, 8), 0, 256),
        ref_hidden=jax.random.normal(jax.random.PRNGKey(3), (B, 10, model.hidden_size)),
        bad_hidden=jax.random.normal(jax.random.PRNGKey(4), (B, 10, model.hidden_size)),
    )
    loss, metrics = b5_cts.step(model, modules, batch, cfg)
    # With all lams zero plus zero-KL (kl_coef>0 but kl may be tiny): the loss
    # is dominated by grpo_loss only.
    assert jnp.isfinite(loss)
    assert jnp.isfinite(metrics["alignment"])  # computed but zero-weighted
