"""Replay-only RL batch construction for the local math fixture lane.

This module is deliberately narrow.  It turns archived ``Quadruple`` records
into a comparison group containing the flawed replay attempt, the corrected
replay revision, and fresh nano-LM samples.  Rewards come from the existing
math final-answer verifier.  It is useful for exercising the GRPO/CTS optimizer
path without a teacher service, but it is not an evaluation of policy quality.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..backends.local_api import LocalModelAPI
from ..data.schema import Quadruple
from ..methods._batch import render_prompt
from ..methods.b2_grpo_outcome import GRPOBatch
from ..rl.critique_reward import critique_reward
from ..rl.rollout import rollout_local
from ..rl.verifier import MathVerifier, extract_final_number
from ..utils.tokenizer import BOS, EOS, PAD, SEP, ByteTokenizer

FIXTURE_REWARD_ID = "fixture_math_final_number_v1"


@dataclass(frozen=True)
class FixtureRolloutReceipt:
    """Non-claiming provenance for a locally built RL comparison group."""

    reward_id: str
    group_size: int
    sampled_rollouts_per_prompt: int
    sampled_rollout_shape: tuple[int, ...]
    candidate_group_shape: tuple[int, ...]
    reward_min: float
    reward_max: float
    reward_mean: float


def _pad_rows(rows: list[list[int]], *, width: int) -> jnp.ndarray:
    out = np.full((len(rows), width), PAD, dtype=np.int32)
    for idx, row in enumerate(rows):
        clipped = row[:width]
        out[idx, : len(clipped)] = clipped
    return jnp.asarray(out)


def _prompt_tokens(
    quads: list[Quadruple], tokenizer: ByteTokenizer, *, max_prompt_tokens: int
) -> jnp.ndarray:
    rows = [[BOS, *tokenizer.encode(render_prompt(q)), SEP] for q in quads]
    width = min(max(len(row) for row in rows), max_prompt_tokens)
    return _pad_rows(rows, width=width)


def _replay_completion(
    text: str, tokenizer: ByteTokenizer, *, max_new_tokens: int
) -> tuple[list[int], list[float]]:
    tokens = [*tokenizer.encode(text), EOS][:max_new_tokens]
    mask = [1.0] * len(tokens)
    if len(tokens) < max_new_tokens:
        missing = max_new_tokens - len(tokens)
        tokens.extend([PAD] * missing)
        mask.extend([0.0] * missing)
    return tokens, mask


def _teacher_forced_logprobs(
    model: LocalModelAPI,
    prompt_ids: jax.Array,
    completion_ids: jax.Array,
) -> jax.Array:
    """Log-probabilities for the fixed comparison group under the old policy."""

    batch_size, group_size, completion_len = completion_ids.shape
    prompt_len = prompt_ids.shape[-1]
    full = jnp.concatenate(
        [
            jnp.broadcast_to(prompt_ids[:, None, :], (batch_size, group_size, prompt_len)),
            completion_ids,
        ],
        axis=-1,
    ).reshape(batch_size * group_size, -1)
    logits = model.forward(full).logits
    next_logprobs = jax.nn.log_softmax(logits[:, :-1], axis=-1)
    targets = full[:, 1:]
    token_logprobs = jnp.take_along_axis(next_logprobs, targets[..., None], axis=-1)[..., 0]
    return token_logprobs[:, -completion_len:].reshape(batch_size, group_size, completion_len)


def _agreement(candidate: jax.Array, target: jax.Array, target_mask: jax.Array) -> jax.Array:
    denom = jnp.maximum(target_mask.sum(), 1.0)
    return ((candidate == target) * target_mask).sum() / denom


def build_fixture_math_grpo_batch(
    model: LocalModelAPI,
    quads: list[Quadruple],
    tokenizer: ByteTokenizer,
    *,
    group_size: int,
    max_new_tokens: int,
    max_prompt_tokens: int,
    temperature: float,
    seed: int,
    reward_kind: str,
    critique_lam: float = 0.3,
) -> tuple[GRPOBatch, FixtureRolloutReceipt]:
    """Build a truthful local comparison group from replay plus fresh samples.

    The first two candidates are the archived ``y0`` and ``y1``.  Remaining
    candidates are sampled from the current nano-LM.  Every candidate is scored
    by the existing math final-number verifier; the critique variant additionally
    uses token agreement with ``y1`` as a transparent fixture-only uptake proxy.
    """

    if not quads:
        raise ValueError("At least one replay quadruple is required")
    if group_size < 3:
        raise ValueError("Fixture RL groups need y0, y1, and at least one sampled rollout")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if max_prompt_tokens < 1:
        raise ValueError("max_prompt_tokens must leave room for at least one prompt token")
    if any(q.meta.domain != "math" for q in quads):
        raise ValueError("The local fixture reward builder supports math quadruples only")

    prompts = _prompt_tokens(quads, tokenizer, max_prompt_tokens=max_prompt_tokens)
    sampled_per_prompt = group_size - 2
    repeated_prompts = jnp.repeat(prompts, sampled_per_prompt, axis=0)
    sampled = rollout_local(
        model,
        repeated_prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        rng=jax.random.key(seed),
    )
    sampled_tokens = sampled.tokens.reshape(len(quads), sampled_per_prompt, max_new_tokens)

    completions = np.full((len(quads), group_size, max_new_tokens), PAD, dtype=np.int32)
    masks = np.zeros((len(quads), group_size, max_new_tokens), dtype=np.float32)
    outcomes = np.zeros((len(quads), group_size), dtype=np.float32)
    uptake = jnp.zeros((len(quads), group_size), dtype=jnp.float32)

    gold = {
        q.meta.task_id: answer for q in quads if (answer := extract_final_number(q.y1)) is not None
    }
    verifier = MathVerifier(gold)

    for batch_idx, quad in enumerate(quads):
        y0_tokens, y0_mask = _replay_completion(quad.y0, tokenizer, max_new_tokens=max_new_tokens)
        y1_tokens, y1_mask = _replay_completion(quad.y1, tokenizer, max_new_tokens=max_new_tokens)
        completions[batch_idx, 0] = y0_tokens
        completions[batch_idx, 1] = y1_tokens
        masks[batch_idx, 0] = y0_mask
        masks[batch_idx, 1] = y1_mask
        outcomes[batch_idx, 0] = verifier.score(quad.meta.task_id, quad.y0)
        outcomes[batch_idx, 1] = verifier.score(quad.meta.task_id, quad.y1)

        target = jnp.asarray(y1_tokens)
        target_mask = jnp.asarray(y1_mask)
        uptake = uptake.at[batch_idx, 0].set(
            _agreement(jnp.asarray(y0_tokens), target, target_mask)
        )
        uptake = uptake.at[batch_idx, 1].set(1.0)

        for sample_idx in range(sampled_per_prompt):
            group_idx = sample_idx + 2
            tokens = np.asarray(sampled_tokens[batch_idx, sample_idx], dtype=np.int32)
            completions[batch_idx, group_idx] = tokens
            masks[batch_idx, group_idx] = 1.0
            decoded = tokenizer.decode(tokens.tolist())
            outcomes[batch_idx, group_idx] = verifier.score(quad.meta.task_id, decoded)
            uptake = uptake.at[batch_idx, group_idx].set(
                _agreement(jnp.asarray(tokens), target, target_mask)
            )

    completion_ids = jnp.asarray(completions)
    mask = jnp.asarray(masks)
    outcome_rewards = jnp.asarray(outcomes)
    if reward_kind == "critique":
        rewards = critique_reward(outcome_rewards, uptake, lam=critique_lam)
    elif reward_kind in {"outcome", "verifier"}:
        rewards = outcome_rewards
    else:
        raise ValueError(f"Unsupported fixture reward kind: {reward_kind!r}")

    old_logprobs = jax.lax.stop_gradient(_teacher_forced_logprobs(model, prompts, completion_ids))
    batch = GRPOBatch(
        prompt_ids=prompts,
        completion_ids=completion_ids,
        mask=mask,
        logprobs_old=old_logprobs,
        rewards=rewards,
    )
    receipt = FixtureRolloutReceipt(
        reward_id=FIXTURE_REWARD_ID,
        group_size=group_size,
        sampled_rollouts_per_prompt=sampled_per_prompt,
        sampled_rollout_shape=tuple(sampled_tokens.shape),
        candidate_group_shape=tuple(completion_ids.shape),
        reward_min=float(rewards.min()),
        reward_max=float(rewards.max()),
        reward_mean=float(rewards.mean()),
    )
    return batch, receipt


def encode_trajectory_rows(
    quads: list[Quadruple],
    tokenizer: ByteTokenizer,
    *,
    field: str,
    max_tokens: int,
) -> jnp.ndarray:
    """Encode ``x`` plus one replay field for the local CTS trajectory path."""

    if field not in {"y0", "y1"}:
        raise ValueError("field must be 'y0' or 'y1'")
    rows = [
        [BOS, *tokenizer.encode(q.x), SEP, *tokenizer.encode(getattr(q, field)), EOS] for q in quads
    ]
    width = min(max(len(row) for row in rows), max_tokens)
    return _pad_rows(rows, width=width)


def encode_critiques(
    quads: list[Quadruple], tokenizer: ByteTokenizer, *, max_tokens: int
) -> jnp.ndarray:
    rows = [[*tokenizer.encode(q.f), EOS] for q in quads]
    width = min(max(len(row) for row in rows), max_tokens)
    return _pad_rows(rows, width=width)
