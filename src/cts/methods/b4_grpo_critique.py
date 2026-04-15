"""B4 — Critique-GRPO: GRPO + natural-language critique as an extra reward channel.

Upstream constructs ``batch.rewards`` via :func:`cts.rl.critique_reward.critique_reward`;
the step function is again a shim over B2.
"""

from __future__ import annotations

from .b2_grpo_outcome import GRPOBatch, step  # noqa: F401

NAME = "grpo_critique"
