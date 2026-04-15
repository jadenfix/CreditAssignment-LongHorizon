"""B3 — GRPO + verifier / process rewards.

Identical surrogate to B2; the difference lives in how ``batch.rewards`` are
constructed upstream (per-step verifier / PRM score rolled up into a scalar).
Keeping the step function a thin shim makes the fairness guarantee obvious.
"""

from __future__ import annotations

from .b2_grpo_outcome import GRPOBatch, step  # noqa: F401

NAME = "grpo_verifier"
