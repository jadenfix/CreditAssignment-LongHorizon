"""Method registry. Each method module exposes ``step(state, batch) -> (loss, metrics)``.

Ablations are config toggles on top of ``b5_cts``; see ``configs/method/cts.yaml``.
"""

from . import b0_sft_revision, b1_dpo, b2_grpo_outcome, b3_grpo_verifier, b4_grpo_critique, b5_cts

REGISTRY = {
    "sft_revision": b0_sft_revision,
    "dpo": b1_dpo,
    "grpo_outcome": b2_grpo_outcome,
    "grpo_verifier": b3_grpo_verifier,
    "grpo_critique": b4_grpo_critique,
    "cts": b5_cts,
}


def get_method(name: str):
    if name not in REGISTRY:
        raise KeyError(f"Unknown method {name!r}. Available: {list(REGISTRY)}")
    return REGISTRY[name]
