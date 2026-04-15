"""``ScaleBackendAdapter`` — the sharding-aware boundary for Tunix / MaxText runs.

CTS-specific code (projections, energy critic, alignment loss) lives *above*
this adapter and consumes whatever hidden tensors the adapter exposes. Intentionally
separate from :class:`LocalModelAPI` — do not merge them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import jax


@dataclass
class RolloutOut:
    tokens: jax.Array  # sharded [B, T]
    hidden_proxy: dict[int, jax.Array]  # layer_id -> [B, T_slice, D] from extract_hidden
    rewards: jax.Array  # [B] outcome-reward (verifier)
    extra: dict[str, Any]


@runtime_checkable
class ScaleBackendAdapter(Protocol):
    """Thin adapter around a large distributed trainer (Tunix / MaxText)."""

    name: str  # 'tunix' | 'maxtext'

    def trainer_handle(self) -> Any:
        """Return the underlying trainer object (Tunix GRPOTrainer, MaxText train state, ...)."""
        ...

    def rollout(self, prompts: jax.Array, cfg: dict[str, Any]) -> RolloutOut: ...

    def extract_hidden(
        self, layer_ids: list[int], token_slice: slice | None = None
    ) -> dict[int, jax.Array]:
        """Pull selected hidden states out of the last forward/rollout."""
        ...

    def checkpoint(self, path: str, step: int) -> None: ...

    def restore(self, path: str) -> int: ...

    def step(self, batch: Any) -> dict[str, jax.Array]: ...
