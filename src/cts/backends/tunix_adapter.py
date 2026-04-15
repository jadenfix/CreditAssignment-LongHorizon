"""Tunix :class:`ScaleBackendAdapter` — stub.

Kept importable without the ``tunix`` package installed; raises at construction
time if the dep is missing. Real wiring (SFT / DPO / PPO / GRPO trainers) lands
in milestone M8.
"""

from __future__ import annotations

from typing import Any

import jax

from .scale_adapter import RolloutOut


class TunixAdapter:
    name = "tunix"

    def __init__(self, cfg: dict[str, Any]):
        try:
            import tunix  # noqa: F401
        except Exception as e:  # pragma: no cover - depends on env
            raise ImportError(
                "TunixAdapter requires the 'tunix' package. Install via `pip install cts[scale]`."
            ) from e
        self.cfg = cfg
        self._trainer: Any = None  # filled in M8

    def trainer_handle(self) -> Any:  # pragma: no cover - M8
        raise NotImplementedError("Tunix trainer wiring lands in M8.")

    def rollout(self, prompts: jax.Array, cfg: dict[str, Any]) -> RolloutOut:  # pragma: no cover
        raise NotImplementedError("Tunix rollout lands in M8.")

    def extract_hidden(
        self, layer_ids: list[int], token_slice: slice | None = None
    ) -> dict[int, jax.Array]:  # pragma: no cover
        raise NotImplementedError

    def checkpoint(self, path: str, step: int) -> None:  # pragma: no cover
        raise NotImplementedError

    def restore(self, path: str) -> int:  # pragma: no cover
        raise NotImplementedError

    def step(self, batch: Any) -> dict[str, jax.Array]:  # pragma: no cover
        raise NotImplementedError
