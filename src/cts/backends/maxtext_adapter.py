"""MaxText :class:`ScaleBackendAdapter` — stub.

Importable without the ``maxtext`` package; concrete wiring lands in M9. The
documented scale path (see README) uses this adapter.
"""

from __future__ import annotations

from typing import Any

import jax

from .scale_adapter import RolloutOut


class MaxTextAdapter:
    name = "maxtext"

    def __init__(self, cfg: dict[str, Any]):
        try:
            import MaxText  # noqa: F401  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "MaxTextAdapter requires the 'maxtext' package on PYTHONPATH. "
                "See README for the scale-reproduction setup."
            ) from e
        self.cfg = cfg

    def trainer_handle(self) -> Any:  # pragma: no cover - M9
        raise NotImplementedError("MaxText train-state wiring lands in M9.")

    def rollout(self, prompts: jax.Array, cfg: dict[str, Any]) -> RolloutOut:  # pragma: no cover
        raise NotImplementedError

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
