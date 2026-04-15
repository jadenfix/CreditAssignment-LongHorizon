"""Tunix :class:`ScaleBackendAdapter` — M8 wiring.

Construction-time guard: without the ``tunix`` package installed this raises
``ImportError`` so it never silently degrades. With Tunix installed, the
adapter dispatches to the trainer matching ``cfg.method.name`` (sft / dpo /
ppo / grpo) and delegates checkpointing to ``orbax-checkpoint`` (which we
already depend on).

Trainer-symbol resolution is table-driven (``_TRAINER_TABLE``) so that bumping
Tunix only requires editing one mapping rather than hunting through the
adapter — Tunix's internal module layout has shifted across releases.

Hidden-state extraction and rollout call into the trainer's underlying model
when present; if the Tunix trainer doesn't expose a hidden-state hook the
extractor raises ``NotImplementedError`` rather than silently returning the
wrong shape (B5/CTS depends on the per-layer cache).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax

from .scale_adapter import RolloutOut

# Edit this when bumping Tunix versions. Keys are method names from
# `cts.methods.*.name` (configs/method/*.yaml).
_TRAINER_TABLE: dict[str, tuple[str, str]] = {
    "sft": ("tunix.sft", "SftTrainer"),
    "sft_revision": ("tunix.sft", "SftTrainer"),
    "dpo": ("tunix.dpo", "DpoTrainer"),
    "grpo_outcome": ("tunix.rl.grpo", "GrpoTrainer"),
    "grpo_verifier": ("tunix.rl.grpo", "GrpoTrainer"),
    "grpo_critique": ("tunix.rl.grpo", "GrpoTrainer"),
    "cts": ("tunix.rl.grpo", "GrpoTrainer"),  # CTS rides on GRPO trainer + aux losses
}


def _resolve_trainer(method: str) -> Any:
    if method not in _TRAINER_TABLE:
        raise ValueError(f"No Tunix trainer mapping for method {method!r}")
    module_path, cls_name = _TRAINER_TABLE[method]
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


class TunixAdapter:
    name = "tunix"

    def __init__(self, cfg: dict[str, Any]):
        try:
            import tunix  # noqa: F401
        except Exception as e:
            raise ImportError(
                "TunixAdapter requires the 'tunix' package. "
                "Install via `uv pip install -e '.[scale]'`."
            ) from e
        self.cfg = cfg
        method_name = cfg["method"]["name"] if isinstance(cfg, dict) else cfg.method.name
        TrainerCls = _resolve_trainer(method_name)
        # Tunix trainer constructors take (model, optimizer, **method_kwargs).
        # We build the model via Tunix's model registry; the user pins the model
        # name in configs/backend/tunix.yaml.
        self._trainer = TrainerCls(**self._trainer_kwargs(cfg))

    @staticmethod
    def _trainer_kwargs(cfg: Any) -> dict[str, Any]:
        # Translate the OmegaConf cfg into Tunix's expected kwargs. Keep this
        # narrow: Tunix expects model name, mesh, optimizer, max_steps, etc.
        b = cfg["backend"]["tunix"] if isinstance(cfg, dict) else cfg.backend.tunix
        t = cfg["trainer"] if isinstance(cfg, dict) else cfg.trainer
        return {
            "model_name": b["model_name"] if isinstance(b, dict) else b.model_name,
            "mesh_shape": tuple(b["mesh_shape"] if isinstance(b, dict) else b.mesh_shape),
            "lr": float(t["lr"] if isinstance(t, dict) else t.lr),
            "max_steps": int(t["max_steps"] if isinstance(t, dict) else t.max_steps),
        }

    def trainer_handle(self) -> Any:
        return self._trainer

    def rollout(self, prompts: jax.Array, cfg: dict[str, Any]) -> RolloutOut:
        out = self._trainer.rollout(prompts, **cfg)
        return RolloutOut(
            tokens=out.tokens,
            hidden_proxy=getattr(out, "hidden_proxy", {}),
            rewards=out.rewards,
            extra=getattr(out, "extra", {}),
        )

    def extract_hidden(
        self, layer_ids: list[int], token_slice: slice | None = None
    ) -> dict[int, jax.Array]:
        hook = getattr(self._trainer, "extract_hidden", None)
        if hook is None:
            raise NotImplementedError(
                "The installed Tunix trainer does not expose extract_hidden(); "
                "B5 (CTS) needs a per-layer hidden cache. Patch tunix or pin "
                "an older revision (see deploy/Dockerfile.tpu)."
            )
        return hook(layer_ids, token_slice)

    def checkpoint(self, path: str, step: int) -> None:
        # Orbax is already a top-level dep; use it directly so checkpointing
        # works even before Tunix exposes its own checkpoint surface.
        import orbax.checkpoint as ocp

        ckptr = ocp.PyTreeCheckpointer()
        Path(path).mkdir(parents=True, exist_ok=True)
        ckptr.save(Path(path) / f"step_{step}", self._trainer.state)

    def restore(self, path: str) -> int:
        import orbax.checkpoint as ocp

        ckptr = ocp.PyTreeCheckpointer()
        steps = sorted(int(p.name.split("_")[1]) for p in Path(path).glob("step_*"))
        if not steps:
            return 0
        last = steps[-1]
        self._trainer.state = ckptr.restore(Path(path) / f"step_{last}")
        return last

    def step(self, batch: Any) -> dict[str, jax.Array]:
        # Tunix trainers expose .step(batch) returning a dict-like metrics object.
        out = self._trainer.step(batch)
        return dict(out) if not isinstance(out, dict) else out
