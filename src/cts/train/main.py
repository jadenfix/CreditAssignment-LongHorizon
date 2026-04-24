"""Hydra/OmegaConf entry point.

Example::

    python -m cts.train.main method=cts task=fixtures_math backend=local_nano \
        trainer.max_steps=20
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from ..backends.nano_lm import NanoLM, NanoLMConfig
from ..data.replay import ReplayDir, ReplayShard, sample_minibatches
from ..methods import get_method
from ..methods._batch import encode_batch, prompt_template_hash
from ..utils.logging import JSONLLogger
from ..utils.tokenizer import ByteTokenizer
from .fairness import fingerprint_from_cfg

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def _load_cfg(overrides: list[str]):
    base = OmegaConf.load(_CONFIG_DIR / "base.yaml")
    # split overrides into group-selects (e.g. `method=cts`) and leaf sets
    group_overrides = []
    leaf_overrides = []
    for ov in overrides:
        if "=" not in ov:
            continue
        k, v = ov.split("=", 1)
        if k in {"method", "task", "backend"} and "/" not in v and "." not in k:
            group_overrides.append((k, v))
        else:
            leaf_overrides.append(ov)
    cfg = OmegaConf.merge(base)
    for group, choice in group_overrides:
        group_cfg = OmegaConf.load(_CONFIG_DIR / group / f"{choice}.yaml")
        cfg = OmegaConf.merge(cfg, group_cfg)
    if leaf_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(leaf_overrides))
    # Stamp the fairness prompt-template hash so sweeps can check consistency.
    cfg.fairness.prompt_template_hash = prompt_template_hash()
    return cfg


def _load_quads(cfg, split: str):
    rpath = Path(cfg.task.replay_path)
    if rpath.is_dir():
        quads = ReplayDir(rpath).load_split(split)
    else:
        quads = ReplayShard(rpath).load_split(split)
    if not quads:
        raise RuntimeError(f"No quadruples at {rpath} split={split}")
    return quads


def _scalar_metrics(metrics: Any) -> dict[str, float]:
    """Coerce whatever the trainer returned into flat float scalars for JSONL."""
    out: dict[str, float] = {}
    if metrics is None:
        return out
    items = metrics.items() if hasattr(metrics, "items") else []
    for k, v in items:
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _tunix_train(cfg) -> None:
    """Multi-step Tunix training with orbax checkpointing.

    Contract assumptions (asserted in ``TunixAdapter.__init__``):
      - ``adapter.step(batch)`` returns a dict of scalar metrics.
      - Tunix's trainer owns tokenization; we pass raw Quadruple dicts so the
        adapter can route them through the trainer's own input pipeline.
    """
    from ..backends.tunix_adapter import TunixAdapter

    adapter = TunixAdapter(cfg)
    print(f"[cts] tunix trainer ready: {type(adapter.trainer_handle()).__name__}")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "ckpt"

    start_step = 0
    if ckpt_dir.exists():
        try:
            start_step = int(adapter.restore(str(ckpt_dir)))
            if start_step:
                print(f"[cts] resumed from step {start_step}")
        except Exception as e:  # noqa: BLE001  — restore failure is informative, not fatal
            print(f"[cts] restore failed ({e}); starting from step 0")
            start_step = 0

    quads = _load_quads(cfg, cfg.task.split_train)
    print(f"[cts] loaded {len(quads)} quadruples from {cfg.task.replay_path}")

    log = JSONLLogger(out_dir / "train.jsonl")

    # Seeded, deterministic minibatches; restart with a new seed each pass so
    # long runs cover fresh orderings without reshuffling mid-epoch.
    def _epoch_iter(epoch: int):
        return sample_minibatches(
            quads,
            cfg.trainer.batch_size,
            seed=int(cfg.seed) + epoch,
            shuffle=True,
            drop_last=True,
        )

    epoch = 0
    batch_iter = _epoch_iter(epoch)

    step = start_step
    max_steps = int(cfg.trainer.max_steps)
    log_every = int(cfg.trainer.log_every)
    ckpt_every = int(cfg.trainer.ckpt_every)

    while step < max_steps:
        try:
            quad_batch = next(batch_iter)
        except StopIteration:
            epoch += 1
            batch_iter = _epoch_iter(epoch)
            quad_batch = next(batch_iter)

        # Raw-quad payload: Tunix's trainer handles tokenization/rollout
        # internally, so we don't pre-encode. The adapter's method-specific
        # translation layer is responsible for any shape coercion.
        batch = [q.model_dump() for q in quad_batch]
        metrics = adapter.step(batch)

        if step % log_every == 0:
            log.log(step, _scalar_metrics(metrics))
        if step > 0 and step % ckpt_every == 0:
            adapter.checkpoint(str(ckpt_dir), step)
        step += 1

    adapter.checkpoint(str(ckpt_dir), step)
    log.close()
    print(f"[cts] done: {step} steps; logs={out_dir / 'train.jsonl'} ckpt={ckpt_dir}")


def _local_nano_smoke(cfg) -> None:
    """Single-forward smoke on the local NanoLM — preserves the pre-existing
    behavior for unit tests and fixture-scale runs. Full local training lives
    in ``cts.train.loop.train_local`` which the unit tests call directly."""
    nano_cfg = NanoLMConfig(
        vocab_size=cfg.backend.nano.vocab_size,
        hidden_size=cfg.backend.nano.hidden_size,
        num_layers=cfg.backend.nano.num_layers,
        num_heads=cfg.backend.nano.num_heads,
        max_seq_len=cfg.backend.nano.max_seq_len,
    )
    model = NanoLM(nano_cfg, seed=cfg.seed)
    tokenizer = ByteTokenizer(vocab_size=nano_cfg.vocab_size)

    quads = _load_quads(cfg, cfg.task.split_train)

    method = get_method(cfg.method.name)
    batch = encode_batch(quads[: cfg.trainer.batch_size], tokenizer)

    if cfg.method.name in {"sft_revision"}:
        loss, metrics = method.step(model, batch)
    elif cfg.method.name == "dpo":
        pref_batch = method.prepare_batch(quads[: cfg.trainer.batch_size], tokenizer)
        loss, metrics = method.step(model, pref_batch)
    else:
        print(
            f"[cts] Method {cfg.method.name!r} requires RL rollouts (see "
            "cts.rl.rollout / TunixAdapter). Entry point stub — run its unit test instead."
        )
        return
    print(f"[cts] step0 loss={float(loss):.4f}")
    print(f"[cts] metrics={ {k: float(v) for k, v in metrics.items()} }")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("overrides", nargs="*")
    ap.add_argument(
        "--gcs-bucket",
        default=None,
        help="If set (gs://…), rsync artifacts/ to this prefix at end of run.",
    )
    args = ap.parse_args()

    cfg = _load_cfg(args.overrides)
    fp = fingerprint_from_cfg(cfg)
    print(f"[cts] method={cfg.method.name} backend={cfg.backend.kind} task={cfg.task.name}")
    print(f"[cts] fairness fingerprint: {fp}")

    if cfg.backend.kind == "tunix":
        _tunix_train(cfg)
    elif cfg.backend.kind == "local_nano":
        _local_nano_smoke(cfg)
    else:
        raise NotImplementedError(
            f"Backend {cfg.backend.kind!r} not wired yet; see cts.backends.{cfg.backend.kind}_adapter."
        )

    if args.gcs_bucket:
        import sys as _sys

        _deploy = Path(__file__).resolve().parents[3] / "deploy"
        _sys.path.insert(0, str(_deploy))
        from gcs_sync import upload  # type: ignore[import-not-found]

        upload("artifacts/", args.gcs_bucket.rstrip("/") + f"/{cfg.out_dir}")


if __name__ == "__main__":
    main()
