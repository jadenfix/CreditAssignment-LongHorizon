"""Hydra/OmegaConf entry point.

Example::

    python -m cts.train.main method=cts task=fixtures_math backend=local_nano \
        trainer.max_steps=20
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from ..backends.nano_lm import NanoLM, NanoLMConfig
from ..data.replay import ReplayShard
from ..methods import get_method
from ..methods._batch import encode_batch, prompt_template_hash
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

    # Backend
    if cfg.backend.kind == "tunix":
        from ..backends.tunix_adapter import TunixAdapter

        adapter = TunixAdapter(cfg)
        print(f"[cts] tunix trainer ready: {type(adapter.trainer_handle()).__name__}")
        # Real training delegated to the Tunix trainer; one step() so smoke
        # surfaces wiring problems immediately.
        # (The replay shard load + batch encode below is shared with local.)
    elif cfg.backend.kind != "local_nano":
        raise NotImplementedError(
            f"Backend {cfg.backend.kind!r} not wired yet; see cts.backends.{cfg.backend.kind}_adapter."
        )
    nano_cfg = NanoLMConfig(
        vocab_size=cfg.backend.nano.vocab_size,
        hidden_size=cfg.backend.nano.hidden_size,
        num_layers=cfg.backend.nano.num_layers,
        num_heads=cfg.backend.nano.num_heads,
        max_seq_len=cfg.backend.nano.max_seq_len,
    )
    model = NanoLM(nano_cfg, seed=cfg.seed)
    tokenizer = ByteTokenizer(vocab_size=nano_cfg.vocab_size)

    # Data
    quads = ReplayShard(cfg.task.replay_path).load_split(cfg.task.split_train)
    if not quads:
        raise RuntimeError(f"No quadruples at {cfg.task.replay_path} split={cfg.task.split_train}")

    # Method
    method = get_method(cfg.method.name)
    batch = encode_batch(quads[: cfg.trainer.batch_size], tokenizer)

    # Smoke: one forward call per method just to prove wiring. Full training
    # loops land in M8 once Tunix is wired; this keeps the entry point
    # importable and exercises the method-dispatch path for every config.
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

    if args.gcs_bucket:
        # Defer import so non-GCP runs don't pay for the dep.
        import sys as _sys

        _deploy = Path(__file__).resolve().parents[3] / "deploy"
        _sys.path.insert(0, str(_deploy))
        from gcs_sync import upload  # type: ignore[import-not-found]

        upload("artifacts/", args.gcs_bucket.rstrip("/") + f"/{cfg.out_dir}")


if __name__ == "__main__":
    main()
