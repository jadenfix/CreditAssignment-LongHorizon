"""Hydra/OmegaConf entry point.

Example::

    python -m cts.train.main method=cts task=fixtures_math backend=local_nano \
        trainer.max_steps=20
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
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
    cfg.fairness.total_opt_steps = cfg.trainer.max_steps
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
    if cfg.backend.kind != "local_nano":
        raise NotImplementedError(
            f"CLI training is implemented only for local_nano, not {cfg.backend.kind!r}. "
            "Use a configured scale adapter explicitly; no local run is a scale experiment."
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

    import jax
    import jax.numpy as jnp
    from flax import nnx

    from ..backends.local_api import DecodeCfg
    from .loop import LoopCfg, train_local

    if cfg.method.name == "sft_revision":
        loss_batch = batch
    elif cfg.method.name == "dpo":
        loss_batch = method.prepare_batch(quads[: cfg.trainer.batch_size], tokenizer)
    else:
        raise NotImplementedError(
            f"CLI optimizer training for {cfg.method.name!r} needs a task-specific rollout/reward "
            "builder. Run the replay/loss integration tests; a dispatch stub is not training."
        )
    loop_cfg = LoopCfg(**{key: cfg.trainer[key] for key in LoopCfg.__dataclass_fields__})
    if loop_cfg.max_steps < 1:
        raise ValueError("trainer.max_steps must be positive")
    before = jax.tree.map(lambda value: value.copy(), nnx.state(model, nnx.Param))
    train_local(
        model, lambda model_, step: method.step(model_, loss_batch), loop_cfg, log_dir=cfg.out_dir
    )
    after = nnx.state(model, nnx.Param)
    delta = sum(
        float(jnp.sum(jnp.abs(a - b)))
        for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True)
    )
    loss, metrics = method.step(model, loss_batch)
    if not bool(jnp.isfinite(loss)) or not bool(jnp.isfinite(delta)):
        raise RuntimeError("Local training produced non-finite loss or parameter change")
    generated = model.generate(batch.input_ids[:, :8], DecodeCfg(max_new_tokens=2, greedy=True))
    receipt = {
        "method": cfg.method.name,
        "backend": cfg.backend.kind,
        "steps": loop_cfg.max_steps,
        "parameter_delta_l1": delta,
        "loss": float(loss),
        "fairness_fingerprint": asdict(fp),
        "replay_path": str(cfg.task.replay_path),
        "generated_token_shape": list(generated.tokens.shape),
        "scope": "tiny local randomly initialized model; no policy-quality claim",
    }
    output = Path(cfg.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "training-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"[cts] trained steps={loop_cfg.max_steps} loss={float(loss):.4f} delta={delta:.6g}")
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
