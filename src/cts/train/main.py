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
from typing import Any, cast

from omegaconf import OmegaConf

from ..backends.nano_lm import NanoLM, NanoLMConfig
from ..data.replay import ReplayShard
from ..data.schema import dataset_hash
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
    from ..losses.alignment import AlignmentCfg
    from ..methods.b5_cts import CTSBatch, CTSCfg, CTSModules
    from ..models.energy import EnergyCfg
    from ..models.projection import ProjectionCfg
    from ..rl.grpo import GRPOCfg
    from .fixture_rl import (
        build_fixture_math_grpo_batch,
        encode_critiques,
        encode_trajectory_rows,
    )
    from .loop import LoopCfg, train_local

    rollout_receipt = None
    if cfg.method.name == "sft_revision":
        loss_batch = batch
        train_target = model

        def supervised_step(model_, step):
            return method.step(model_, loss_batch)

        step_fn = supervised_step
    elif cfg.method.name == "dpo":
        loss_batch = method.prepare_batch(quads[: cfg.trainer.batch_size], tokenizer)
        train_target = model

        def preference_step(model_, step):
            return method.step(model_, loss_batch)

        step_fn = preference_step
    elif cfg.method.name in {"grpo_outcome", "grpo_verifier", "grpo_critique", "cts"}:
        selected = quads[: cfg.trainer.batch_size]
        reward_kind = (
            "verifier" if cfg.method.name == "cts" else cfg.method_cfg.get("reward", "verifier")
        )
        grpo_batch, rollout_receipt = build_fixture_math_grpo_batch(
            model,
            selected,
            tokenizer,
            group_size=int(cfg.method_cfg.get("group_size", 4)),
            max_new_tokens=int(cfg.method_cfg.get("max_new_tokens", 24)),
            max_prompt_tokens=nano_cfg.max_seq_len - int(cfg.method_cfg.get("max_new_tokens", 24)),
            temperature=float(cfg.fairness.decode_temperature),
            seed=int(cfg.seed),
            reward_kind=str(reward_kind),
            critique_lam=float(cfg.method_cfg.get("critique_lam", 0.3)),
        )
        grpo_cfg_data = cfg.method_cfg.get("grpo", cfg.method_cfg)
        grpo_cfg = GRPOCfg(
            kl_coef=float(grpo_cfg_data.get("kl_coef", 0.01)),
            clip=float(grpo_cfg_data.get("clip", 0.2)),
            eps=float(grpo_cfg_data.get("eps", 1e-6)),
        )
        if cfg.method.name == "cts":
            alignment_cfg = AlignmentCfg(
                **cast(dict[str, Any], OmegaConf.to_container(cfg.method_cfg.alignment))
            )
            projection_cfg = ProjectionCfg(
                **cast(dict[str, Any], OmegaConf.to_container(cfg.method_cfg.projection))
            )
            energy_cfg = EnergyCfg(
                **cast(dict[str, Any], OmegaConf.to_container(cfg.method_cfg.energy))
            )
            cts_cfg = CTSCfg(
                lam_align=float(cfg.method_cfg.lam_align),
                lam_decode=float(cfg.method_cfg.lam_decode),
                lam_energy=float(cfg.method_cfg.lam_energy),
                edit_eta=float(cfg.method_cfg.edit_eta),
                use_ref_trajectory=bool(cfg.method_cfg.use_ref_trajectory),
                online_critique=bool(cfg.method_cfg.online_critique),
                alignment=alignment_cfg,
                projection=projection_cfg,
                energy=energy_cfg,
                grpo=grpo_cfg,
            )
            modules = CTSModules(cts_cfg, rngs=nnx.Rngs(cfg.seed + 1))
            trajectory_len = min(int(cfg.method_cfg.max_len), nano_cfg.max_seq_len)
            loss_batch = CTSBatch(
                grpo=grpo_batch,
                input_ids=batch.input_ids,
                target_ids=batch.target_ids,
                y1_mask=batch.y1_mask,
                critique_ids=encode_critiques(
                    selected, tokenizer, max_tokens=min(64, trajectory_len)
                ),
                bad_input_ids=encode_trajectory_rows(
                    selected, tokenizer, field="y0", max_tokens=trajectory_len
                ),
                ref_input_ids=encode_trajectory_rows(
                    selected, tokenizer, field="y1", max_tokens=trajectory_len
                ),
            )
            train_target = (model, modules)

            def cts_step(model_, modules_, step):
                return method.step(model_, modules_, loss_batch, cts_cfg)

            step_fn = cts_step
        else:
            loss_batch = grpo_batch
            train_target = model

            def grpo_step(model_, step):
                return method.step(model_, loss_batch, grpo_cfg)

            step_fn = grpo_step
    else:
        raise NotImplementedError(f"CLI optimizer training is unavailable for {cfg.method.name!r}")
    loop_cfg = LoopCfg(**{key: cfg.trainer[key] for key in LoopCfg.__dataclass_fields__})
    if loop_cfg.max_steps < 1:
        raise ValueError("trainer.max_steps must be positive")
    before = jax.tree.map(lambda value: value.copy(), nnx.state(train_target, nnx.Param))
    train_local(train_target, step_fn, loop_cfg, log_dir=cfg.out_dir)
    after = nnx.state(train_target, nnx.Param)
    delta = sum(
        float(jnp.sum(jnp.abs(a - b)))
        for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True)
    )
    if cfg.method.name == "cts":
        loss, metrics = method.step(model, modules, loss_batch, cts_cfg)
    elif cfg.method.name in {"grpo_outcome", "grpo_verifier", "grpo_critique"}:
        loss, metrics = method.step(model, loss_batch, grpo_cfg)
    else:
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
        "replay_hash": dataset_hash(quads),
        "generated_token_shape": list(generated.tokens.shape),
        "scope": "tiny random fixture on CPU; no policy-quality or scale claim",
    }
    if rollout_receipt is not None:
        receipt["fixture_rollout"] = asdict(rollout_receipt)
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
