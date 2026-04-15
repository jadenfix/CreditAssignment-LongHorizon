"""Minimal train loop for the local (``LocalModelAPI``) path.

Calls ``method.step(model, batch, cfg_obj)`` and applies Optax updates. Real
experiments route through ``cts.backends.tunix_adapter.TunixAdapter.step``
instead; this loop exists so every method can be smoke-tested on nano-LM.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import optax
from flax import nnx

from ..utils.logging import JSONLLogger


@dataclass
class LoopCfg:
    lr: float = 3e-4
    max_steps: int = 100
    grad_clip: float = 1.0
    log_every: int = 10
    ckpt_every: int = 100
    warmup_steps: int = 10


def _make_optimizer(cfg: LoopCfg) -> optax.GradientTransformation:
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.lr,
        warmup_steps=cfg.warmup_steps,
        decay_steps=max(cfg.max_steps, 1),
        end_value=cfg.lr * 0.1,
    )
    return optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adamw(schedule))


def train_local(
    model: nnx.Module | tuple[nnx.Module, ...],
    step_fn: Callable[..., tuple[jax.Array, dict]],
    cfg: LoopCfg,
    *,
    log_dir: str | Path,
    freeze_mask: Any = None,
) -> None:
    """Local Optax/NNX training loop.

    ``model`` may be a single ``nnx.Module`` or a tuple of modules — all of
    their ``nnx.Param`` subtrees are collected into one combined state and
    optimized jointly. This is what lets CTS train its backbone together with
    the projection, critique embedding, and energy critic in one step (without
    it, the CTS-specific modules silently stay at init).

    ``step_fn`` is called as ``step_fn(model, step)`` when ``model`` is a single
    module, or ``step_fn(*model, step)`` when it's a tuple.

    ``freeze_mask`` (optional): a pytree with the same structure as the
    combined param state whose leaves are booleans — ``True`` means freeze
    (zero gradient update) that leaf. Used e.g. to implement A5 at the
    optimizer level as an alternative to the in-step stop_gradient.
    """
    modules = model if isinstance(model, tuple) else (model,)
    multi = isinstance(model, tuple)

    # Modern nnx grad pattern: split once to get an immutable graphdef per
    # module and mutate only the `params` pytree inside the traced function.
    # Calling nnx.update inside jax.grad trips Flax's trace-context guard.
    graphdefs_and_rest = [nnx.split(m, nnx.Param, ...) for m in modules]
    graphdefs = tuple(x[0] for x in graphdefs_and_rest)
    rests = tuple(x[2] for x in graphdefs_and_rest)

    def _merge(params):
        return tuple(nnx.merge(g, p, r) for g, p, r in zip(graphdefs, params, rests, strict=False))

    base_opt = _make_optimizer(cfg)
    if freeze_mask is not None:
        opt = optax.chain(
            optax.masked(optax.set_to_zero(), freeze_mask),
            base_opt,
        )
    else:
        opt = base_opt

    params = tuple(x[1] for x in graphdefs_and_rest)
    opt_state = opt.init(params)
    logger = JSONLLogger(Path(log_dir) / "train.jsonl")

    for step in range(cfg.max_steps):

        def loss_and_metrics(params, step=step):
            merged = _merge(params)
            if multi:
                return step_fn(*merged, step)
            return step_fn(merged[0], step)

        (loss, metrics), grads = jax.value_and_grad(loss_and_metrics, has_aux=True)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if step % cfg.log_every == 0:
            logger.log(step, {"loss": float(loss), **{k: float(v) for k, v in metrics.items()}})

    # Write final params back to the user-visible module objects (outside trace).
    for m, p in zip(modules, params, strict=False):
        nnx.update(m, p)

    logger.close()
