"""Seed helpers. Keep RNG sources explicit; avoid hidden global state.

Note: any ``jax.config.update(...)`` flags (notably ``jax_enable_x64``) must be
set before the first JAX array is materialised — call them at program start,
before invoking :func:`seed_everything`.
"""

from __future__ import annotations

import random

import jax
import numpy as np


def seed_everything(seed: int) -> jax.Array:
    random.seed(seed)
    np.random.seed(seed)
    return jax.random.key(seed)
