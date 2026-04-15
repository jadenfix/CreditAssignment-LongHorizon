"""The projection ``P_phi : h_t -> r_t`` onto the alignment manifold.

A small MLP that lifts per-token hidden states from the backbone into a
lower-dimensional "reasoning-trajectory" space where the alignment loss lives.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
from flax import nnx


@dataclass
class ProjectionCfg:
    hidden_size: int = 64
    out_dim: int = 32
    num_layers: int = 2


class Projection(nnx.Module):
    def __init__(self, cfg: ProjectionCfg, *, rngs: nnx.Rngs):
        self.cfg = cfg
        dims = [cfg.hidden_size] + [cfg.hidden_size] * (cfg.num_layers - 1) + [cfg.out_dim]
        self.layers = nnx.List(
            [nnx.Linear(dims[i], dims[i + 1], rngs=rngs) for i in range(len(dims) - 1)]
        )

    def __call__(self, h: jax.Array) -> jax.Array:
        x = h
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = jax.nn.gelu(x)
        return x
