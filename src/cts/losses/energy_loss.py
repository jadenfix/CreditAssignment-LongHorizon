"""Contrastive training loss for the energy critic ``E_psi``.

We want ``E_psi`` to be higher on failed trajectory / critique pairs and lower
on corrected / critique pairs. A simple margin / BCE loss on matched and
mismatched pairs is sufficient for this first scaffold; replace with InfoNCE
later if it becomes a bottleneck.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def energy_contrastive_loss(
    e_bad: jax.Array,  # [B] scalar energy assigned to (R_bad, e_f)
    e_good: jax.Array,  # [B] scalar energy assigned to (R_good, e_f)
    *,
    margin: float = 1.0,
) -> jax.Array:
    """Margin-ranking loss: ``relu(margin + e_good - e_bad)`` averaged."""
    return jnp.mean(jax.nn.relu(margin + e_good - e_bad))
