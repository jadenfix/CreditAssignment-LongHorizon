import jax
import jax.numpy as jnp
import pytest

from cts.losses.alignment import AlignmentCfg, alignment_loss
from cts.losses.decode_anchor import decode_anchor_loss
from cts.losses.energy_loss import energy_contrastive_loss
from cts.losses.sinkhorn import sinkhorn_divergence_loss
from cts.losses.soft_dtw import soft_dtw_loss


def test_decode_anchor_shapes_and_grad():
    rng = jax.random.PRNGKey(0)
    logits = jax.random.normal(rng, (2, 5, 7))
    tgt = jnp.zeros((2, 5), dtype=jnp.int32)
    mask = jnp.ones((2, 5))
    loss = decode_anchor_loss(logits, tgt, mask)
    assert loss.shape == ()
    g = jax.grad(lambda L: decode_anchor_loss(L, tgt, mask))(logits)
    assert g.shape == logits.shape


def test_sinkhorn_fallback_symmetry_nonneg():
    rng = jax.random.PRNGKey(1)
    a = jax.random.normal(rng, (2, 6, 4))
    b = jax.random.normal(rng, (2, 7, 4))
    v_ab = sinkhorn_divergence_loss(a, b, epsilon=0.5, num_iters=200, use_ott=False)
    v_ba = sinkhorn_divergence_loss(b, a, epsilon=0.5, num_iters=200, use_ott=False)
    v_aa = sinkhorn_divergence_loss(a, a, epsilon=0.5, num_iters=200, use_ott=False)
    assert jnp.abs(v_ab - v_ba) < 1e-2
    assert v_ab >= -1e-3
    assert jnp.abs(v_aa) < 1e-2


def test_soft_dtw_monotone_in_gamma():
    rng = jax.random.PRNGKey(2)
    x = jax.random.normal(rng, (1, 4, 3))
    y = jax.random.normal(rng, (1, 5, 3))
    v_small = soft_dtw_loss(x, y, gamma=0.1)
    v_large = soft_dtw_loss(x, y, gamma=5.0)
    # soft-DTW is non-increasing as gamma grows (softer min -> smaller value), within tol
    assert v_large <= v_small + 1e-4


def test_alignment_dispatcher_routes():
    rng = jax.random.PRNGKey(3)
    a = jax.random.normal(rng, (2, 5, 4))
    b = jax.random.normal(rng, (2, 5, 4))
    l2 = alignment_loss(a, b, AlignmentCfg(kind="l2", weight=1.0))
    dtw = alignment_loss(a, b, AlignmentCfg(kind="soft_dtw", weight=1.0, gamma=1.0))
    sk = alignment_loss(
        a, b, AlignmentCfg(kind="sinkhorn", weight=1.0, num_iters=20, use_ott=False)
    )
    zero = alignment_loss(a, b, AlignmentCfg(kind="sinkhorn", weight=0.0))
    for v in (l2, dtw, sk):
        assert jnp.isfinite(v)
    assert jnp.asarray(zero) == 0.0


def test_alignment_unknown_kind_raises():
    with pytest.raises(ValueError):
        alignment_loss(jnp.zeros((1, 2, 3)), jnp.zeros((1, 2, 3)), AlignmentCfg(kind="nope"))  # type: ignore[arg-type]


def test_sinkhorn_fallback_small_epsilon_stable():
    # Pre-fix: K = exp(-C/eps) underflows to 0 here and the result is NaN/0.
    rng = jax.random.PRNGKey(7)
    a = jax.random.normal(rng, (1, 5, 3))
    b = jax.random.normal(rng, (1, 6, 3))
    v = sinkhorn_divergence_loss(a, b, epsilon=1e-3, num_iters=50, use_ott=False)
    assert jnp.isfinite(v)
    assert v >= -1e-3


def test_energy_contrastive_has_grad():
    rng = jax.random.PRNGKey(4)
    eb = jax.random.normal(rng, (4,))
    eg = jax.random.normal(rng, (4,))
    loss = energy_contrastive_loss(eb, eg, margin=1.0)
    assert jnp.isfinite(loss)
    g = jax.grad(lambda x: energy_contrastive_loss(x, eg))(eb)
    assert g.shape == eb.shape
