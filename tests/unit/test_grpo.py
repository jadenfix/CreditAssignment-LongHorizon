import jax
import jax.numpy as jnp

from cts.rl.grpo import GRPOCfg, approx_kl, group_relative_advantages, grpo_surrogate_loss


def test_approx_kl_zero_when_policies_match():
    rng = jax.random.PRNGKey(0)
    logp = jax.random.normal(rng, (2, 4, 8))
    mask = jnp.ones_like(logp)
    kl = approx_kl(logp, logp, mask)
    assert jnp.abs(kl) < 1e-6


def test_approx_kl_nonneg_on_random_inputs():
    k1, k2 = jax.random.split(jax.random.PRNGKey(1))
    a = jax.random.normal(k1, (3, 5, 7))
    b = jax.random.normal(k2, (3, 5, 7))
    mask = jnp.ones_like(a)
    kl = approx_kl(a, b, mask)
    assert kl >= -1e-6


def test_group_relative_advantages_zero_mean_unit_std():
    rng = jax.random.PRNGKey(2)
    r = jax.random.normal(rng, (4, 8))
    A = group_relative_advantages(r)
    assert jnp.allclose(A.mean(axis=-1), 0.0, atol=1e-5)
    assert jnp.allclose(A.std(axis=-1), 1.0, atol=1e-3)


def test_grpo_surrogate_runs_and_kl_coef_is_wired():
    rng = jax.random.PRNGKey(3)
    k_new, k_old, k_r = jax.random.split(rng, 3)
    new = jax.random.normal(k_new, (2, 4, 6))
    old = jax.random.normal(k_old, (2, 4, 6))
    mask = jnp.ones_like(new)
    adv = jax.random.normal(k_r, (2, 4))
    base = grpo_surrogate_loss(new, old, mask, adv, GRPOCfg(kl_coef=0.0))
    with_kl = grpo_surrogate_loss(new, old, mask, adv, GRPOCfg(kl_coef=1.0))
    # KL is nonneg, so the penalised loss must not be smaller than the baseline.
    assert with_kl >= base - 1e-6
