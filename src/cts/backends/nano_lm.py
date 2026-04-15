"""A minimal Flax NNX transformer — tiny enough to run on CPU in unit tests.

Not a faithful GPT: it exists only so every loss and method can be exercised
end-to-end against :class:`LocalModelAPI`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from .local_api import DecodeCfg, ForwardOut, GenerateOut


@dataclass
class NanoLMConfig:
    vocab_size: int = 128
    hidden_size: int = 32
    num_layers: int = 2
    num_heads: int = 4
    max_seq_len: int = 384
    mlp_ratio: int = 2


class _MHA(nnx.Module):
    def __init__(self, cfg: NanoLMConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.qkv = nnx.Linear(cfg.hidden_size, 3 * cfg.hidden_size, rngs=rngs)
        self.out = nnx.Linear(cfg.hidden_size, cfg.hidden_size, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        B, T, D = x.shape
        h = self.cfg.num_heads
        dh = D // h
        qkv = self.qkv(x).reshape(B, T, 3, h, dh)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        attn = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(dh)
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))[None, None]
        attn = jnp.where(mask, attn, -1e9)
        w = jax.nn.softmax(attn, axis=-1)
        o = jnp.einsum("bhts,bshd->bthd", w, v).reshape(B, T, D)
        return self.out(o)


class _Block(nnx.Module):
    def __init__(self, cfg: NanoLMConfig, *, rngs: nnx.Rngs):
        self.ln1 = nnx.LayerNorm(cfg.hidden_size, rngs=rngs)
        self.attn = _MHA(cfg, rngs=rngs)
        self.ln2 = nnx.LayerNorm(cfg.hidden_size, rngs=rngs)
        self.fc1 = nnx.Linear(cfg.hidden_size, cfg.mlp_ratio * cfg.hidden_size, rngs=rngs)
        self.fc2 = nnx.Linear(cfg.mlp_ratio * cfg.hidden_size, cfg.hidden_size, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(jax.nn.gelu(self.fc1(self.ln2(x))))
        return x


class NanoLM(nnx.Module):
    """Tiny causal LM. Implements the :class:`LocalModelAPI` protocol."""

    def __init__(self, cfg: NanoLMConfig | None = None, *, seed: int = 0):
        cfg = cfg or NanoLMConfig()
        self.cfg = cfg
        rngs = nnx.Rngs(seed)
        self.tok_emb = nnx.Embed(cfg.vocab_size, cfg.hidden_size, rngs=rngs)
        self.pos_emb = nnx.Embed(cfg.max_seq_len, cfg.hidden_size, rngs=rngs)
        self.blocks = nnx.List([_Block(cfg, rngs=rngs) for _ in range(cfg.num_layers)])
        self.ln_f = nnx.LayerNorm(cfg.hidden_size, rngs=rngs)
        self.lm_head = nnx.Linear(cfg.hidden_size, cfg.vocab_size, rngs=rngs)

    @property
    def vocab_size(self) -> int:  # LocalModelAPI attr
        return self.cfg.vocab_size

    @property
    def hidden_size(self) -> int:  # LocalModelAPI attr
        return self.cfg.hidden_size

    def _embed(self, tokens: jax.Array) -> jax.Array:
        T = tokens.shape[-1]
        pos = jnp.arange(T)[None, :]
        return self.tok_emb(tokens) + self.pos_emb(pos)

    def forward(self, tokens: jax.Array) -> ForwardOut:
        h = self._embed(tokens)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        return ForwardOut(logits=self.lm_head(h), hidden=h)

    def generate(self, prompt_tokens: jax.Array, cfg: DecodeCfg) -> GenerateOut:
        B, T_in = prompt_tokens.shape
        N = cfg.max_new_tokens
        L = T_in + N
        D = self.cfg.hidden_size
        temperature = max(cfg.temperature, 1e-6)
        greedy = cfg.greedy

        buf0 = jnp.zeros((B, L), dtype=prompt_tokens.dtype).at[:, :T_in].set(prompt_tokens)
        hidden0 = jnp.zeros((B, N, D))
        logp0 = jnp.zeros((B, N))
        rng0 = cfg.rng if cfg.rng is not None else jax.random.key(0)
        batch_idx = jnp.arange(B)

        def step(carry, t):
            buf, rng, hidden_buf, lp_buf = carry
            out = self.forward(buf)
            ctx = T_in - 1 + t
            last_logits = out.logits[:, ctx]
            if greedy:
                next_tok = jnp.argmax(last_logits, axis=-1)
                new_rng = rng
            else:
                new_rng, sub = jax.random.split(rng)
                next_tok = jax.random.categorical(sub, last_logits / temperature)
            lp = jax.nn.log_softmax(last_logits, axis=-1)
            tok_lp = lp[batch_idx, next_tok]
            buf = buf.at[:, T_in + t].set(next_tok)
            hidden_buf = hidden_buf.at[:, t].set(out.hidden[:, ctx])
            lp_buf = lp_buf.at[:, t].set(tok_lp)
            return (buf, new_rng, hidden_buf, lp_buf), None

        (buf, _, hidden, logprobs), _ = jax.lax.scan(
            step, (buf0, rng0, hidden0, logp0), jnp.arange(N)
        )
        gen_tokens = buf[:, T_in:]
        return GenerateOut(tokens=gen_tokens, hidden=hidden, logprobs=logprobs)

    def params(self) -> Any:
        return nnx.state(self, nnx.Param)

    def apply_grads(self, grads: Any, lr: float) -> None:  # pragma: no cover - overwritten by trainer
        raise NotImplementedError(
            "NanoLM does not own the optimizer; use cts.train.loop which wraps Optax."
        )


def default_nano_lm(seed: int = 0) -> NanoLM:
    return NanoLM(NanoLMConfig(), seed=seed)
