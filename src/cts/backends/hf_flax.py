"""Thin HF Flax wrapper conforming to :class:`LocalModelAPI`.

Kept optional — imported lazily so the base install does not require
``transformers``. Used for small-model local runs (e.g. distilgpt2) beyond the
nano-LM.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .local_api import DecodeCfg, ForwardOut, GenerateOut


class HFFlaxLM:
    """Wraps a ``transformers`` Flax causal LM to match :class:`LocalModelAPI`."""

    def __init__(self, model_name: str = "distilgpt2"):
        try:
            from transformers import AutoTokenizer, FlaxAutoModelForCausalLM
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "HFFlaxLM requires `transformers`. Install `cts[hf]`."
            ) from e
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = FlaxAutoModelForCausalLM.from_pretrained(model_name)
        self.model_name = model_name

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def forward(self, tokens: jax.Array) -> ForwardOut:
        out = self.model(tokens, output_hidden_states=True)
        hidden = jnp.asarray(out.hidden_states[-1])
        return ForwardOut(logits=jnp.asarray(out.logits), hidden=hidden)

    def generate(self, prompt_tokens: jax.Array, cfg: DecodeCfg) -> GenerateOut:
        # Minimal token-by-token loop so we can capture hidden states per step.
        tokens = prompt_tokens
        hiddens: list[jax.Array] = []
        logps: list[jax.Array] = []
        rng = cfg.rng if cfg.rng is not None else jax.random.PRNGKey(0)
        for _ in range(cfg.max_new_tokens):
            fo = self.forward(tokens)
            last = fo.logits[:, -1]
            if cfg.greedy:
                nxt = jnp.argmax(last, -1)
            else:
                rng, sub = jax.random.split(rng)
                nxt = jax.random.categorical(sub, last / max(cfg.temperature, 1e-6))
            lp = jax.nn.log_softmax(last, -1)
            logps.append(lp[jnp.arange(tokens.shape[0]), nxt])
            hiddens.append(fo.hidden[:, -1])
            tokens = jnp.concatenate([tokens, nxt[:, None]], axis=-1)
        gen = tokens[:, prompt_tokens.shape[-1]:]
        return GenerateOut(
            tokens=gen,
            hidden=jnp.stack(hiddens, axis=1),
            logprobs=jnp.stack(logps, axis=1),
        )

    def params(self) -> Any:
        return self.model.params

    def apply_grads(self, grads: Any, lr: float) -> None:  # pragma: no cover
        raise NotImplementedError("HFFlaxLM is for eval / smoke only; wire Optax externally.")
