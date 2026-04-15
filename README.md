# Critique-Transport Smoothing

**Critique-Transport Smoothing: Improving Credit Assignment in LLMs via Latent Trajectory Alignment.**

This repo is the JAX-native research framework for the paper. It tests a single, narrow question:

> Does critique-localized latent trajectory alignment improve delayed credit assignment beyond a strong GRPO + verifier baseline on long-horizon reasoning tasks?

The framework supports the full baseline ladder (SFT revision, DPO, GRPO outcome-only, GRPO + verifier, GRPO + critique reward, CTS) on math (GSM8K) and code (APPS, primary; MBPP for smoke) and includes the CTS auxiliary loss: a critique-localized latent-edit pseudo-target plus a time-aware trajectory alignment loss (Sinkhorn or soft-DTW) plus a decode anchor.

## Stack

JAX · Flax NNX · Optax · Orbax · OTT-JAX (entropic / unbalanced Sinkhorn) · Tunix (SFT/DPO/PPO/GRPO trainers) · MaxText (scale path) · HuggingFace `datasets` + Flax `transformers` · OmegaConf · pytest.

## Two backends

`LocalModelAPI` — the smoke / unit-test path (nano-LM, optional HF Flax). Lets every loss and method run on a laptop.
`ScaleBackendAdapter` — the sharding-aware path. Two impls: `TunixAdapter` (M8) and `MaxTextAdapter` (M9). Methods are written against one side or the other; we deliberately do **not** force them through a single protocol.

## Quickstart

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,ot,hf]"
uv run pytest -q                              # unit + integration on fixtures
bash scripts/smoke_all_methods.sh             # nano-LM smoke for every method
uv run python -m cts.train.main \
  method=cts task=fixtures_math backend=local_nano trainer.max_steps=20
```

Real data (requires API keys for the teacher pass):

```bash
uv pip install -e ".[teacher]"
uv run python -m cts.data.teacher.cli \
  --task gsm8k --split train --teacher anthropic \
  --out artifacts/replay/gsm8k/train.jsonl --limit 200
uv run python -m cts.train.main \
  method=cts task=gsm8k backend=tunix trainer.max_steps=200    # M8
```

Scale path (MaxText, M9):

```bash
uv pip install -e ".[scale]"
# Add MaxText to PYTHONPATH; configure src/cts/configs/backend/maxtext.yaml.
bash scripts/run_maxtext_sweep.sh             # METHOD=cts TASK=gsm8k STEPS=20
```

## Directory map

```
src/cts/
  configs/                 base + backend/{local_nano,local_hf,tunix,maxtext} + task/* + method/*
  data/                    schema (Quadruple), replay (the only trainer interface),
                           fixtures/ (zero-key smoke data), gsm8k.py, apps.py, mbpp.py,
                           teacher/{anthropic,openai,cli}, taxonomy, splits
  backends/                local_api (smoke), scale_adapter (Tunix/MaxText), nano_lm,
                           hf_flax, tunix_adapter, maxtext_adapter
  models/                  projection (P_phi), energy (E_psi + edit_trajectory)
  losses/                  decode_anchor, sinkhorn, soft_dtw, alignment dispatcher,
                           energy_loss
  rl/                      rollout, verifier (math + sandboxed code), grpo, critique_reward
  methods/                 b0_sft_revision, b1_dpo, b2_grpo_outcome,
                           b3_grpo_verifier, b4_grpo_critique, b5_cts
  eval/                    metrics (Δ_critique), horizon, stats (paired bootstrap,
                           Wilcoxon, Holm, Cohen κ), human_eval, run_eval
  train/                   loop, fairness (sweep guard), main (Hydra entry)
tests/                     unit + integration (keyless, fixture-driven)
scripts/                   smoke_all_methods.sh, gen_teacher_data.py, run_maxtext_sweep.sh
artifacts/                 generated data, checkpoints, logs (git-ignored)
```

## CTS at a glance

```
L = L_GRPO + λ₁·L_align(R_θ, R*) + λ₂·L_decode(y¹|x,y⁰,f) + λ₃·L_E

R_θ = P_φ(h_{1:T})                               # projected reasoning trajectory
R*  = R_θ - η ∇_{R_θ} U_ψ(R_θ, f)                # critique-localized latent edit
U_ψ = Σ_t α_t E_ψ(r_t, e_f)                      # energy critic on (state, critique)
L_align = time-augmented Sinkhorn or soft-DTW    # see cts.losses.alignment
```

Ablations (single config flips):

| Flag | Ablation |
|---|---|
| `cts.lam_align=0` | A1 — remove CTS aux entirely |
| `cts.alignment.kind=l2` | A2 — replace alignment with plain L2 |
| `cts.alignment.kind=soft_dtw` | A3 — soft-DTW vs Sinkhorn |
| `cts.lam_decode=0` | A4 — remove the decode anchor |
| `cts.energy.freeze=true` | A5 — freeze the energy critic |
| `cts.online_critique=true` | A6 — current-policy critique refresh |

## Fairness guard

`cts.train.fairness.fingerprint_from_cfg` extracts the compute / prompt / verifier fingerprint that *every method in a sweep must share*. The Hydra loader stamps `prompt_template_hash` automatically; `assert_consistent` is what `cts-train` calls before launching. Mismatches fail loudly.

## What this paper does **not** claim

CTS replaces language-model losses. CTS is a new universal alignment paradigm. CTS beats process supervision everywhere. (See plan §16.)
