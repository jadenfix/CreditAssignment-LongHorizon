# Contributor notes (CTS)

## Rules

- **Layout**: source under `src/cts/`, tests under `tests/`, all generated artifacts under `artifacts/` (git-ignored).
- **Replay-only data**: anything in `src/cts/train/` or `src/cts/methods/` MUST consume `Quadruple` from a `ReplayShard` / `ReplayDir`. Never call a teacher API from a trainer.
- **Two backend abstractions**: `LocalModelAPI` (smoke / unit tests) and `ScaleBackendAdapter` (Tunix / MaxText). Don't merge them.
- **Method shape**: every method in `cts.methods` exposes `step(model, batch[, cfg]) -> (loss, metrics)`. Ablations are config toggles on top of `b5_cts`, never new methods.
- **Fairness**: any sweep that varies methods MUST keep `cts.train.fairness.fingerprint_from_cfg` equal across runs. The loader will refuse to launch otherwise.

## Conventions

- Format / lint: `uv run ruff check src tests && uv run ruff format src tests`.
- Type-check: `uv run pyright`.
- Tests: `uv run pytest -q`. Mark slow paths with `@pytest.mark.slow`.
- No emojis or decorative comments. Comments only when *why* is non-obvious.

## Make targets

- `make setup` / `make setup-all` — sync deps (core / + hf, teacher, gcp).
- `make fmt` / `make lint` / `make type` / `make test` — the usual.
- `make smoke` — `scripts/smoke_all_methods.sh`.
- `make matrix-local` — run the paper matrix on CPU fixtures + aggregate.
- `make teacher-data` — generate the full replay shards (needs API keys).
- `make docker-cpu` / `make docker-tpu` — build deploy images locally.

## `deploy/` boundary

Anything under `deploy/` is launchers, Dockerfiles, and Cloud Build config — **no
algorithm code**. The CPU image runs CI smoke; the TPU image is the actual training
target. `deploy/launch_tpu.sh` and `deploy/launch_vertex.py` are the only entry
points; both honor a `DRY_RUN=1` env to print the underlying `gcloud` invocations.

## Common workflows

Generate teacher data into a replay shard:
```
uv run python -m cts.data.teacher.cli --task gsm8k --split train \
  --teacher anthropic --out artifacts/replay/gsm8k/train.jsonl --limit 200
```

Run a method end-to-end (local nano-LM):
```
uv run python -m cts.train.main method=cts task=fixtures_math backend=local_nano \
  trainer.max_steps=20
```

Smoke every method:
```
bash scripts/smoke_all_methods.sh
```

Eval on a fixed test split:
```
uv run python -m cts.eval.run_eval \
  --replay src/cts/data/fixtures/gsm8k_tiny.jsonl \
  --split test --domain math \
  --out artifacts/results/fixtures/results.json
```

## Adding an ablation

1. Add a single config field under `configs/method/cts.yaml`.
2. Branch on it inside `b5_cts.step` (or upstream batch construction).
3. Add a test under `tests/integration/test_methods_smoke.py` that exercises both states of the toggle.

## Adding a new task

1. Loader under `src/cts/data/<task>.py` that emits **shells** (empty `y0/f/y1`).
2. Verifier wired into `cts.rl.verifier`.
3. Config under `src/cts/configs/task/<task>.yaml`.
4. Generate quadruples via the teacher CLI; commit a *tiny* fixture under `src/cts/data/fixtures/`.
