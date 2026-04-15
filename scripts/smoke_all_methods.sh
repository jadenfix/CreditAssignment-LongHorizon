#!/usr/bin/env bash
# Smoke all methods on nano-LM + fixtures. Not a training run, just dispatch.
set -euo pipefail
cd "$(dirname "$0")/.."

for method in sft dpo; do
  echo "=== $method ==="
  python -m cts.train.main method=$method task=fixtures_math backend=local_nano \
    trainer.max_steps=1 trainer.batch_size=2
done

echo "=== RL methods (pytest smoke) ==="
python -m pytest -q tests/integration/test_methods_smoke.py
