#!/usr/bin/env bash
# Scale-path scaffold. Requires MaxText on PYTHONPATH + Tunix installed.
# Real wiring lands in milestone M9 (see cts.backends.maxtext_adapter).
set -euo pipefail
cd "$(dirname "$0")/.."

METHOD="${METHOD:-cts}"
TASK="${TASK:-gsm8k}"
STEPS="${STEPS:-20}"

echo "[maxtext sweep] method=$METHOD task=$TASK steps=$STEPS"
python -m cts.train.main \
  method="$METHOD" task="$TASK" backend=maxtext \
  trainer.max_steps="$STEPS"
