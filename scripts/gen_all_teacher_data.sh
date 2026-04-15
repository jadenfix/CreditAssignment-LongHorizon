#!/usr/bin/env bash
# Drive the teacher CLI across (task × split) for the full paper dataset.
# Requires: ANTHROPIC_API_KEY (or set TEACHER=openai + OPENAI_API_KEY).
set -euo pipefail
cd "$(dirname "$0")/.."

TEACHER=${TEACHER:-anthropic}
OUT_ROOT=${OUT_ROOT:-artifacts/replay}

# Per-split limits. Override with TRAIN_LIMIT/VAL_LIMIT/TEST_LIMIT for smoke runs.
TRAIN_LIMIT=${TRAIN_LIMIT:-2000}
VAL_LIMIT=${VAL_LIMIT:-300}
TEST_LIMIT=${TEST_LIMIT:-600}

for task in gsm8k apps; do
  for split in train val test; do
    case "$split" in
      train) limit=$TRAIN_LIMIT;;
      val) limit=$VAL_LIMIT;;
      test) limit=$TEST_LIMIT;;
    esac
    out="$OUT_ROOT/$task/$split.jsonl"
    mkdir -p "$(dirname "$out")"
    echo "=== $task/$split (limit=$limit, teacher=$TEACHER) ==="
    uv run python -m cts.data.teacher.cli \
      --task "$task" --split "$split" --teacher "$TEACHER" \
      --out "$out" --limit "$limit"
  done
done

echo "Done. Replay shards under $OUT_ROOT."
