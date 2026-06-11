#!/usr/bin/env bash
# Smoke test: ONE checkpoint, ONE task, 2 episodes. Proves the full pipeline
# (HF pull -> policy server -> Docker sim -> results.json) end-to-end before you
# commit to the full ~36 GB / many-hour run. Takes a few minutes.
#
# Usage: bash scripts/lambda/02_smoke_test.sh [TASK] [SIZE]
#   defaults: PushBox transformer_77m
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/lambda/env.sh

TASK="${1:-PushBox}"
SIZE="${2:-transformer_77m}"
REPO="$(hf_repo_for "$TASK" "$SIZE")"

mkdir -p "$OUTPUT_DIR" && chmod 777 "$OUTPUT_DIR"   # Docker writes as a different user

echo ">>> smoke test: $REPO  task=$TASK  episodes=0:2"
uv run python vla_foundry/eval/run_evaluation.py "$REPO" \
  --model_name "$SIZE" \
  --tasks "$TASK" \
  --num_episodes "0:2" \
  --max_sample_size "$MAX_SAMPLE_SIZE" \
  --num_gpus 1 \
  --output_dir "$OUTPUT_DIR"

echo
echo "If a results.json with a real success/fail appeared under"
echo "  $OUTPUT_DIR/$SIZE/$TASK/<timestamp>/  -> the box is good."
echo "If episodes show total_time:0 + a gRPC traceback, that's an infra crash"
echo "(check the policy-server and .docker.log files printed above)."
