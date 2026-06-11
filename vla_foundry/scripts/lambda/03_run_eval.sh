#!/usr/bin/env bash
# Full eval: every (task, size) checkpoint on its own finetuned task.
# 3 tasks x 3 sizes = 9 separate evaluations (each is a distinct HF checkpoint).
#
# Results layout: rollouts/<size>/<Task>/<timestamp>/results.json
#   -> the dashboard then compares 77m vs 205m vs 410m within each task.
#
# RUN THIS UNDER tmux/screen - a 0:200 sweep is many hours and an SSH drop
# would otherwise kill it:   tmux new -s eval   (Ctrl-b d to detach)
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/lambda/env.sh

mkdir -p "$OUTPUT_DIR" && chmod 777 "$OUTPUT_DIR"

echo "============================================================"
echo "  Full sweep: ${#TASKS[@]} tasks x ${#SIZES[@]} sizes"
echo "  episodes=$NUM_EPISODES  num_gpus=$NUM_GPUS  out=$OUTPUT_DIR"
echo "============================================================"

for task in "${TASKS[@]}"; do
  for size in "${SIZES[@]}"; do
    repo="$(hf_repo_for "$task" "$size")"
    echo
    echo ">>> [$task / $size]  checkpoint: $repo"
    uv run python vla_foundry/eval/run_evaluation.py "$repo" \
      --model_name "$size" \
      --tasks "$task" \
      --num_episodes "$NUM_EPISODES" \
      --max_sample_size "$MAX_SAMPLE_SIZE" \
      --num_gpus "$NUM_GPUS" \
      --output_dir "$OUTPUT_DIR"
  done
done

echo
echo "Sweep done. Inspect locally with the dashboard:"
echo "  uv run --group dashboard python vla_foundry/eval/results_explorer.py $OUTPUT_DIR"
echo "Or sync results off the box: bash scripts/lambda/04_sync_results.sh <dest>"
