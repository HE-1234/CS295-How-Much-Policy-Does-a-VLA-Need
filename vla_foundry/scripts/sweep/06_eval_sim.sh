#!/usr/bin/env bash
# Closed-loop sim evaluation — REQUIRES DOCKER.
# Skip this script if you don't have Docker (rely on 05_collect_val_losses.sh
# for an open-loop proxy signal).
#
# Image is ~several GB; pulled once and cached.

set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/sweep/env.sh
cd "${REPO_ROOT}"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not installed. Closed-loop sim is unavailable." >&2
  echo "Use scripts/sweep/05_collect_val_losses.sh for an open-loop proxy." >&2
  exit 1
fi

docker pull "${DOCKER_IMAGE}"

mkdir -p "${EXP_ROOT}/rollouts"

for SIZE_PRESET in "${RUNGS[@]}"; do
  # Experiment dir for this rung — no timestamp subdir because --name is
  # passed explicitly during training.
  CKPT_DIR="${EXP_ROOT}/runs/${TASK}_${SIZE_PRESET}_budget-${BUDGET}/policy_size_${SIZE_PRESET}"
  if [[ ! -d "${CKPT_DIR}/checkpoints" ]]; then
    echo "WARN: no trained checkpoint for ${SIZE_PRESET}, skipping" >&2
    continue
  fi

  echo
  echo "============================================================"
  echo "  Evaluating ${SIZE_PRESET}"
  echo "  Checkpoint: ${CKPT_DIR}"
  echo "============================================================"

  uv run python vla_foundry/eval/run_evaluation.py "${CKPT_DIR}" \
    --model_name "${SIZE_PRESET}" \
    --tasks "${TASK}" \
    --num_episodes "${EVAL_NUM_EPISODES}" \
    --max_sample_size "${EVAL_MAX_SAMPLE_SIZE}" \
    --num_gpus "${EVAL_NUM_GPUS}" \
    --output_dir "${EXP_ROOT}/rollouts"
done

echo
echo "All rungs evaluated. Open the dashboard:"
echo "  bash scripts/sweep/07_open_dashboard.sh"
