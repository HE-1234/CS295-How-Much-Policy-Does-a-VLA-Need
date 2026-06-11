#!/usr/bin/env bash
# Train all three policy size rungs sequentially on PickAndPlaceBox.
# Pass a single rung name as $1 to train just that one
# (e.g. ./04_train_rungs.sh transformer_77m).
#
# Expect ~1-2 hours per rung on a single Blackwell GPU.

set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/sweep/env.sh
cd "${REPO_ROOT}"

DATA_DIR="${EXP_ROOT}/data/${TASK}"
# Pick the right manifests based on whether we split for val loss.
if [[ -f "${DATA_DIR}/manifest_train.jsonl" ]]; then
  TRAIN_MANIFEST="${DATA_DIR}/manifest_train.jsonl"
  VAL_MANIFEST="${DATA_DIR}/manifest_val.jsonl"
else
  TRAIN_MANIFEST="${DATA_DIR}/manifest.jsonl"
  VAL_MANIFEST=""
fi

# Allow caller to pick a single rung; default to all three.
if [[ $# -ge 1 ]]; then
  RUNGS=("$1")
fi

for SIZE_PRESET in "${RUNGS[@]}"; do
  RUN="${EXP_ROOT}/runs/${TASK}_${SIZE_PRESET}_budget-${BUDGET}"
  mkdir -p "${RUN}"

  echo
  echo "============================================================"
  echo "  Training rung: ${SIZE_PRESET}    budget=${BUDGET} samples"
  echo "  Run dir: ${RUN}"
  echo "============================================================"

  CMD=(
    uv run python vla_foundry/main.py
    --config_path vla_foundry/config_presets/training_jobs/vla_pickandplace_box.yaml
    --save_path "${RUN}"
    --name "policy_size_${SIZE_PRESET}"
    --wandb False
    --db_logging False
    --distributed.fsdp False
    --hparams.torchcompile False
    --hparams.per_gpu_batch_size "${PER_GPU_BATCH_SIZE}"
    --hparams.global_batch_size "${GLOBAL_BATCH_SIZE}"
    --hparams.lr "${LR}"
    --hparams.warmup "${WARMUP}"
    --hparams.optimizer adamw
    --hparams.lr_scheduler cosine
    --total_train_samples "${BUDGET}"
    --num_checkpoints 1
    --data.num_workers 4
    --data.prefetch_factor 2
    --data.shuffle_buffer_size 400
    --data.dataset_manifest "[${TRAIN_MANIFEST}]"
    --data.dataset_statistics "[${DATA_DIR}/stats.json]"
    --model.transformer "include vla_foundry/config_presets/models/${SIZE_PRESET}.yaml"
    --model.transformer.is_causal False
    --model.transformer.vocab_size 0
  )

  # Append val-loss flags only if both the split exists and the toggle is on.
  if [[ "${WITH_VAL_LOSS}" == "1" && -n "${VAL_MANIFEST}" ]]; then
    CMD+=(
      --data.val_dataset_manifest "[${VAL_MANIFEST}]"
      --data.val_dataset_statistics "[${DATA_DIR}/stats.json]"
      --data.val_dataset_weighting "[1.0]"
      --total_val_samples "${VAL_SAMPLES}"
      --val_every_n_checkpoints "${VAL_EVERY_N_CHECKPOINTS}"
    )
    echo "  WITH_VAL_LOSS=1 → val loss logged every ${VAL_EVERY_N_CHECKPOINTS} checkpoint(s)"
  fi

  "${CMD[@]}"
done

echo
echo "All requested rungs trained. Checkpoints under:"
echo "  ${EXP_ROOT}/runs/${TASK}_*_budget-${BUDGET}/policy_size_*/<timestamp>/checkpoints/"
