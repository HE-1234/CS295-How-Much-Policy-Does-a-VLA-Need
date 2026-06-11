#!/usr/bin/env bash
# Tiny end-to-end run on the full-size rung to flush out integration bugs.
# First invocation downloads the SmolVLM backbone (~6 GB) into the HF cache.
# Total wall-clock: a few minutes on a Blackwell.

set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/sweep/env.sh
cd "${REPO_ROOT}"

DATA_DIR="${EXP_ROOT}/data/${TASK}"
SMOKE="${EXP_ROOT}/smoke"
mkdir -p "${SMOKE}"

# Use the train split if it exists; else fall back to the full manifest.
if [[ -f "${DATA_DIR}/manifest_train.jsonl" ]]; then
  TRAIN_MANIFEST="${DATA_DIR}/manifest_train.jsonl"
else
  TRAIN_MANIFEST="${DATA_DIR}/manifest.jsonl"
fi

uv run python vla_foundry/main.py \
  --config_path vla_foundry/config_presets/training_jobs/vla_pickandplace_box.yaml \
  --save_path "${SMOKE}" --name smoke_full \
  --wandb False --db_logging False --distributed.fsdp False \
  --hparams.torchcompile False \
  --hparams.per_gpu_batch_size 4 --hparams.global_batch_size 16 \
  --total_train_samples 256 --num_checkpoints 1 \
  --data.dataset_manifest "[${TRAIN_MANIFEST}]" \
  --data.dataset_statistics "[${DATA_DIR}/stats.json]" \
  --model.transformer "include vla_foundry/config_presets/models/transformer_410m.yaml" \
  --model.transformer.is_causal False \
  --model.transformer.vocab_size 0

echo
echo "--- Gate checks ---"
echo "Look for these in the log above:"
echo "  1. 'Total parameters: 1,852,396,116'    (full model: backbone 1527M + policy 325M)"
echo "  2. 'Trainable parameters: 325,026,836'  (policy only — backbone freeze worked)"
echo "  3. A 'checkpoint_0.pt' file written under ${SMOKE}/smoke_full/<exp-dir>/checkpoints/"
echo
echo "If trainable ≈ total (1.85B), the freeze didn't take — debug before continuing."
echo "If GPU OOMs at batch 4, the full rung also won't fit at 16 — drop PER_GPU_BATCH_SIZE."
echo
echo "Smoke test FINISHED. Inspect log above before running 04_train_rungs.sh."
