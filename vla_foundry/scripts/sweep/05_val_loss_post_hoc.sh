#!/usr/bin/env bash
# Post-hoc validation: for each rung's saved checkpoint(s), compute val loss
# on the held-out manifest_val.jsonl shards. Decoupled from training so train
# runs at full throughput.
#
# Output: writes ${EXP_ROOT}/val_losses.csv automatically (one row per
# rung × checkpoint) and also echoes rows to stdout for live progress.
# Override the destination by passing a path as the first argument:
#
#   bash scripts/sweep/05_val_loss_post_hoc.sh                     # → $EXP_ROOT/val_losses.csv
#   bash scripts/sweep/05_val_loss_post_hoc.sh /tmp/other.csv      # → /tmp/other.csv

set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/sweep/env.sh
cd "${REPO_ROOT}"

DATA_DIR="${EXP_ROOT}/data/${TASK}"
VAL_MANIFEST="${DATA_DIR}/manifest_val.jsonl"
VAL_STATS="${DATA_DIR}/stats.json"

if [[ ! -f "${VAL_MANIFEST}" ]]; then
  echo "ERROR: ${VAL_MANIFEST} missing. Run 02_split_val.sh first." >&2
  exit 1
fi

OUT_CSV="${1:-${EXP_ROOT}/val_losses.csv}"
mkdir -p "$(dirname "${OUT_CSV}")"
echo "Writing val losses to: ${OUT_CSV}" >&2

# Header
printed_header=0

{
  for SIZE_PRESET in "${RUNGS[@]}"; do
    RUN_DIR="${EXP_ROOT}/runs/${TASK}_${SIZE_PRESET}_budget-${BUDGET}/policy_size_${SIZE_PRESET}"
    CONFIG="${RUN_DIR}/config.yaml"
    CKPT_DIR="${RUN_DIR}/checkpoints"

    if [[ ! -f "${CONFIG}" ]]; then
      echo "WARN: ${CONFIG} not found (rung ${SIZE_PRESET} not trained?), skipping" >&2
      continue
    fi

    # Score every checkpoint_N.pt the rung wrote (1 for num_checkpoints=1; more
    # if you ever raise it). Sorted by numeric N for deterministic CSV order.
    mapfile -t CKPTS < <(ls "${CKPT_DIR}"/checkpoint_*.pt 2>/dev/null | sort -V)
    if [[ ${#CKPTS[@]} -eq 0 ]]; then
      echo "WARN: no checkpoints under ${CKPT_DIR}, skipping ${SIZE_PRESET}" >&2
      continue
    fi

    for CKPT in "${CKPTS[@]}"; do
      HEADER_FLAG=""
      if [[ ${printed_header} -eq 0 ]]; then
        HEADER_FLAG="--header"
        printed_header=1
      fi
      uv run python scripts/sweep/val_loss_post_hoc.py \
        --config_path "${CONFIG}" \
        --checkpoint  "${CKPT}" \
        --val_manifest "${VAL_MANIFEST}" \
        --val_stats    "${VAL_STATS}" \
        --val_samples  "${VAL_SAMPLES}" \
        --rung_name    "${SIZE_PRESET}" \
        ${HEADER_FLAG}
    done
  done
} | tee "${OUT_CSV}"

echo "Done. CSV at: ${OUT_CSV}" >&2
