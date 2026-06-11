#!/usr/bin/env bash
# Split the PickAndPlaceBox manifest into train (first 46 shards) and val
# (remaining ~6 shards). Only needed if WITH_VAL_LOSS=1.
# Idempotent: overwrites the split manifests every run.

set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/sweep/env.sh
cd "${REPO_ROOT}"

DATA_DIR="${EXP_ROOT}/data/${TASK}"
MANIFEST="${DATA_DIR}/manifest.jsonl"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: ${MANIFEST} not found. Run 01_download_data.sh first." >&2
  exit 1
fi

TOTAL=$(wc -l < "${MANIFEST}")
TRAIN_N=$(( TOTAL * 90 / 100 ))   # ~90/10 split

head -n "${TRAIN_N}" "${MANIFEST}" > "${DATA_DIR}/manifest_train.jsonl"
tail -n +"$((TRAIN_N+1))" "${MANIFEST}" > "${DATA_DIR}/manifest_val.jsonl"

echo "Train: $(wc -l < "${DATA_DIR}/manifest_train.jsonl") shards → ${DATA_DIR}/manifest_train.jsonl"
echo "Val:   $(wc -l < "${DATA_DIR}/manifest_val.jsonl") shards → ${DATA_DIR}/manifest_val.jsonl"
echo
echo "Val split PASSED."
