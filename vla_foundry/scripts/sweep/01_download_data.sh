#!/usr/bin/env bash
# Download PickAndPlaceBox preprocessed shards (~1.2 GB) from the public registry.
# Idempotent: skips files that already exist with the correct size.

set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/sweep/env.sh
cd "${REPO_ROOT}"

DATA_DIR="${EXP_ROOT}/data/${TASK}"
mkdir -p "${DATA_DIR}"

uv run python vla_foundry/data/scripts/download_dataset.py \
  --task "${TASK}" \
  --local_path "${DATA_DIR}"

echo
echo "--- Inventory ---"
ls -lh "${DATA_DIR}"/manifest.jsonl "${DATA_DIR}"/stats.json
ls "${DATA_DIR}"/shards/ | wc -l | xargs -I{} echo "{} shards under ${DATA_DIR}/shards/"
echo
echo "Data download PASSED."
