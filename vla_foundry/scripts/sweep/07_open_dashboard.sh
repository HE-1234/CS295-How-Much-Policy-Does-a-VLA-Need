#!/usr/bin/env bash
# Launch the streamlit dashboard at http://localhost:8505.
# Reads results.json files written by 06_eval_sim.sh.

set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/sweep/env.sh
cd "${REPO_ROOT}"

uv run --group dashboard python vla_foundry/eval/results_explorer.py \
  "${EXP_ROOT}/rollouts/"
