#!/usr/bin/env bash
# One-time setup on the rented box: install Python deps and pre-pull the sim image.
# Run from inside the cloned repo, after 00_preflight.sh passes.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/lambda/env.sh

# uv installs the exact pinned env (torch 2.7 cu128, etc.). Install it if missing.
if ! command -v uv >/dev/null 2>&1; then
  echo ">>> installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo ">>> syncing python deps (inference + dashboard groups) ..."
uv sync --group inference --group dashboard

echo ">>> pulling sim image ($DOCKER_IMAGE) - several GB, cached after first time ..."
docker pull "$DOCKER_IMAGE"

echo
echo "Setup complete. Next: bash scripts/lambda/02_smoke_test.sh"
