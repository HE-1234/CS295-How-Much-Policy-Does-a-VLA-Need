#!/usr/bin/env bash
# Gate: does PyTorch actually see and use this GPU?
#
# The Blackwell RTX PRO 6000 reports compute capability sm_120 which older
# PyTorch wheels (built for sm_50..sm_90) do not support. If this script
# prints a warning or errors out, install a newer PyTorch before continuing:
#
#   uv pip install --upgrade torch torchvision \
#     --index-url https://download.pytorch.org/whl/cu126
#
# Then re-run this script.

set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/sweep/env.sh
cd "${REPO_ROOT}"

uv run python - <<'PY'
import torch
print(f"torch={torch.__version__}  cuda={torch.version.cuda}")
print(f"device_count={torch.cuda.device_count()}")
if torch.cuda.device_count() == 0:
    raise SystemExit("No CUDA device visible to PyTorch.")
print(f"device_0={torch.cuda.get_device_name(0)}")
print(f"capability={torch.cuda.get_device_capability(0)}")
# Force a real kernel launch — this is where sm_120 incompat actually errors.
x = torch.randn(1024, 1024, device="cuda")
y = x @ x
torch.cuda.synchronize()
print(f"matmul OK, result.device={y.device}, sum={y.sum().item():.2f}")
print()
print("GPU gate PASSED.")
PY
