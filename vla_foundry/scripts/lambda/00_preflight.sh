#!/usr/bin/env bash
# Preflight: confirm THIS rented box can actually run the Drake sim BEFORE you
# spend GPU-hours. If any check fails, stop and pick a different instance type.
set -uo pipefail
cd "$(dirname "$0")/../.."

echo "============================================================"
echo "  Lambda box compatibility preflight"
echo "============================================================"

fail=0

# 1. Architecture must be x86_64 (the lbm-eval image is linux/amd64).
arch="$(uname -m)"
echo -n "[1] arch is x86_64           ... "
if [[ "$arch" == "x86_64" ]]; then echo "OK ($arch)"; else echo "FAIL ($arch) - avoid ARM/GH200"; fail=1; fi

# 2. GPU render node for Drake EGL rendering. THE most common cloud blocker.
echo -n "[2] /dev/dri/renderD128      ... "
if ls /dev/dri/renderD128 >/dev/null 2>&1; then echo "OK"; else echo "FAIL - no render node, Drake can't render"; fail=1; fi

# 3. Docker present.
echo -n "[3] docker installed         ... "
if command -v docker >/dev/null 2>&1; then echo "OK"; else echo "FAIL - install Docker"; fail=1; fi

# 4. Docker can see the GPU via the NVIDIA runtime.
echo -n "[4] docker + GPU runtime     ... "
if docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
  echo "OK"
else
  echo "FAIL - NVIDIA Container Toolkit not working"; fail=1
fi

# 5. VRAM sanity (policy server + sim share one GPU; want >=24GB, 40GB+ comfy).
echo -n "[5] VRAM per GPU             ... "
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/        /'
else
  echo "WARN - nvidia-smi not found"
fi

echo "------------------------------------------------------------"
if [[ "$fail" -eq 0 ]]; then
  echo "PREFLIGHT PASSED - this box can run the sim. Continue to 01_setup.sh"
else
  echo "PREFLIGHT FAILED - do NOT proceed. Re-launch a different instance type."
  exit 1
fi
