#!/usr/bin/env bash
# Package eval results so you can pull them off the (ephemeral) Lambda box before
# you terminate it. Run this ON Lambda; then pull the tarball from pyxis/laptop.
#
# By default excludes the large .mp4 recordings (keep results.json + logs).
# Pass --with-videos to include them.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/lambda/env.sh

stamp="$(date +%Y%m%d_%H%M%S)"
tarball="eval_results_${stamp}.tar.gz"

if [[ "${1:-}" == "--with-videos" ]]; then
  tar -czf "$tarball" "$OUTPUT_DIR"
else
  tar --exclude='*.mp4' -czf "$tarball" "$OUTPUT_DIR"
fi

echo "Wrote $tarball ($(du -h "$tarball" | cut -f1))"
echo
echo "Now pull it FROM pyxis or your laptop (run this THERE, not here):"
echo "  scp -i <lambda-key.pem> ubuntu@<lambda-ip>:$(pwd)/$tarball ."
echo "Then: tar -xzf $tarball  and open the dashboard locally."
