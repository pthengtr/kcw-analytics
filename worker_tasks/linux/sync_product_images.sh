#!/usr/bin/env bash
# Linux stand-in for run_product_image_sync.bat.
# Needs LEGACY_PRODUCT_IMAGE_DIR as a real POSIX path (SMB mount), not \\KSS\...
source "$(dirname "$0")/common.sh"
echo "Product image sync"
"$PY" "$REPO/worker_tasks/sync_product_images.py"
echo "DONE: sync_product_images"
