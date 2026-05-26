#!/usr/bin/env bash
# egs2/libritts_r/enh1/setup.sh
#
# Run this ONCE after cloning the repo to create the standard ESPnet
# symlinks that run.sh depends on.
#
# Usage:
#   cd ~/Workspace/dahee/espnet/egs2/libritts_r/enh1
#   bash setup.sh
#
# What it does:
#   1. Creates symlinks: scripts/ steps/ utils/ → TEMPLATE directories
#   2. Copies enh.sh from egs2/TEMPLATE/enh1/
#   3. Makes run.sh and local scripts executable

set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ESPNET_ROOT="$(cd "${RECIPE_DIR}/../../../" && pwd)"
TEMPLATE="${ESPNET_ROOT}/egs2/TEMPLATE/enh1"

echo "ESPnet root : ${ESPNET_ROOT}"
echo "Recipe dir  : ${RECIPE_DIR}"
echo "Template    : ${TEMPLATE}"

cd "${RECIPE_DIR}"

# ── 1. Symlinks to shared utilities ──────────────────────────────────────
for d in scripts pyscripts steps utils; do
    src="${ESPNET_ROOT}/egs2/TEMPLATE/asr1/${d}"
    # Some dirs live under enh1 template
    [ ! -d "${src}" ] && src="${TEMPLATE}/${d}"
    # Fall back to asr1 template (which has steps/ utils/)
    [ ! -d "${src}" ] && src="${ESPNET_ROOT}/egs2/TEMPLATE/asr1/${d}"
    if [ -d "${src}" ]; then
        ln -sfn "${src}" "${d}"
        echo "  linked ${d} → ${src}"
    else
        echo "  SKIP ${d} (source not found at ${src})"
    fi
done

# ── 2. enh.sh from TEMPLATE ──────────────────────────────────────────────
if [ ! -f enh.sh ]; then
    if [ -f "${TEMPLATE}/enh.sh" ]; then
        cp "${TEMPLATE}/enh.sh" enh.sh
        echo "  copied enh.sh from ${TEMPLATE}"
    else
        echo "  SKIP enh.sh (not found in template)"
    fi
fi

# ── 3. Executable permissions ────────────────────────────────────────────
chmod +x run.sh local/data.sh local/prepare_noisy_data.py \
         local/upsample_wav_scp.py local/score.py 2>/dev/null || true

echo ""
echo "Setup complete.  Now edit db.sh and run:"
echo "  ./run.sh --stage 1 --stop_stage 1"
