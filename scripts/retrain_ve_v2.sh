#!/usr/bin/env bash
# retrain_ve_v2.sh
# ──────────────────────────────────────────────────────────────────────────────
# Retrain the VE UNet (v2) on the expanded dataset:
#   • data/labelme_work/aoi_*_2016_2020/   — original hand labels (2016–2020)
#   • data/labelme_work/ve_pred_2021_2026/ — corrected auto-labels (2021–2026)
#
# Train split : 2016–2023
# Val split   : 2024–2026  (temporal generalisation test)
# Holdout     : aoi_09     (unseen AOI)
#
# Prerequisites:
#   1. Python environment with torch, numpy, pillow, scipy activated.
#   2. Dataset extracted:  bash scripts/setup_training_data.sh
#
# Usage:
#   bash scripts/retrain_ve_v2.sh              # auto-detect GPU
#   bash scripts/retrain_ve_v2.sh --device cpu # force CPU (slow)
#
# Any extra arguments are forwarded to train_ve_unet.py, e.g.:
#   bash scripts/retrain_ve_v2.sh --batch-size 4 --epochs 150
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

LABELME_ROOT="data/labelme_work"
OUTPUT_DIR="data/models/ve_unet_v2"
LOG_FILE="$OUTPUT_DIR/train.log"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if [ ! -d "$LABELME_ROOT" ]; then
    echo "[ERROR] Dataset not found at $LABELME_ROOT"
    echo "Run first: bash scripts/setup_training_data.sh labelme_work.tar.gz"
    exit 1
fi

n_json=$(find "$LABELME_ROOT" -name "*.json" | wc -l)
echo "Dataset   : $LABELME_ROOT  ($n_json JSON files)"

if [ "$n_json" -lt 100 ]; then
    echo "[ERROR] Too few samples ($n_json). Check your dataset."
    exit 1
fi

if [ "$n_json" -lt 500 ]; then
    echo "[WARN] Only $n_json samples — expected ~700+ with the 2021–2026 labels."
    echo "       Training will proceed but results may be weaker."
fi

# ── GPU check ─────────────────────────────────────────────────────────────────
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "GPU       : $GPU_NAME"
    DEVICE="cuda"
else
    echo "[WARN] CUDA not available — training on CPU (this will be slow)."
    DEVICE="cpu"
fi

mkdir -p "$OUTPUT_DIR"
echo "Output    : $OUTPUT_DIR"
echo "Log       : $LOG_FILE"
echo ""

# ── Train ─────────────────────────────────────────────────────────────────────
echo "Starting training  $(date '+%Y-%m-%d %H:%M:%S')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python scripts/train_ve_unet.py \
    --labelme-root   "$LABELME_ROOT"  \
    --output-dir     "$OUTPUT_DIR"    \
    --val-start-year 2024             \
    --exclude-aois   aoi_09           \
    --epochs         100              \
    --batch-size     8                \
    --lr             1e-4             \
    --weight-decay   1e-4             \
    --pos-weight     5.0              \
    --early-stop-patience 15          \
    --grad-clip      1.0              \
    --device         "$DEVICE"        \
    "$@" \
    2>&1 | tee "$LOG_FILE"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Done  $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Best checkpoint : $OUTPUT_DIR/ve_robust_unet_best.pth"
echo "Summary         : $OUTPUT_DIR/train_summary.json"
echo "Full log        : $LOG_FILE"
