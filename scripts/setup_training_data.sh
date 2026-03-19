#!/usr/bin/env bash
# setup_training_data.sh
# ──────────────────────────────────────────────────────────────────────────────
# Extract the labelme dataset archive on the server and verify its structure.
#
# Usage:
#   bash scripts/setup_training_data.sh                      # looks for labelme_work.tar.gz in CWD
#   bash scripts/setup_training_data.sh /path/to/archive.tar.gz
#
# Run from the project root (same directory that contains data/ and scripts/).
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ARCHIVE="${1:-labelme_work.tar.gz}"
DATA_DIR="data"
LABELME_DIR="$DATA_DIR/labelme_work"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if [ ! -f "$ARCHIVE" ]; then
    echo "[ERROR] Archive not found: $ARCHIVE"
    echo ""
    echo "Package it on your local machine first:"
    echo "  powershell -File scripts/package_labelme_dataset.ps1"
    echo ""
    echo "Then transfer it to the server, e.g.:"
    echo "  scp labelme_work.tar.gz user@server:/path/to/TERRA_Coastline_DT/"
    exit 1
fi

# ── Extract ───────────────────────────────────────────────────────────────────
echo "Extracting $ARCHIVE  →  $LABELME_DIR ..."
mkdir -p "$DATA_DIR"
tar -xzf "$ARCHIVE" -C "$DATA_DIR"
echo "Extraction complete."
echo ""

# ── Verify structure ──────────────────────────────────────────────────────────
echo "Verifying dataset structure:"
echo "──────────────────────────────────────────────────"

EXPECTED_SUBDIRS=(
    "aoi_01_02_2016_2020"
    "aoi_03_04_2016_2020"
    "aoi_05_06_2016_2020"
    "aoi_07_08_2016_2020"
    "aoi_09_2016_2020"
    "ve_pred_2021_2026"
)

all_ok=true
for subdir in "${EXPECTED_SUBDIRS[@]}"; do
    path="$LABELME_DIR/$subdir"
    if [ -d "$path" ]; then
        count=$(find "$path" -name "*.json" | wc -l)
        printf "  %-35s %4d JSON files\n" "$subdir" "$count"
    else
        printf "  %-35s  MISSING\n" "$subdir"
        all_ok=false
    fi
done

echo "──────────────────────────────────────────────────"
total=$(find "$LABELME_DIR" -name "*.json" 2>/dev/null | wc -l)
echo "  Total JSON files: $total"
echo ""

if [ "$all_ok" = false ]; then
    echo "[WARN] Some expected subdirectories are missing — check your archive."
elif [ "$total" -lt 300 ]; then
    echo "[WARN] Only $total JSON files found; expected ~700+ (original 330 + corrected 2021–2026)."
else
    echo "[OK] Dataset looks complete. Ready to train."
fi
