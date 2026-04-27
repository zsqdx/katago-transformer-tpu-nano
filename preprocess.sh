#!/bin/bash
# Preprocess KataGo NPZ training data for faster loading.
#
# Usage:
#   ./preprocess.sh <input_base_dir> <output_base_dir> [options...]
#
# Example:
#   ./preprocess.sh ../data/shuffleddata/kata1 ../data/preprocessed/kata1 \
#       --pos-len 19 --symmetry-type xyt --symmetry-mode expand --workers 8

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_base_dir> <output_base_dir> [preprocess.py options...]"
    echo ""
    echo "Preprocesses train/ (with symmetry) and val/ (no symmetry, unpackbits only)."
    echo "Extra arguments are passed to preprocess.py for the train set."
    exit 1
fi

INPUT_BASE="$1"
OUTPUT_BASE="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Save all extra args for train
ALL_ARGS=("$@")

# Parse --pos-len, --workers, and history options for val (defaults if not provided)
POS_LEN="19"
WORKERS="4"
ENABLE_HISTORY=""
NUM_GLOBAL_FEATURES=""
while [ $# -gt 0 ]; do
    case "$1" in
        --pos-len) POS_LEN="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --enable-history-matrices) ENABLE_HISTORY="--enable-history-matrices"; shift ;;
        --num-global-features) NUM_GLOBAL_FEATURES="--num-global-features $2"; shift 2 ;;
        *) shift ;;
    esac
done

# Train: use user-specified symmetry options
train_dir="${INPUT_BASE}/train"
if [ -d "$train_dir" ]; then
    echo "=== Preprocessing train/ ==="
    python3 "${SCRIPT_DIR}/preprocess.py" \
        --input-dir "$train_dir" \
        --output-dir "${OUTPUT_BASE}/train" \
        "${ALL_ARGS[@]}"
else
    echo "Skipping train/ (not found: ${train_dir})"
fi

# Val: only unpackbits, no symmetry augmentation
val_dir="${INPUT_BASE}/val"
if [ -d "$val_dir" ]; then
    echo "=== Preprocessing val/ (no symmetry) ==="
    python3 "${SCRIPT_DIR}/preprocess.py" \
        --input-dir "$val_dir" \
        --output-dir "${OUTPUT_BASE}/val" \
        --pos-len "$POS_LEN" \
        --workers "$WORKERS" \
        --symmetry-type none \
        $ENABLE_HISTORY $NUM_GLOBAL_FEATURES
else
    echo "Skipping val/ (not found: ${val_dir})"
fi

echo "=== All done ==="
