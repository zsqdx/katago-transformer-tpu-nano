#!/bin/bash
set -euo pipefail

# TPU v6e-1 real-data smoke training.
#
# Expected data layout for train.py:
#   DATADIR/
#     train/*.npz
#     val/*.npz
#
# If this repo only has ./val with real data, this script creates a temporary
# ./.tpu_real_data directory where both train/ and val/ point at ./val. That is
# useful for testing the full training pipeline, but not for real experiments.

cd "$(dirname "$0")"

export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"

if [ -n "${DATADIR:-}" ]; then
    DATA_ROOT="${DATADIR}"
elif [ -d "./train" ] && [ -d "./val" ]; then
    DATA_ROOT="."
elif [ -d "./val" ]; then
    DATA_ROOT="./.tpu_real_data"
    rm -rf "${DATA_ROOT}"
    mkdir -p "${DATA_ROOT}"
    ln -s "$(pwd)/val" "${DATA_ROOT}/train"
    ln -s "$(pwd)/val" "${DATA_ROOT}/val"
    echo "Using ./val for both train/ and val/ under ${DATA_ROOT} (smoke test only)."
else
    echo "ERROR: expected DATADIR with train/ and val/, or a local ./val directory." >&2
    exit 1
fi

EXTRA_FLAGS=()
if [ "${ENABLE_HISTORY_MATRICES:-1}" != "0" ]; then
    EXTRA_FLAGS+=(--enable-history-matrices)
fi

python -u train.py \
    --device xla \
    --traindir "${TRAINDIR:-./tpu_real_run}" \
    --datadir "${DATA_ROOT}" \
    --pos-len "${POS_LEN:-19}" \
    --batch-size "${BATCH_SIZE:-16}" \
    --model-kind "${MODEL_KIND:-b12c192}" \
    --lr "${LR:-2e-4}" \
    --lr-schedule "${LR_SCHEDULE:-constant}" \
    --max-training-samples "${MAX_TRAINING_SAMPLES:-1024}" \
    --symmetry-type "${SYMMETRY_TYPE:-xyt}" \
    --print-every "${PRINT_EVERY:-10}" \
    --save-every-samples "${SAVE_EVERY_SAMPLES:-1024}" \
    --val-every-samples "${VAL_EVERY_SAMPLES:-1024}" \
    --max-val-batches "${MAX_VAL_BATCHES:-4}" \
    --warmup-samples "${WARMUP_SAMPLES:-256}" \
    --prefetch-batches 0 \
    --no-compile \
    --no-tensorboard \
    --allow-nonfull-mask \
    --xla-peak-tflops "${XLA_PEAK_TFLOPS:-918}" \
    --amp-dtype "${AMP_DTYPE:-bf16}" \
    --seed "${SEED:-1234}" \
    "${EXTRA_FLAGS[@]}" \
    "$@"
