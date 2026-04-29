#!/bin/bash
set -euo pipefail

# Convenience wrapper for searching batch size around the current best
# single-chip TPU shape.

cd "$(dirname "$0")"

MODEL_KIND_VALUE="${MODEL_KIND:-b24c2048}"
OPTIMIZER_VALUE="${OPTIMIZER:-adamw}"
if [ "${OPTIMIZER_VALUE}" = "muon" ]; then
    export MUON_TARGET="${MUON_TARGET:-attn}"
    export MUON_POLAR_STEPS="${MUON_POLAR_STEPS:-3}"
    export MUON_ROW_SPLIT_SIZE="${MUON_ROW_SPLIT_SIZE:-64}"
    export MUON_GROUP_BLOCKS="${MUON_GROUP_BLOCKS:-0}"
fi

if [ -n "${BATCH_SIZES:-}" ]; then
    BATCH_SIZES_VALUE="${BATCH_SIZES}"
elif [ "${MODEL_KIND_VALUE}" = "b24c1024" ]; then
    if [ "${OPTIMIZER_VALUE}" = "muon" ]; then
        BATCH_SIZES_VALUE="10 12 14 15 16 17 18 19 20 22 24 28 32"
    else
        BATCH_SIZES_VALUE="8 12 16 20 24 28 32 40 48 56 64 80 96 112 128 160 192"
    fi
else
    BATCH_SIZES_VALUE="16 20 22 24 26 28 32"
fi

specs=""
for batch in ${BATCH_SIZES_VALUE}; do
    specs="${specs} ${MODEL_KIND_VALUE}:${batch}"
done

SWEEP_SPECS="${SWEEP_SPECS:-${specs# }}" \
SWEEP_FAST_BF16="${SWEEP_FAST_BF16:-1}" \
OPTIMIZER="${OPTIMIZER_VALUE}" \
bash sweep_jax_shapes.sh
