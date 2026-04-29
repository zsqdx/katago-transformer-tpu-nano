#!/bin/bash
set -euo pipefail

# Convenience wrapper for searching batch size around the current best
# single-chip TPU shape.

cd "$(dirname "$0")"

MODEL_KIND_VALUE="${MODEL_KIND:-b24c2048}"
if [ -n "${BATCH_SIZES:-}" ]; then
    BATCH_SIZES_VALUE="${BATCH_SIZES}"
elif [ "${MODEL_KIND_VALUE}" = "b24c1024" ]; then
    BATCH_SIZES_VALUE="8 12 16 20 24 28 32 40 48 56 64 80 96 112 128 160 192"
else
    BATCH_SIZES_VALUE="16 20 22 24 26 28 32"
fi

specs=""
for batch in ${BATCH_SIZES_VALUE}; do
    specs="${specs} ${MODEL_KIND_VALUE}:${batch}"
done

SWEEP_SPECS="${SWEEP_SPECS:-${specs# }}" \
SWEEP_FAST_BF16="${SWEEP_FAST_BF16:-1}" \
bash sweep_jax_shapes.sh
