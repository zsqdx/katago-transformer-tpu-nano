#!/bin/bash
set -euo pipefail

# Convenience wrapper for searching batch size around the current best
# single-chip TPU shape.

cd "$(dirname "$0")"

MODEL_KIND_VALUE="${MODEL_KIND:-b24c2048}"
BATCH_SIZES_VALUE="${BATCH_SIZES:-16 20 22 24 26 28 32}"

specs=""
for batch in ${BATCH_SIZES_VALUE}; do
    specs="${specs} ${MODEL_KIND_VALUE}:${batch}"
done

SWEEP_SPECS="${SWEEP_SPECS:-${specs# }}" \
SWEEP_FAST_BF16="${SWEEP_FAST_BF16:-1}" \
bash sweep_jax_shapes.sh
