#!/bin/bash
set -euo pipefail

# v6e-8 Zhizi experiment entrypoint.
# BATCH_SIZE is global across the 8 local TPU devices; the default 128 means
# per-device batch 16.

cd "$(dirname "$0")"

export DATADIR="${DATADIR:-/mnt/data/datasets/shuffle-2405-2604-zhizi/main}"
export TRAINDIR="${TRAINDIR:-/mnt/ckpt/runs/zhizi-v6e8-exp001}"

export MODEL_KIND="${MODEL_KIND:-b40c768}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export DATA_PARALLEL="${DATA_PARALLEL:-1}"
export XLA_PEAK_TFLOPS="${XLA_PEAK_TFLOPS:-7344}"

export OPTIMIZER="${OPTIMIZER:-muon}"
if [ "${OPTIMIZER}" = "muon" ]; then
    export MUON_TARGET="${MUON_TARGET:-attn}"
    export MUON_POLAR_STEPS="${MUON_POLAR_STEPS:-3}"
    export MUON_ROW_SPLIT_SIZE="${MUON_ROW_SPLIT_SIZE:-64}"
    export MUON_GROUP_BLOCKS="${MUON_GROUP_BLOCKS:-0}"
fi

export ACTIVATION_DTYPE="${ACTIVATION_DTYPE:-bf16}"
export PARAM_DTYPE="${PARAM_DTYPE:-bf16}"
export OPT_STATE_DTYPE="${OPT_STATE_DTYPE:-bf16}"
export ROPE_DTYPE="${ROPE_DTYPE:-bf16}"
export FFN_MUL_DTYPE="${FFN_MUL_DTYPE:-bf16}"
export ATTENTION_LOGITS_DTYPE="${ATTENTION_LOGITS_DTYPE:-bf16}"
export DONATE_TRAIN_BUFFERS="${DONATE_TRAIN_BUFFERS:-1}"

export MAX_TRAINING_SAMPLES="${MAX_TRAINING_SAMPLES:-1048576}"
export WARMUP_SAMPLES="${WARMUP_SAMPLES:-131072}"
export SAVE_EVERY_SAMPLES="${SAVE_EVERY_SAMPLES:-262144}"
export VAL_EVERY_SAMPLES="${VAL_EVERY_SAMPLES:-262144}"
export MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-64}"
export PRINT_EVERY="${PRINT_EVERY:-20}"
export STEPS_PER_JIT="${STEPS_PER_JIT:-1}"

if [ ! -d "${DATADIR}/train" ] || [ ! -d "${DATADIR}/val" ]; then
    echo "ERROR: DATADIR must contain train/ and val/: ${DATADIR}" >&2
    exit 1
fi

mkdir -p "${TRAINDIR}"

exec bash train_jax.sh "$@"
