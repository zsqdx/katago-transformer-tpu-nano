#!/bin/bash
set -euo pipefail

# JAX TPU v6e-1 prototype. Mirrors train.sh's data layout handling, but runs
# train_jax.py without importing PyTorch.

cd "$(dirname "$0")"

export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"

if [ -n "${DATADIR:-}" ]; then
    DATA_ROOT="${DATADIR}"
elif [ -d "./train" ] && [ -d "./val" ]; then
    DATA_ROOT="."
elif [ -d "./val" ]; then
    DATA_ROOT="./.jax_real_data"
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
if [ "${NO_RESUME:-0}" != "0" ]; then
    EXTRA_FLAGS+=(--no-resume)
fi
if [ "${NO_FINAL_SAVE:-0}" != "0" ]; then
    EXTRA_FLAGS+=(--no-final-save)
fi
if [ "${SEPARATE_PROJECTIONS:-0}" != "0" ]; then
    EXTRA_FLAGS+=(--separate-projections)
fi
if [ "${FUSE_PROJECTIONS:-0}" != "0" ]; then
    EXTRA_FLAGS+=(--fuse-projections)
fi
if [ "${LOG_GRAD_NORM:-0}" != "0" ]; then
    EXTRA_FLAGS+=(--log-grad-norm)
fi
if [ "${LOG_STEP_TIME:-0}" != "0" ]; then
    EXTRA_FLAGS+=(--log-step-time)
fi
if [ "${COMPONENT_PROFILE:-0}" != "0" ]; then
    EXTRA_FLAGS+=(--component-profile)
fi
if [ "${COMPONENT_PROFILE_GRAD:-0}" != "0" ]; then
    EXTRA_FLAGS+=(--component-profile-grad)
fi
if [ "${DONATE_TRAIN_BUFFERS:-0}" != "0" ]; then
    EXTRA_FLAGS+=(--donate-train-buffers)
fi
if [ "${MUON_SPLIT_JIT:-0}" != "0" ]; then
    EXTRA_FLAGS+=(--muon-split-jit)
fi
if [ "${STACK_BLOCKS:-0}" != "0" ]; then
    EXTRA_FLAGS+=(--stack-blocks)
fi
if [ "${SCAN_BLOCKS:-0}" != "0" ]; then
    EXTRA_FLAGS+=(--scan-blocks)
fi
if [ "${REMAT_BLOCKS:-0}" != "0" ]; then
    EXTRA_FLAGS+=(--remat-blocks)
fi

MAX_TRAINING_SAMPLES_VALUE="${MAX_TRAINING_SAMPLES:-32768}"
SAVE_EVERY_SAMPLES_VALUE="${SAVE_EVERY_SAMPLES:-${MAX_TRAINING_SAMPLES_VALUE}}"
VAL_EVERY_SAMPLES_VALUE="${VAL_EVERY_SAMPLES:-${MAX_TRAINING_SAMPLES_VALUE}}"
WARMUP_SAMPLES_VALUE="${WARMUP_SAMPLES:-4096}"

python -u train_jax.py \
    --traindir "${TRAINDIR:-./jax_tpu_run}" \
    --datadir "${DATA_ROOT}" \
    --pos-len "${POS_LEN:-19}" \
    --batch-size "${BATCH_SIZE:-16}" \
    --model-kind "${MODEL_KIND:-b12c2048}" \
    --lr "${LR:-2e-4}" \
    --wd "${WD:-0.1}" \
    --optimizer "${OPTIMIZER:-adamw}" \
    --muon-lr-multiplier "${MUON_LR_MULTIPLIER:-0.2}" \
    --muon-momentum "${MUON_MOMENTUM:-0.95}" \
    --muon-row-split-size "${MUON_ROW_SPLIT_SIZE:-0}" \
    --muon-target "${MUON_TARGET:-all}" \
    --muon-polar-steps "${MUON_POLAR_STEPS:-5}" \
    --loss-profile "${LOSS_PROFILE:-full}" \
    --grad-clip-norm "${GRAD_CLIP_NORM:-0}" \
    --lr-schedule "${LR_SCHEDULE:-cosine}" \
    --max-training-samples "${MAX_TRAINING_SAMPLES_VALUE}" \
    --warmup-samples "${WARMUP_SAMPLES_VALUE}" \
    --print-every "${PRINT_EVERY:-20}" \
    --steps-per-jit "${STEPS_PER_JIT:-1}" \
    --component-profile-repeats "${COMPONENT_PROFILE_REPEATS:-3}" \
    --save-every-samples "${SAVE_EVERY_SAMPLES_VALUE}" \
    --val-every-samples "${VAL_EVERY_SAMPLES_VALUE}" \
    --max-val-batches "${MAX_VAL_BATCHES:-16}" \
    --symmetry-type "${SYMMETRY_TYPE:-xyt}" \
    --score-mode "${SCORE_MODE:-simple}" \
    --attention-impl "${ATTENTION_IMPL:-manual}" \
    --activation-dtype "${ACTIVATION_DTYPE:-float32}" \
    --param-dtype "${PARAM_DTYPE:-float32}" \
    --opt-state-dtype "${OPT_STATE_DTYPE:-float32}" \
    --opt-update-dtype "${OPT_UPDATE_DTYPE:-float32}" \
    --rope-dtype "${ROPE_DTYPE:-float32}" \
    --ffn-mul-dtype "${FFN_MUL_DTYPE:-float32}" \
    --attention-logits-dtype "${ATTENTION_LOGITS_DTYPE:-float32}" \
    --xla-peak-tflops "${XLA_PEAK_TFLOPS:-918}" \
    --allow-nonfull-mask \
    --seed "${SEED:-1234}" \
    "${EXTRA_FLAGS[@]}" \
    "$@"
