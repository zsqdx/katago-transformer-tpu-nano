#!/bin/bash
set -euo pipefail

# Current best single-chip TPU v6e profile from local sweeps:
# b24c2048, batch 24, full BF16 storage/activations, BF16 RoPE/SwiGLU/attention
# logits, and train-buffer donation.

cd "$(dirname "$0")"

if [ "${OPTIMIZER:-adamw}" = "muon" ]; then
    export MUON_ROW_SPLIT_SIZE="${MUON_ROW_SPLIT_SIZE:-64}"
    export MUON_GROUP_BLOCKS="${MUON_GROUP_BLOCKS:-0}"
fi

MODEL_KIND="${MODEL_KIND:-b24c2048}" \
BATCH_SIZE="${BATCH_SIZE:-24}" \
ACTIVATION_DTYPE="${ACTIVATION_DTYPE:-bf16}" \
PARAM_DTYPE="${PARAM_DTYPE:-bf16}" \
OPT_STATE_DTYPE="${OPT_STATE_DTYPE:-bf16}" \
ROPE_DTYPE="${ROPE_DTYPE:-bf16}" \
FFN_MUL_DTYPE="${FFN_MUL_DTYPE:-bf16}" \
ATTENTION_LOGITS_DTYPE="${ATTENTION_LOGITS_DTYPE:-bf16}" \
DONATE_TRAIN_BUFFERS="${DONATE_TRAIN_BUFFERS:-1}" \
TRAINDIR="${TRAINDIR:-./jax_tpu_run_b24c2048_b24_fastbf16}" \
bash train_jax.sh "$@"
