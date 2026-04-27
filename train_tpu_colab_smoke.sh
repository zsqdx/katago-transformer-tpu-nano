#!/bin/bash
set -euo pipefail

# Run from the nano/ directory on a Colab v6e-1 TPU runtime.
# If torch_xla is missing, install the matching PyTorch/XLA package first:
#   pip install -U torch torch_xla[tpu]

export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"

python - <<'PY'
import torch
import torch_xla
print("torch", torch.__version__)
print("torch_xla", torch_xla.__version__)
PY

python make_smoke_data.py \
    --out-dir ./tpu_smoke_data \
    --pos-len 9 \
    --samples-per-file 8 \
    --train-files 2 \
    --val-files 1 \
    --seed 1234

python -u train.py \
    --device xla \
    --traindir ./tpu_smoke_run \
    --datadir ./tpu_smoke_data \
    --pos-len 9 \
    --batch-size 2 \
    --num-layers 1 \
    --hidden-size 32 \
    --num-heads 2 \
    --lr 1e-4 \
    --max-training-samples 8 \
    --save-every-samples 8 \
    --val-every-samples 8 \
    --warmup-samples 2 \
    --symmetry-type none \
    --print-every 1 \
    --prefetch-batches 0 \
    --no-compile \
    --amp-dtype bf16 \
    --seed 1234
