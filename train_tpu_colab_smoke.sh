#!/bin/bash
set -euo pipefail

# Run from the nano/ directory on a Colab v6e-1 TPU runtime.
# If torch_xla fails to import with an _XLAC undefined symbol error, run:
#   bash colab_install_torch_xla.sh

export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"

python - <<'PY'
import importlib.metadata as md
import sys

print("python", sys.version.replace("\n", " "))
for package in ("torch", "torch_xla", "libtpu"):
    try:
        print(package, md.version(package))
    except md.PackageNotFoundError:
        print(package, "not installed")

try:
    import torch
    import torch_xla
except Exception as exc:
    print("\nFailed to import torch_xla. This usually means torch and torch_xla are ABI-mismatched.")
    print("Repair command:")
    print("  bash colab_install_torch_xla.sh")
    raise SystemExit(1) from exc

print("torch import", torch.__version__)
print("torch_xla import", torch_xla.__version__)
print("xla device", torch.tensor(1.0, device="xla").device)
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
    --no-tensorboard \
    --amp-dtype bf16 \
    --seed 1234
