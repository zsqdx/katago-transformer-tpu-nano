#!/bin/bash
set -euo pipefail

# Colab TPU runtimes can come with a torch build that does not match the
# installed torch_xla wheel. Pin both packages to the same release to avoid
# _XLAC undefined-symbol import errors.

TORCH_XLA_VERSION="${TORCH_XLA_VERSION:-2.9.0}"

python - <<'PY'
import sys
print("python", sys.version.replace("\n", " "))
PY

python -m pip uninstall -y torch_xla torch torchvision torchaudio libtpu || true

python -m pip install --no-cache-dir --force-reinstall \
    "torch==${TORCH_XLA_VERSION}" \
    "torch_xla[tpu]==${TORCH_XLA_VERSION}" \
    numpy \
    tensorboard

python - <<'PY'
import importlib.metadata as md
import torch
import torch_xla

print("torch", torch.__version__)
print("torch_xla", torch_xla.__version__)
try:
    print("libtpu", md.version("libtpu"))
except md.PackageNotFoundError:
    print("libtpu package metadata not found")
print("xla device", torch.tensor(1.0, device="xla").device)
PY

echo "PyTorch/XLA install check passed. If this was run inside a notebook cell, restart the runtime once if later imports still see the old torch."
