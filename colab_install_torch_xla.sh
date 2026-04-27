#!/bin/bash
set -euo pipefail

# Colab TPU runtimes can come with a torch build that does not match the
# installed torch_xla wheel. Pin both packages to the same release to avoid
# _XLAC undefined-symbol import errors.

TORCH_XLA_VERSION="${TORCH_XLA_VERSION:-2.9.0}"
PIP_RETRIES="${PIP_RETRIES:-10}"
PIP_TIMEOUT="${PIP_TIMEOUT:-1000}"

python - <<'PY'
import sys
print("python", sys.version.replace("\n", " "))
PY

python -m pip uninstall -y torch_xla torch torchvision torchaudio libtpu || true

python -m pip install --upgrade pip

# Colab Python 3.12 may keep an old apt-provided pkg_resources on sys.path.
# tpu-info/protobuf imports can fail with pkgutil.ImpImporter unless setuptools
# is upgraded in the active Python environment.
python -m pip install --upgrade "setuptools>=70" wheel

# Use the official PyTorch CPU wheel index. TPU/XLA does not need the large
# CUDA-enabled PyPI torch wheel, and Colab downloads are prone to timeouts.
python -m pip install --retries "${PIP_RETRIES}" --timeout "${PIP_TIMEOUT}" --force-reinstall \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==${TORCH_XLA_VERSION}"

# Install XLA and TPU runtime deps from PyPI. The CPU torch wheel above
# satisfies torch_xla's torch dependency without pulling CUDA packages.
python -m pip install --retries "${PIP_RETRIES}" --timeout "${PIP_TIMEOUT}" --force-reinstall \
    "torch_xla[tpu]==${TORCH_XLA_VERSION}" \
    numpy \
    tensorboard

# Some Colab Python 3.12 images leave a regular google package at
# site-packages/google/__init__.py. That breaks google.protobuf namespace
# imports used by tpu-info and can still trigger pkgutil.ImpImporter errors even
# after setuptools is upgraded. Remove only this namespace-conflict file/cache.
python - <<'PY'
import pathlib
import shutil
import site
import sysconfig

roots = set(site.getsitepackages())
purelib = sysconfig.get_path("purelib")
if purelib:
    roots.add(purelib)

for root in sorted(roots):
    google_dir = pathlib.Path(root) / "google"
    init_py = google_dir / "__init__.py"
    pycache = google_dir / "__pycache__"
    if init_py.exists():
        print(f"Removing legacy google namespace stub: {init_py}")
        init_py.unlink()
    if pycache.exists():
        print(f"Removing google namespace cache: {pycache}")
        shutil.rmtree(pycache)
PY

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

tpu-info --version || true

echo "PyTorch/XLA install check passed. If this was run inside a notebook cell, restart the runtime once if later imports still see the old torch."
