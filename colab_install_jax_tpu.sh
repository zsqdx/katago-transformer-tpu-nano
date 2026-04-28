#!/bin/bash
set -euo pipefail

python --version
python -m pip install --upgrade pip

if [ "${JAX_NIGHTLY:-0}" = "1" ]; then
    python -m pip install -U --pre jax jaxlib libtpu requests \
        -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ \
        -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
else
    python -m pip install -U "jax[tpu]" \
        -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
fi

python - <<'PY'
import jax
print("jax", jax.__version__)
print("devices", jax.devices())
PY
