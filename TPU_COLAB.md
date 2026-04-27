# TPU v6e-1 Colab Smoke Test

This is the first TPU path for `nano/train.py`. It is intentionally narrow:

- single-device PyTorch/XLA only (`--device xla`)
- intended for Colab `v6e-1` smoke testing
- AdamW only
- no TransformerEngine, FP8, CUDA profiler, `torch.compile`, DDP, or ZeRO

Validated on Colab TPU v6e-1 with Python 3.12, `torch==2.9.0+cpu`,
`torch_xla==2.9.0`, and `libtpu==0.0.21`.

## Colab Setup

Select a TPU runtime in Colab, preferably `v6e-1`, then run:

```bash
bash colab_install_torch_xla.sh
```

This pins `torch` and `torch_xla` to the same release. That matters because an
ABI mismatch shows up as an `_XLAC` undefined-symbol import error.
The installer uses the official PyTorch CPU wheel index for `torch`, avoiding
the much larger CUDA wheel that PyPI may otherwise select on Colab.

Restart the runtime if Colab asks you to after installation, or if a notebook
cell has already imported `torch` before the reinstall.

Clone or upload this repository, then:

```bash
cd KataGo_Transformer-TPU/nano
export PJRT_DEVICE=TPU
bash train_tpu_colab_smoke.sh
```

The script generates tiny synthetic data in `./tpu_smoke_data` and trains a
small 1-layer model into `./tpu_smoke_run`.

The first few optimizer steps may take several seconds while XLA traces and
compiles the graph. Later steps should get much faster once the compiled graph
is cached.

## Fixing `_XLAC` Import Errors

If `import torch_xla` fails with an error like:

```text
ImportError: ... _XLAC...so: undefined symbol ...
```

then Colab has loaded incompatible `torch` and `torch_xla` wheels. Reinstall
the matching pair and rerun the smoke test:

```bash
cd /content/katago-transformer-tpu-nano
bash colab_install_torch_xla.sh
bash train_tpu_colab_smoke.sh
```

The installer defaults to `2.9.0`, which supports Python 3.12 wheels. To try a
different matched release:

```bash
TORCH_XLA_VERSION=2.9.0 bash colab_install_torch_xla.sh
```

If a download times out, rerun the same command. The script keeps pip's cache
enabled and uses longer retry/timeout settings. For a particularly flaky Colab
session, you can increase them:

```bash
PIP_RETRIES=20 PIP_TIMEOUT=2000 bash colab_install_torch_xla.sh
```

## Real Data Test

After the smoke test passes, run the real loader with conservative settings:

```bash
export PJRT_DEVICE=TPU
bash train.sh
```

`train.sh` expects either:

- `DATADIR` pointing to a directory with `train/` and `val/` subdirectories, or
- a local `./val` directory. If only `./val` exists, the script uses it for both
  training and validation as a pipeline smoke test.

The script defaults to `model-kind=b12c192`, `batch-size=16`, and
`max-training-samples=1024`. Override them with environment variables:

```bash
BATCH_SIZE=32 MAX_TRAINING_SAMPLES=4096 bash train.sh
```

For reference, the command expanded by `train.sh` is equivalent to:

```bash
export PJRT_DEVICE=TPU
python -u train.py \
  --device xla \
  --traindir ./tpu_real_run \
  --datadir /path/to/shuffleddata \
  --pos-len 19 \
  --batch-size 16 \
  --model-kind b12c192 \
  --lr 2e-4 \
  --max-training-samples 1024 \
  --symmetry-type xyt \
  --print-every 1 \
  --save-every-samples 1024 \
  --val-every-samples 1024 \
  --warmup-samples 256 \
  --prefetch-batches 0 \
  --no-compile \
  --no-tensorboard \
  --amp-dtype bf16
```

Increase `--batch-size` only after the first run completes. If BF16 autocast
hits an XLA compatibility issue, retry once with `--amp-dtype none`.

## Notes

- `--device auto` also selects XLA when `PJRT_DEVICE=TPU` is already set.
- Threaded prefetch is disabled on XLA for now because this data generator moves
  tensors to the training device directly.
- Multi-chip TPU support should use `torch_xla.launch`, `MpDeviceLoader`, and
  `xm.optimizer_step`; that is deliberately left for the next migration step.
