# TPU v6e-1 Colab Smoke Test

This is the first TPU path for `nano/train.py`. It is intentionally narrow:

- single-device PyTorch/XLA only (`--device xla`)
- intended for Colab `v6e-1` smoke testing
- AdamW only
- no TransformerEngine, FP8, CUDA profiler, `torch.compile`, DDP, or ZeRO

## Colab Setup

Select a TPU runtime in Colab, preferably `v6e-1`, then run:

```bash
bash colab_install_torch_xla.sh
```

This pins `torch` and `torch_xla` to the same release. That matters because an
ABI mismatch shows up as an `_XLAC` undefined-symbol import error.

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

## Real Data Test

After the smoke test passes, run the real loader with conservative settings:

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
