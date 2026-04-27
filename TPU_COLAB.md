# TPU v6e-1 Colab Smoke Test

This is the first TPU path for `nano/train.py`. It is intentionally narrow:

- single-device PyTorch/XLA only (`--device xla`)
- intended for Colab `v6e-1` smoke testing
- AdamW only
- no TransformerEngine, FP8, CUDA profiler, `torch.compile`, DDP, or ZeRO

## Colab Setup

Select a TPU runtime in Colab, preferably `v6e-1`, then run:

```bash
pip install -U torch torch_xla[tpu]
```

Restart the runtime if Colab asks you to after installation.

Clone or upload this repository, then:

```bash
cd KataGo_Transformer-TPU/nano
export PJRT_DEVICE=TPU
bash train_tpu_colab_smoke.sh
```

The script generates tiny synthetic data in `./tpu_smoke_data` and trains a
small 1-layer model into `./tpu_smoke_run`.

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
