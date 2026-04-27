# KataGo Transformer Nano TPU

Standalone `nano/` training code for transformer-based KataGo models, with an
initial PyTorch/XLA path for testing on Google Colab TPU v6e-1.

This repository is split from the larger KataGo Transformer project so the nano
training loop can be iterated on independently.

## Current Status

- CUDA training remains the primary, more complete path.
- TPU support is an initial single-device smoke-test path via `--device xla`.
- The TPU path is intended for Colab `v6e-1` first, using AdamW only.
- TransformerEngine, FP8, CUDA profiling, ZeRO, multi-GPU DDP, and
  `torch.compile` are not enabled on XLA yet.
- Colab TPU v6e-1 smoke test has passed with Python 3.12,
  `torch==2.9.0+cpu`, and `torch_xla==2.9.0`.

## Files

- `train.py`: main training loop.
- `model.py`: pure PyTorch transformer model.
- `model_te.py`: NVIDIA TransformerEngine variant for CUDA.
- `data.py`: `.npz` training data loading and symmetry augmentation.
- `losses.py`: postprocessing, loss, metrics, and FLOPs estimates.
- `optimizers.py`, `zero.py`: Adam/Muon/Shampoo and ZeRO helpers.
- `make_smoke_data.py`: tiny synthetic data generator for smoke tests.
- `train_tpu_colab_smoke.sh`: Colab TPU v6e-1 smoke-test runner.
- `TPU_COLAB.md`: Colab setup and TPU test notes.

## Quick CPU Smoke Test

```bash
python make_smoke_data.py --out-dir ./smoke_data --pos-len 9 --samples-per-file 4

python -u train.py \
  --device cpu \
  --traindir ./smoke_run \
  --datadir ./smoke_data \
  --pos-len 9 \
  --batch-size 2 \
  --num-layers 1 \
  --hidden-size 32 \
  --num-heads 2 \
  --max-training-samples 4 \
  --save-every-samples 4 \
  --val-every-samples 4 \
  --warmup-samples 2 \
  --symmetry-type none \
  --print-every 1 \
  --prefetch-batches 0 \
  --no-compile \
  --amp-dtype none
```

## Colab TPU v6e-1 Smoke Test

In a Colab TPU runtime:

```bash
export PJRT_DEVICE=TPU
bash colab_install_torch_xla.sh
bash train_tpu_colab_smoke.sh
```

See `TPU_COLAB.md` for the real-data follow-up command.

With real data in a local `val/` directory, run:

```bash
export PJRT_DEVICE=TPU
bash train.sh
```

`train.sh` uses `val/` as both `train/` and `val/` for a pipeline smoke test if
no separate `train/` directory is present. It now defaults to a moderate TPU
run: batch size 256, 262,144 training samples, 32,768 warmup samples, cosine
LR, and validation capped to 16 batches. By default, periodic save/validation
run at the end of the default training window to avoid interrupting TPU
throughput tests. Validation metrics are accumulated on the device and copied
back once per print window or validation pass. On TPU, AdamW uses capturable
step tensors, CPU batches are transferred to XLA in one batched call when
available, and random symmetries/history preprocessing are applied before
transfer to XLA to avoid repeated training-graph compilation. For throughput
experiments, `train.sh` also exposes `GRAD_ACCUM_STEPS` and `GRAD_CLIP_NORM`
environment overrides.

Available TPU width presets include `b12c512`, `b12c768`, and `b12c1024` for
checking whether a wider trunk improves TPU utilization over the default
`b12c192`.

## CUDA Training Example

```bash
python -u train.py \
  --traindir ../data/train/nano_test \
  --datadir ../data/shuffleddata/kata1_trainingdata_25q4_2601 \
  --pos-len 19 \
  --batch-size 1024 \
  --model-kind b12c768 \
  --lr 2e-4 \
  --max-training-samples 300000000 \
  --symmetry-type xyt \
  --print-every 1 \
  --save-every-samples 1000000 \
  --val-every-samples 1000000 \
  --warmup-samples 2000000 \
  --enable-history-matrices
```

## License

MIT License. See `LICENSE`.
