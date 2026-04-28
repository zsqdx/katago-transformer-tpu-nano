# KataGo Transformer Nano TPU

Standalone `nano/` training code for transformer-based KataGo models, with an
initial PyTorch/XLA path for testing on Google Colab TPU v6e-1.

This repository is split from the larger KataGo Transformer project so the nano
training loop can be iterated on independently.

## Current Status

- CUDA training remains the primary, more complete path.
- TPU support is an initial single-device smoke-test path via `--device xla`.
- A separate JAX TPU prototype is available for direct PyTorch/XLA vs JAX/XLA
  throughput comparisons on the same data and model presets.
- The TPU path is intended for Colab `v6e-1` first, using AdamW only.
- TransformerEngine, FP8, CUDA profiling, ZeRO, multi-GPU DDP, and
  `torch.compile` are not enabled on XLA yet.
- Colab TPU v6e-1 smoke test has passed with Python 3.12,
  `torch==2.9.0+cpu`, and `torch_xla==2.9.0`.

## Files

- `train.py`: main training loop.
- `train_jax.py`: JAX TPU prototype training loop.
- `model.py`: pure PyTorch transformer model.
- `jax_model.py`, `jax_losses.py`, `jax_data.py`: JAX/NumPy model, loss, and
  data path used by `train_jax.py`.
- `model_te.py`: NVIDIA TransformerEngine variant for CUDA.
- `data.py`: `.npz` training data loading and symmetry augmentation.
- `losses.py`: postprocessing, loss, metrics, and FLOPs estimates.
- `optimizers.py`, `zero.py`: Adam/Muon/Shampoo and ZeRO helpers.
- `make_smoke_data.py`: tiny synthetic data generator for smoke tests.
- `train_tpu_colab_smoke.sh`: Colab TPU v6e-1 smoke-test runner.
- `train_jax.sh`: real-data JAX TPU prototype runner.
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

Available TPU width presets include `b12c512`, `b12c768`, `b12c1024`,
`b12c1536`, and `b12c2048` for checking whether a wider trunk improves TPU
utilization over the default `b12c192`.

## JAX TPU Prototype

The JAX path is intended for performance A/B testing after the PyTorch/XLA
baseline shows low TPU utilization. It currently supports the fixed-board
real-data path, AdamW, warmup+cosine LR, history matrices, X/Y/T symmetries,
BF16 matmul/conv compute, `score_mode=simple`, validation, automatic resume,
and pickle checkpoints. It does not yet implement multi-device sharding,
variable-size board masks, or the `mixop` score-belief head.

In a Colab TPU runtime:

```bash
cd KataGo_Transformer-TPU/nano
bash colab_install_jax_tpu.sh

MODEL_KIND=b12c2048 \
BATCH_SIZE=16 \
MAX_TRAINING_SAMPLES=32768 \
WARMUP_SAMPLES=4096 \
TRAINDIR=./jax_tpu_run_b12c2048_b16 \
bash train_jax.sh
```

If the stable windows after the first compile land far above the PyTorch/XLA
run, the next migration step is making the JAX path the default TPU runner and
then filling in the remaining feature gaps.

For deeper/narrower presets such as `b24c1024`, try `STEPS_PER_JIT=4` or `8`
to execute several optimizer steps per XLA call. This reduces per-step dispatch
overhead without changing the per-sample LR schedule or checkpoint semantics.
Projection layers are separate by default because early TPU A/B runs showed the
fused QKV/SwiGLU layout was slower for `b24c1024`; set `FUSE_PROJECTIONS=1`
only for that A/B. To test JAX/XLA's built-in attention lowering:

```bash
MODEL_KIND=b24c1024 \
BATCH_SIZE=16 \
ATTENTION_IMPL=xla \
NO_RESUME=1 \
TRAINDIR=./jax_tpu_run_b24c1024_b16_attn_xla \
bash train_jax.sh
```

Gradient norm logging is off by default when `GRAD_CLIP_NORM=0`, avoiding an
extra full-gradient reduction on large models. Set `LOG_GRAD_NORM=1` to restore
the old log field for debugging.
Set `LOG_STEP_TIME=1` to synchronously log every compiled train call. With the
default `STEPS_PER_JIT=1`, each `STEP_TIME` line is one optimizer step; with
larger chunks, `per_step_total` is the chunk average. This mode is for timing
debugging and intentionally adds synchronization.

Mixed-precision A/B switches are also available for TPU profiling:
`ACTIVATION_DTYPE=bf16` keeps trunk activations in BF16 between transformer
ops, `PARAM_DTYPE=bf16` stores trainable parameters in BF16, and
`OPT_STATE_DTYPE=bf16` stores AdamW moment buffers in BF16.
`OPT_UPDATE_DTYPE=bf16` also runs the AdamW update math in BF16, which is a
more aggressive optimizer-throughput A/B. All default to `float32` until their
training impact is measured.
Set `OPTIMIZER=none` or `OPTIMIZER=sgd` only for profiling optimizer overhead:
`none` skips parameter updates and may let XLA eliminate unused backward work,
so treat it as a forward/loss lower-bound probe; `sgd` keeps gradients live
with a minimal weight-decayed SGD update.
Set `LOSS_PROFILE=policy_value`, `policy_only`, or `value_only` only for
profiling loss/head overhead. These modes deliberately drop auxiliary losses
and let XLA eliminate unused heads, so they are not training-equivalent.
Set `DONATE_TRAIN_BUFFERS=1` to let JAX donate the parameter buffers across
each compiled train call; this may reduce memory pressure, but it is kept
opt-in because broader donation can expose buffer-aliasing issues.
Set `STACK_BLOCKS=1` to store all transformer block parameters as stacked layer
arrays while still unrolling the block loop; this isolates the optimizer/tree
layout effect from the slower `lax.scan` execution path.
Set `SCAN_BLOCKS=1` to store transformer block parameters as stacked layer
arrays and run the trunk with `jax.lax.scan`; this can reduce compile latency,
but early `b24c1024` TPU profiling showed lower stable throughput than the
unscanned loop. Set `REMAT_BLOCKS=1` to checkpoint/rematerialize each
transformer block, which trades extra recomputation for lower activation memory
pressure and is intended as a separate throughput A/B.

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
