# KataGo Transformer Nano TPU

Standalone `nano/` training code for transformer-based KataGo models, with an
initial PyTorch/XLA path for testing on Google Colab TPU v6e-1.

This repository is split from the larger KataGo Transformer project so the nano
training loop can be iterated on independently.

## Current Status

- CUDA training remains the primary, more complete path.
- PyTorch/XLA TPU support remains an initial single-device smoke-test path via
  `--device xla`.
- The JAX TPU path is the current TPU performance path. It supports the
  single-chip Colab `v6e-1` profile and single-process `pmap` data parallel
  training on local multi-chip TPU VMs such as `v6e-8`.
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
- `train_jax_zhizi_v6e8_b40c768.sh`: data-parallel v6e-8 Zhizi experiment runner.
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
`b12c1536`, `b12c1792`, `b8c2048`, `b10c2048`, `b12c2048`, `b14c2048`,
`b16c1536`, `b16c2048`, `b18c2048`, `b20c2048`, `b22c2048`, `b24c2048`,
`b24c1024`, and `b40c768` for checking whether a wider trunk improves TPU
utilization over the default `b12c192`.

## JAX TPU Prototype

The JAX path is intended for performance A/B testing after the PyTorch/XLA
baseline shows low TPU utilization. It currently supports the fixed-board
real-data path, AdamW/Muon, warmup+cosine LR, history matrices, X/Y/T
symmetries, BF16 matmul/conv compute, `score_mode=simple`, validation,
automatic resume, and pickle checkpoints. It does not yet implement
model sharding, multi-host sharding, variable-size board masks, or the `mixop`
score-belief head.

In a Colab TPU runtime:

```bash
cd KataGo_Transformer-TPU/nano
bash colab_install_jax_tpu.sh

NO_RESUME=1 bash train_jax_best_tpu.sh
```

If the stable windows after the first compile land far above the PyTorch/XLA
run, the next migration step is making the JAX path the default TPU runner and
then filling in the remaining feature gaps.

For the Zhizi `v6e-8` spot/GCS experiment, use the dedicated data-parallel
entrypoint:

```bash
bash train_jax_zhizi_v6e8_b40c768.sh
```

It defaults to `DATADIR=/mnt/data/datasets/shuffle-2405-2604-zhizi/main`,
`TRAINDIR=/mnt/ckpt/runs/zhizi-v6e8-exp001`, `MODEL_KIND=b40c768`, global
`BATCH_SIZE=128` (per-device batch 16 on 8 local TPU devices), full BF16, and
attention-only Muon. Checkpoints are written directly under `TRAINDIR`; leave
`NO_RESUME` unset for spot-preemption recovery, or set `NO_RESUME=1` for a
fresh run.

For `b24c1024`, the public profile keeps `STEPS_PER_JIT=1`; short sweeps found
`4` and `8` did not improve stable MFU. Use larger chunks only as a dispatch
overhead A/B because they change how often Python synchronizes with the TPU.
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
Set `COMPONENT_PROFILE=1` to run one separately-jitted component microprofile
before training starts. It prints `COMPONENT_TIME` rows for `stem_fwd`,
`block0_qkv_proj`, `block0_attention_core`, `block0_ffn_upgate`,
`trunk_all_blocks_fwd`, `heads_fwd`, `full_forward`, and `loss_forward`.
Use `COMPONENT_PROFILE_REPEATS=5` to change the stable timing repeats, and
`COMPONENT_PROFILE_GRAD=1` to include the much heavier `loss_grad`,
optimizer-update, and full train-step probes. These component timings are
directional because XLA can fuse and schedule operations differently inside
the real training step.

Mixed-precision A/B switches are also available for TPU profiling:
`ACTIVATION_DTYPE=bf16` keeps trunk activations in BF16 between transformer
ops, `PARAM_DTYPE=bf16` stores trainable parameters in BF16, and
`OPT_STATE_DTYPE=bf16` stores AdamW moment buffers in BF16.
`OPT_UPDATE_DTYPE=bf16` also runs the AdamW update math in BF16, which is a
more aggressive optimizer-throughput A/B. All default to `float32` until their
training impact is measured.
More granular BF16 probes are available for the current TPU bottlenecks:
`ROPE_DTYPE=bf16` computes RoPE rotations in BF16, `FFN_MUL_DTYPE=bf16` keeps
the SwiGLU gate product in BF16, and `ATTENTION_LOGITS_DTYPE=bf16` runs the
manual-attention softmax logits in BF16. These are profiling switches because
they can affect numerical behavior.
Experimental AQT INT8 training is available as an opt-in probe. Install the
optional AQT package first:

```bash
INSTALL_AQT=1 bash colab_install_jax_tpu.sh
```

Then quantize selected JAX `dot_general` operations during training:

```bash
INT8_TRAIN=1 INT8_TARGET=ffn NO_RESUME=1 bash train_jax_best_tpu.sh
```

`INT8_TARGET` can be `none`, `ffn`, `attn`, `attn_proj`, `attn_core`, `heads`,
`stem`, `trunk`, or `all`. `INT8_FWD_BITS` and `INT8_BWD_BITS` default to `8`.
Parameters and optimizer state remain floating point; AQT
quantizes the selected matmul inputs in forward/backward and updates latent
floating-point weights. Use this as a throughput/quality A/B and compare
samples/s, not only MFU, because the default MFU denominator is still the BF16
TPU peak unless `XLA_PEAK_TFLOPS` is overridden.

The recommended public single-chip TPU v6e profile is packaged as
`NO_RESUME=1 bash train_jax_best_tpu.sh`: `b24c1024`, `BATCH_SIZE=16`,
attention-only Muon, full BF16 storage/activations, BF16 RoPE/SwiGLU/attention
logits, and train-buffer donation. In short-window sweeps this profile reached
about 36.9% MFU and 453 samples/s. Longer runs should still validate training
quality.
The wrapper defaults to `OPTIMIZER=muon`, `MUON_TARGET=attn`,
`MUON_POLAR_STEPS=3`, `MUON_ROW_SPLIT_SIZE=64`, and `MUON_GROUP_BLOCKS=0`.
Stem/head/norm parameters continue to use AdamW, while targeted transformer
block matrix weights use Muon. The defaults match the PyTorch Muon
hyperparameters (`MUON_LR_MULTIPLIER=0.2`, `MUON_MOMENTUM=0.95`). Use
`NO_RESUME=1` when switching an existing JAX run between AdamW and Muon because
the optimizer state layout differs.
JAX checkpoints can be exported to ONNX through the same exporter used by the
PyTorch path:

```bash
python export_onnx.py \
  --checkpoint ./jax_tpu_run_b24c1024_b16_muon_fastbf16/checkpoint_jax.pkl \
  --output ./jax_tpu_run_b24c1024_b16_muon_fastbf16/model.onnx
```

The exporter auto-detects `.pkl` as a JAX checkpoint, converts the JAX parameter
tree to the `model.py` state dict, and then runs the legacy ONNX export path.
Install `onnx` first if your export environment does not already have it; add
`onnxruntime` only if you want `--verify`. Use `--checkpoint-format jax` if the
filename is nonstandard. JAX ONNX export currently supports the JAX training
path's `score_mode=simple` fixed-board model; AQT INT8 is a training-time dot
quantization probe and exported weights remain floating point.
For a wider AdamW throughput baseline, run
`OPTIMIZER=adamw MODEL_KIND=b24c2048 BATCH_SIZE=24 NO_RESUME=1 bash train_jax_best_tpu.sh`;
that short-sweep profile reached about 43.5% MFU.
Muon's optimizer work is not included in the standard model-FLOPs MFU log, and
full-matrix Muon on wide FFN weights can dominate wall time. Set
`MUON_ROW_SPLIT_SIZE=256` to split large block matrices into row chunks before
the polar iteration; try `128`, `256`, and `512` as TPU throughput/quality A/Bs.
For faster Muon probes, set `MUON_TARGET=attn`, `ffn`, `square`, or `none`
instead of the default `all`; non-targeted leaves fall back to AdamW. You can
also set `MUON_POLAR_STEPS=3` or `4` to trade optimizer precision for speed.
Use `bash sweep_jax_muon_fast.sh` for a very short throughput sweep over these
Muon knobs; override `SWEEP_SPECS` with entries like `attn:3:64`, and set
`SWEEP_STACK_BLOCKS=1` to test the stacked block layout.
For list-layout models, set `MUON_GROUP_BLOCKS=0` to update each block layer
directly instead of stacking same-name block weights inside the optimizer. This
is the default in the TPU best-profile wrapper when `OPTIMIZER=muon`.
Use `bash sweep_jax_batches.sh` to reproduce the default `b24c1024`
attention-Muon batch search; override `BATCH_SIZES` to refine around a winner.
Set `OPTIMIZER=adamw MODEL_KIND=b24c2048` to sweep the wider AdamW baseline.
Muon normally compiles as one train-step JIT so XLA can keep gradients inside
the compiled program. If that compile stalls on a larger shape, set
`MUON_SPLIT_JIT=1` as a slower fallback that compiles loss/grad and optimizer
update separately.
Set `OPTIMIZER=none` or `OPTIMIZER=sgd` only for profiling optimizer overhead:
`none` skips parameter updates and may let XLA eliminate unused backward work,
so treat it as a forward/loss lower-bound probe; `sgd` keeps gradients live
with a minimal weight-decayed SGD update.
Set `LOSS_PROFILE=policy_value`, `policy_only`, or `value_only` only for
profiling loss/head overhead. These modes deliberately drop auxiliary losses
and let XLA eliminate unused heads, so they are not training-equivalent.
Set `DONATE_TRAIN_BUFFERS=1` to let JAX donate parameter buffers across each
compiled train call; for AdamW this also donates the optimizer-state buffers.
This may reduce memory pressure, but it is kept opt-in because broader donation
can expose buffer-aliasing issues.
Set `STACK_BLOCKS=1` to store all transformer block parameters as stacked layer
arrays while still unrolling the block loop; this isolates the optimizer/tree
layout effect from the slower `lax.scan` execution path.
Set `SCAN_BLOCKS=1` to store transformer block parameters as stacked layer
arrays and run the trunk with `jax.lax.scan`; this can reduce compile latency,
but early `b24c1024` TPU profiling showed lower stable throughput than the
unscanned loop. Set `REMAT_BLOCKS=1` to checkpoint/rematerialize each
transformer block, which trades extra recomputation for lower activation memory
pressure and is intended as a separate throughput A/B.

To quickly search for a better single-chip TPU shape, run:

```bash
bash sweep_jax_shapes.sh
```

The sweep uses short full-BF16 JAX training windows and writes sorted results
to `jax_shape_sweep_*/summary.tsv`. Override the candidate list with
`SWEEP_SPECS="b8c2048:32 b12c2048:16"`; each entry is `MODEL_KIND:BATCH_SIZE`.
Set `SWEEP_COMPONENT_PROFILE=1` to also run the component microprofile for
each candidate. The sweep disables final checkpoint saves to keep the search
fast and disk-light.
Set `SWEEP_FAST_BF16=1` to sweep with the current fast TPU profile
(`ROPE_DTYPE=bf16`, `FFN_MUL_DTYPE=bf16`, `ATTENTION_LOGITS_DTYPE=bf16`, and
`DONATE_TRAIN_BUFFERS=1`). To sweep batch size around the public Muon default,
run `bash sweep_jax_batches.sh`. For the wider AdamW baseline, run:

```bash
MODEL_KIND=b24c2048 OPTIMIZER=adamw BATCH_SIZES="16 20 22 24 26 28 32" bash sweep_jax_batches.sh
```

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
