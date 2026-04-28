# TPU v6e-1 Colab Smoke Test

This documents the first TPU paths for `nano/`: the existing PyTorch/XLA
trainer and a separate JAX/XLA prototype. The PyTorch/XLA path is intentionally
narrow:

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

## Colab TPU Monitor Cell

Run this in a separate notebook cell while `train.sh` is running. It displays
the latest training log lines plus `tpu-info` live TPU status. It requests HBM
usage, duty cycle, TensorCore utilization, and power metrics when the current
Colab/libtpu runtime exposes them.

```python
import pathlib
import shutil
import site
import sys
import sysconfig
import subprocess
import time
from IPython.display import clear_output

LOG = pathlib.Path("/content/katago-transformer-tpu-nano/tpu_real_run/train0.log")
TPU_INFO_READY = None
TPU_INFO_ERROR = ""
TPU_INFO_REPAIRED = False

def run(cmd, timeout=10):
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )

def short_error(text, max_lines=12):
    if "pkgutil.ImpImporter" in text or "pkg_resources" in text:
        return (
            "tpu-info failed because Colab has a stale Python 3.12 package "
            "layout: old pkg_resources and/or a regular google/__init__.py "
            "that breaks google.protobuf namespace imports.\n"
            "This cell tried to repair it with:\n"
            "  python -m pip install -U 'setuptools>=70' wheel\n"
            "  rm -f /usr/local/lib/python3.12/dist-packages/google/__init__.py\n"
            "  rm -rf /usr/local/lib/python3.12/dist-packages/google/__pycache__\n"
            "If the error remains, run those commands once in a separate cell "
            "and restart the runtime."
        )
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:]) if lines else "(no output)"

def remove_legacy_google_init():
    roots = set(site.getsitepackages())
    purelib = sysconfig.get_path("purelib")
    if purelib:
        roots.add(purelib)

    removed = []
    for root in sorted(roots):
        google_dir = pathlib.Path(root) / "google"
        init_py = google_dir / "__init__.py"
        pycache = google_dir / "__pycache__"
        if init_py.exists():
            init_py.unlink()
            removed.append(str(init_py))
        if pycache.exists():
            shutil.rmtree(pycache)
            removed.append(str(pycache))
    return removed

def ensure_tpu_info():
    global TPU_INFO_READY, TPU_INFO_ERROR, TPU_INFO_REPAIRED
    if TPU_INFO_READY is True:
        return True

    try:
        probe = run(["tpu-info", "--version"])
    except Exception as exc:
        TPU_INFO_READY = False
        TPU_INFO_ERROR = f"tpu-info failed: {exc}"
        return False
    if probe.returncode == 0:
        TPU_INFO_READY = True
        TPU_INFO_ERROR = ""
        return True

    output = probe.stdout
    needs_setuptools = "pkgutil.ImpImporter" in output or "pkg_resources" in output
    if needs_setuptools and not TPU_INFO_REPAIRED:
        TPU_INFO_REPAIRED = True
        try:
            repair = run(
                [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "setuptools>=70", "wheel"],
                timeout=120,
            )
            removed = remove_legacy_google_init()
            probe = run(["tpu-info", "--version"])
        except Exception as exc:
            TPU_INFO_READY = False
            TPU_INFO_ERROR = f"tpu-info repair failed: {exc}"
            return False
        if probe.returncode == 0:
            TPU_INFO_READY = True
            TPU_INFO_ERROR = ""
            return True
        output = repair.stdout + "\nRemoved: " + ", ".join(removed) + "\n" + probe.stdout

    TPU_INFO_READY = False
    TPU_INFO_ERROR = short_error(output)
    return False

while True:
    clear_output(wait=True)
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("\n=== train0.log tail ===")
    if LOG.exists():
        print("\n".join(LOG.read_text(errors="replace").splitlines()[-20:]))
    else:
        print(f"{LOG} not found yet")

    if ensure_tpu_info():
        try:
            metrics_proc = run(["tpu-info", "--list_metrics"])
            list_metrics = metrics_proc.stdout if metrics_proc.returncode == 0 else ""
        except Exception as exc:
            list_metrics = ""

        requested = [
            name for name in (
                "duty_cycle_percent",
                "hbm_usage",
                "tensorcore_utilization",
                "power",
                "power_usage",
                "chip_power",
                "total_power",
            )
            if name in list_metrics
        ]

        commands = []
        if requested:
            commands.append(["tpu-info", "--metric", *requested])
        commands.extend((["tpu-info"], ["tpu-info", "--process"]))

        for cmd in commands:
            print("\n=== " + " ".join(cmd) + " ===")
            try:
                out = run(cmd)
                print(out.stdout if out.returncode == 0 else short_error(out.stdout))
            except Exception as exc:
                print(f"{' '.join(cmd)} failed: {exc}")
    else:
        print("\n=== tpu-info ===")
        print(TPU_INFO_ERROR)

    time.sleep(5)
```

Training logs also report TPU MFU. For XLA runs, the normal `MFU` is wall-clock
MFU including compile/data overhead, while `xla_mfu` is based on XLA
`ExecuteTime` and is closer to device execution MFU.

For TPU runs, random board symmetries and history-matrix preprocessing are
applied on the host before tensors are moved to XLA. This avoids turning data
augmentation variants into separate XLA training graphs. If `xla_d2h` remains
high in the logs, look for explicit host reads such as `.item()`, `.tolist()`,
or `.cpu()` in the active training path.

If `tpu-info` still reports `pkgutil.ImpImporter` after the cell's automatic
repair attempt, run this once and restart the Colab runtime:

```bash
python -m pip install -U "setuptools>=70" wheel
rm -f /usr/local/lib/python3.12/dist-packages/google/__init__.py
rm -rf /usr/local/lib/python3.12/dist-packages/google/__pycache__
tpu-info
```

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

After the smoke test passes, run the real loader with a moderate TPU v6e-1
training configuration:

```bash
export PJRT_DEVICE=TPU
bash train.sh
```

`train.sh` expects either:

- `DATADIR` pointing to a directory with `train/` and `val/` subdirectories, or
- a local `./val` directory. If only `./val` exists, the script uses it for both
  training and validation as a pipeline smoke test.

The script defaults to `model-kind=b12c192`, `batch-size=256`,
`max-training-samples=262144`, `warmup-samples=32768`, warmup+cosine LR, and
validation capped to 16 batches. By default, save and validation run at the end
of the default training window, because Colab TPU checkpoint/validation
interruptions are much slower than the steady training loop. AdamW uses
capturable step tensors on XLA to avoid per-step optimizer host reads, and CPU
batches are sent to XLA with a batched transfer path when the installed
`torch_xla` wheel supports it. Validation metrics are accumulated on-device and
copied to the host once per print window or validation pass. The default
`PRINT_EVERY=20` also avoids forcing TPU metric transfers every step. Override
settings with environment variables:

```bash
BATCH_SIZE=128 MAX_TRAINING_SAMPLES=131072 WARMUP_SAMPLES=16384 bash train.sh
```

If Colab runs out of HBM, retry with `BATCH_SIZE=128`.

For frequent checkpoints/validation while debugging:

```bash
SAVE_EVERY_SAMPLES=65536 VAL_EVERY_SAMPLES=65536 bash train.sh
```

The `GRAD_ACCUM_STEPS=2` probe keeps the per-microbatch size at 256, so it does
not improve TPU matrix utilization. If HBM allows it, test a true larger
microbatch next:

```bash
BATCH_SIZE=512 \
MAX_TRAINING_SAMPLES=524288 \
WARMUP_SAMPLES=65536 \
SAVE_EVERY_SAMPLES=524288 \
VAL_EVERY_SAMPLES=524288 \
TRAINDIR=./tpu_real_run_b512 \
bash train.sh
```

The batch-512 `b12c192` run is still too narrow to improve utilization much.
The next useful sweep is widening the trunk. Current v6e-1 probes show
`b12c1536` with `BATCH_SIZE=32` as the best practical point so far: about
26 TFLOPS / 2.9% wall-clock MFU in stable windows. Start conservatively, then
raise `BATCH_SIZE` if HBM allows it:

```bash
MODEL_KIND=b12c512 \
BATCH_SIZE=128 \
MAX_TRAINING_SAMPLES=131072 \
WARMUP_SAMPLES=16384 \
TRAINDIR=./tpu_real_run_b12c512_b128 \
bash train.sh

MODEL_KIND=b12c768 \
BATCH_SIZE=64 \
MAX_TRAINING_SAMPLES=131072 \
WARMUP_SAMPLES=16384 \
TRAINDIR=./tpu_real_run_b12c768_b64 \
bash train.sh

MODEL_KIND=b12c1024 \
BATCH_SIZE=32 \
MAX_TRAINING_SAMPLES=65536 \
WARMUP_SAMPLES=8192 \
TRAINDIR=./tpu_real_run_b12c1024_b32 \
bash train.sh

MODEL_KIND=b12c1536 \
BATCH_SIZE=32 \
MAX_TRAINING_SAMPLES=65536 \
WARMUP_SAMPLES=8192 \
TRAINDIR=./tpu_real_run_b12c1536_b32 \
bash train.sh

MODEL_KIND=b12c2048 \
BATCH_SIZE=8 \
MAX_TRAINING_SAMPLES=16384 \
WARMUP_SAMPLES=2048 \
TRAINDIR=./tpu_real_run_b12c2048_b8 \
bash train.sh
```

If `b12c2048_b8` fits comfortably, try:

```bash
MODEL_KIND=b12c2048 \
BATCH_SIZE=16 \
MAX_TRAINING_SAMPLES=32768 \
WARMUP_SAMPLES=4096 \
TRAINDIR=./tpu_real_run_b12c2048_b16 \
bash train.sh
```

For each width, compare only stable windows after `xla_compile=0.0s/0`. If the
run fits comfortably, try doubling the batch once. If it runs out of HBM, halve
the batch. If `b12c2048` does not beat the `b12c1536_b32` stable-window TFLOPS,
stay with `b12c1536_b32` for v6e-1.

To reduce optimizer/clip overhead per sample without increasing activation
memory, try gradient accumulation:

```bash
GRAD_ACCUM_STEPS=2 TRAINDIR=./tpu_real_run_b256_acc2 bash train.sh
```

For a short performance-only check, you can also temporarily disable gradient
clipping:

```bash
GRAD_CLIP_NORM=0 TRAINDIR=./tpu_real_run_b256_noclip bash train.sh
```

If capturable AdamW causes an optimizer compatibility error with a specific
Colab wheel, rerun with:

```bash
TRAINDIR=./tpu_real_run_b256_nocap bash train.sh --disable-xla-capturable-adamw
```

If the batched XLA transfer path falls back or regresses on a specific wheel,
rerun with:

```bash
TRAINDIR=./tpu_real_run_b256_notransferbatch bash train.sh --disable-xla-batched-transfer
```

This run intentionally switches back to warmup+cosine. Watch `xla_compile`
after the first few print windows; if it becomes nonzero every window again,
isolate LR scheduling with:

```bash
LR_SCHEDULE=constant bash train.sh
```

For a full validation pass on all validation rows:

```bash
MAX_VAL_BATCHES=0 bash train.sh
```

## JAX TPU Prototype

The PyTorch/XLA sweeps above top out at low single-digit wall-clock MFU even
after widening the trunk. To separate model math from PyTorch/XLA overhead, the
repo also includes a direct JAX/XLA training prototype. It keeps the PyTorch
path intact, uses the same `.npz` data format and model presets, and starts
with the performance-critical training step only.

Install JAX in a TPU runtime:

```bash
cd /content/katago-transformer-tpu-nano
bash colab_install_jax_tpu.sh
```

Then run the first comparison at the same `b12c2048_b16` scale:

```bash
MODEL_KIND=b12c2048 \
BATCH_SIZE=16 \
MAX_TRAINING_SAMPLES=32768 \
WARMUP_SAMPLES=4096 \
PRINT_EVERY=20 \
TRAINDIR=./jax_tpu_run_b12c2048_b16 \
bash train_jax.sh
```

`train_jax.sh` uses `./val` as both `train/` and `val/` when only `./val`
exists, matching `train.sh` for smoke runs. The prototype currently supports
fixed-board training, AdamW, warmup+cosine LR, host-side history matrices and
symmetry augmentation, BF16 matmul/conv compute, `score_mode=simple`,
validation, automatic resume, and pickle checkpoints. It intentionally does
not yet include multi-device sharding, variable-board masks, or
`score_mode=mixop`.

Judge the run by stable post-compile windows. If JAX removes the repeated
compile and host-transfer stalls, the next useful step is making it the main
TPU path and filling in the remaining feature gaps.

For deeper/narrower shapes like `b24c1024`, the per-step dispatch overhead is a
larger fraction of runtime than it is for `b12c2048`. Use `STEPS_PER_JIT=4`
first, then try `8` if HBM is comfortable:

```bash
MODEL_KIND=b24c1024 \
BATCH_SIZE=16 \
STEPS_PER_JIT=4 \
MAX_TRAINING_SAMPLES=32768 \
WARMUP_SAMPLES=4096 \
TRAINDIR=./jax_tpu_run_b24c1024_b16_s4 \
bash train_jax.sh
```

Projection layers are separate by default because early TPU A/B runs showed
the fused QKV/SwiGLU layout was slower for `b24c1024`. Set
`FUSE_PROJECTIONS=1` only when you want to compare against that fused layout.
The next useful code-level A/B is JAX/XLA's built-in dot-product attention:

```bash
MODEL_KIND=b24c1024 \
BATCH_SIZE=16 \
ATTENTION_IMPL=xla \
NO_RESUME=1 \
TRAINDIR=./jax_tpu_run_b24c1024_b16_attn_xla \
bash train_jax.sh
```

For reference, the command expanded by `train.sh` is equivalent to:

```bash
export PJRT_DEVICE=TPU
python -u train.py \
  --device xla \
  --traindir ./tpu_real_run \
  --datadir /path/to/shuffleddata \
  --pos-len 19 \
  --batch-size 256 \
  --grad-accum-steps 1 \
  --model-kind b12c192 \
  --lr 2e-4 \
  --grad-clip-norm 1.0 \
  --lr-schedule cosine \
  --max-training-samples 262144 \
  --symmetry-type xyt \
  --print-every 20 \
  --save-every-samples 262144 \
  --val-every-samples 262144 \
  --max-val-batches 16 \
  --warmup-samples 32768 \
  --prefetch-batches 0 \
  --no-compile \
  --no-tensorboard \
  --allow-nonfull-mask \
  --xla-peak-tflops 918 \
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
