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

## Colab TPU Monitor Cell

Run this in a separate notebook cell while `train.sh` is running. It displays
the latest training log lines plus `tpu-info` live TPU status. It requests HBM
usage, duty cycle, TensorCore utilization, and power metrics when the current
Colab/libtpu runtime exposes them.

```python
import pathlib
import sys
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
            "tpu-info failed because Colab is using an old pkg_resources "
            "package that is incompatible with Python 3.12.\n"
            "This cell tried to repair it with:\n"
            "  python -m pip install -U 'setuptools>=70' wheel\n"
            "If the error remains, run that command once in a separate cell "
            "and restart the runtime."
        )
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:]) if lines else "(no output)"

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
            probe = run(["tpu-info", "--version"])
        except Exception as exc:
            TPU_INFO_READY = False
            TPU_INFO_ERROR = f"tpu-info repair failed: {exc}"
            return False
        if probe.returncode == 0:
            TPU_INFO_READY = True
            TPU_INFO_ERROR = ""
            return True
        output = repair.stdout + "\n" + probe.stdout

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

If `tpu-info` still reports `pkgutil.ImpImporter` after the cell's automatic
repair attempt, run this once and restart the Colab runtime:

```bash
python -m pip install -U "setuptools>=70" wheel
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

After the smoke test passes, run the real loader with conservative settings:

```bash
export PJRT_DEVICE=TPU
bash train.sh
```

`train.sh` expects either:

- `DATADIR` pointing to a directory with `train/` and `val/` subdirectories, or
- a local `./val` directory. If only `./val` exists, the script uses it for both
  training and validation as a pipeline smoke test.

The script defaults to `model-kind=b12c192`, `batch-size=16`,
`max-training-samples=1024`, constant LR, and validation capped to 4 batches.
Constant LR avoids recompiling the XLA optimizer graph every step during short
smoke tests. Validation metrics are accumulated on-device and copied to the
host once per print window or validation pass. The default `PRINT_EVERY=10`
also avoids forcing TPU metric transfers every step. Override settings with
environment variables:

```bash
BATCH_SIZE=32 MAX_TRAINING_SAMPLES=4096 bash train.sh
```

For a full validation pass on all validation rows:

```bash
MAX_VAL_BATCHES=0 bash train.sh
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
  --lr-schedule constant \
  --max-training-samples 1024 \
  --symmetry-type xyt \
  --print-every 10 \
  --save-every-samples 1024 \
  --val-every-samples 1024 \
  --max-val-batches 4 \
  --warmup-samples 256 \
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
