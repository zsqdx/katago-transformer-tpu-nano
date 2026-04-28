#!/bin/bash
set -euo pipefail

# Fast Muon knob sweep for the current best single-chip TPU shape. Each spec is:
#   MUON_TARGET:MUON_POLAR_STEPS:MUON_ROW_SPLIT_SIZE
#
# This intentionally runs only a tiny training window. It is for throughput
# triage, not loss-quality decisions.

cd "$(dirname "$0")"

SWEEP_ROOT="${SWEEP_ROOT:-./jax_muon_sweep_$(date +%Y%m%d_%H%M%S)}"
SWEEP_SPECS="${SWEEP_SPECS:-attn:5:128 attn:3:64 all:3:64 all:5:32 ffn:3:32 none:5:0}"
SWEEP_SKIP_WINDOWS="${SWEEP_SKIP_WINDOWS:-1}"
SWEEP_TEE="${SWEEP_TEE:-1}"

MODEL_KIND_VALUE="${MODEL_KIND:-b24c2048}"
BATCH_SIZE_VALUE="${BATCH_SIZE:-24}"
MAX_TRAINING_SAMPLES_VALUE="${MAX_TRAINING_SAMPLES:-960}"
WARMUP_SAMPLES_VALUE="${WARMUP_SAMPLES:-480}"
PRINT_EVERY_VALUE="${PRINT_EVERY:-20}"

mkdir -p "${SWEEP_ROOT}/logs"

echo "JAX Muon fast sweep"
echo "  root: ${SWEEP_ROOT}"
echo "  model: ${MODEL_KIND_VALUE}"
echo "  batch: ${BATCH_SIZE_VALUE}"
echo "  specs: ${SWEEP_SPECS}"
echo "  max_training_samples: ${MAX_TRAINING_SAMPLES_VALUE}"
echo "  warmup_samples: ${WARMUP_SAMPLES_VALUE}"
echo "  print_every: ${PRINT_EVERY_VALUE}"
echo

run_index=0
for spec in ${SWEEP_SPECS}; do
    run_index=$((run_index + 1))
    IFS=: read -r muon_target muon_polar_steps muon_row_split_size extra <<< "${spec}"
    if [ -n "${extra:-}" ] || [ -z "${muon_target:-}" ] || [ -z "${muon_polar_steps:-}" ] || [ -z "${muon_row_split_size:-}" ]; then
        echo "Skipping invalid spec '${spec}', expected TARGET:POLAR_STEPS:ROW_SPLIT" >&2
        continue
    fi

    run_name="$(printf "%02d_%s_p%s_r%s" "${run_index}" "${muon_target}" "${muon_polar_steps}" "${muon_row_split_size}")"
    traindir="${SWEEP_ROOT}/${run_name}"
    log_path="${SWEEP_ROOT}/logs/${run_name}.log"

    echo "=== ${run_name} ==="
    set +e
    if [ "${SWEEP_TEE}" != "0" ]; then
        (
            OPTIMIZER=muon \
            MUON_TARGET="${muon_target}" \
            MUON_POLAR_STEPS="${muon_polar_steps}" \
            MUON_ROW_SPLIT_SIZE="${muon_row_split_size}" \
            MODEL_KIND="${MODEL_KIND_VALUE}" \
            BATCH_SIZE="${BATCH_SIZE_VALUE}" \
            MAX_TRAINING_SAMPLES="${MAX_TRAINING_SAMPLES_VALUE}" \
            WARMUP_SAMPLES="${WARMUP_SAMPLES_VALUE}" \
            PRINT_EVERY="${PRINT_EVERY_VALUE}" \
            SAVE_EVERY_SAMPLES=1000000000 \
            VAL_EVERY_SAMPLES=1000000000 \
            MAX_VAL_BATCHES=0 \
            NO_RESUME=1 \
            NO_FINAL_SAVE=1 \
            TRAINDIR="${traindir}" \
            bash train_jax_best_tpu.sh
        ) 2>&1 | tee "${log_path}"
        status=${PIPESTATUS[0]}
    else
        OPTIMIZER=muon \
        MUON_TARGET="${muon_target}" \
        MUON_POLAR_STEPS="${muon_polar_steps}" \
        MUON_ROW_SPLIT_SIZE="${muon_row_split_size}" \
        MODEL_KIND="${MODEL_KIND_VALUE}" \
        BATCH_SIZE="${BATCH_SIZE_VALUE}" \
        MAX_TRAINING_SAMPLES="${MAX_TRAINING_SAMPLES_VALUE}" \
        WARMUP_SAMPLES="${WARMUP_SAMPLES_VALUE}" \
        PRINT_EVERY="${PRINT_EVERY_VALUE}" \
        SAVE_EVERY_SAMPLES=1000000000 \
        VAL_EVERY_SAMPLES=1000000000 \
        MAX_VAL_BATCHES=0 \
        NO_RESUME=1 \
        NO_FINAL_SAVE=1 \
        TRAINDIR="${traindir}" \
        bash train_jax_best_tpu.sh >"${log_path}" 2>&1
        status=$?
    fi
    set -e

    if [ "${status}" -ne 0 ]; then
        echo "Run ${run_name} failed with status ${status}; continuing." >&2
        echo "${status}" > "${traindir}.failed"
    fi
    echo
done

python - "${SWEEP_ROOT}" "${SWEEP_SKIP_WINDOWS}" <<'PY'
import ast
import csv
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
skip_windows = int(sys.argv[2])
logs = sorted((root / "logs").glob("*.log"))

train_re = re.compile(
    r"step=(?P<step>\d+), samples=(?P<samples>\d+), time=(?P<time>[0-9.]+)s, .*?"
    r"sps=(?P<sps>[0-9.]+), TFLOPS=(?P<tflops>[0-9.]+), MFU=(?P<mfu>[0-9.]+)%"
)
muon_flops_re = re.compile(r"Muon update approx FLOPs/step=(?P<flops>[0-9.]+)T")


def parse_args_line(text):
    for line in text.splitlines():
        marker = "Args: "
        if marker in line:
            try:
                return ast.literal_eval(line.split(marker, 1)[1])
            except Exception:
                return {}
    return {}


def parse_name_fallback(stem):
    parts = stem.split("_")
    if len(parts) >= 4 and parts[0].isdigit():
        target = parts[1]
        polar_steps = parts[2][1:] if parts[2].startswith("p") else "?"
        row_split = parts[3][1:] if parts[3].startswith("r") else "?"
        return target, polar_steps, row_split
    return "?", "?", "?"


rows = []
for log in logs:
    text = log.read_text(errors="replace")
    args = parse_args_line(text)
    fallback_target, fallback_polar_steps, fallback_row_split = parse_name_fallback(log.stem)
    train_rows = [
        {
            "step": int(m.group("step")),
            "samples": int(m.group("samples")),
            "time": float(m.group("time")),
            "sps": float(m.group("sps")),
            "tflops": float(m.group("tflops")),
            "mfu": float(m.group("mfu")),
        }
        for m in train_re.finditer(text)
    ]
    stable = train_rows[skip_windows:] if len(train_rows) > skip_windows else train_rows
    muon_flops_match = muon_flops_re.search(text)

    status = "ok" if "Training complete" in text else "failed"
    row = {
        "status": status,
        "target": args.get("muon_target", fallback_target),
        "polar_steps": args.get("muon_polar_steps", fallback_polar_steps),
        "row_split": args.get("muon_row_split_size", fallback_row_split),
        "model": args.get("model_kind", "?"),
        "batch": args.get("batch_size", "?"),
        "muon_update_t_per_step": float(muon_flops_match.group("flops")) if muon_flops_match else None,
        "windows": len(train_rows),
        "stable_windows": len(stable),
        "log": str(log),
    }
    if stable:
        mfus = [x["mfu"] for x in stable]
        tflops = [x["tflops"] for x in stable]
        sps = [x["sps"] for x in stable]
        best = max(stable, key=lambda x: x["mfu"])
        row.update({
            "best_mfu": best["mfu"],
            "median_mfu": statistics.median(mfus),
            "last_mfu": stable[-1]["mfu"],
            "best_tflops": best["tflops"],
            "median_tflops": statistics.median(tflops),
            "median_sps": statistics.median(sps),
        })
    else:
        row.update({
            "best_mfu": 0.0,
            "median_mfu": 0.0,
            "last_mfu": 0.0,
            "best_tflops": 0.0,
            "median_tflops": 0.0,
            "median_sps": 0.0,
        })
    rows.append(row)

rows.sort(key=lambda x: (x["best_mfu"], x["median_mfu"]), reverse=True)
fields = [
    "status", "target", "polar_steps", "row_split", "model", "batch",
    "best_mfu", "median_mfu", "last_mfu", "best_tflops", "median_tflops",
    "median_sps", "muon_update_t_per_step", "windows", "stable_windows", "log",
]
summary_path = root / "summary.tsv"
with summary_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

print("=== Muon fast sweep summary (sorted by best stable MFU) ===")
print("\t".join(fields))
for row in rows:
    def fmt(value):
        if isinstance(value, float):
            return f"{value:.2f}"
        if value is None:
            return ""
        return str(value)
    print("\t".join(fmt(row[field]) for field in fields))
print(f"\nWrote {summary_path}")
PY
