#!/bin/bash
set -euo pipefail

# Quick single-TPU JAX shape sweep. Each spec runs train_jax.sh for a short
# window, captures the log, and prints a sorted MFU/TFLOPS summary.

cd "$(dirname "$0")"

SWEEP_ROOT="${SWEEP_ROOT:-./jax_shape_sweep_$(date +%Y%m%d_%H%M%S)}"
SWEEP_SPECS="${SWEEP_SPECS:-b24c1024:16 b12c1536:16 b12c1536:32 b16c1536:16 b12c1792:16 b8c2048:16 b8c2048:32 b12c2048:16 b16c2048:8 b16c2048:16}"
SWEEP_SKIP_WINDOWS="${SWEEP_SKIP_WINDOWS:-1}"
SWEEP_TEE="${SWEEP_TEE:-1}"
SWEEP_COMPONENT_PROFILE="${SWEEP_COMPONENT_PROFILE:-0}"
SWEEP_COMPONENT_PROFILE_GRAD="${SWEEP_COMPONENT_PROFILE_GRAD:-0}"

MAX_TRAINING_SAMPLES_VALUE="${MAX_TRAINING_SAMPLES:-4096}"
WARMUP_SAMPLES_VALUE="${WARMUP_SAMPLES:-1024}"
PRINT_EVERY_VALUE="${PRINT_EVERY:-20}"
SAVE_EVERY_SAMPLES_VALUE="${SAVE_EVERY_SAMPLES:-1000000000}"
VAL_EVERY_SAMPLES_VALUE="${VAL_EVERY_SAMPLES:-1000000000}"
MAX_VAL_BATCHES_VALUE="${MAX_VAL_BATCHES:-0}"

mkdir -p "${SWEEP_ROOT}/logs"

echo "JAX shape sweep"
echo "  root: ${SWEEP_ROOT}"
echo "  specs: ${SWEEP_SPECS}"
echo "  max_training_samples: ${MAX_TRAINING_SAMPLES_VALUE}"
echo "  warmup_samples: ${WARMUP_SAMPLES_VALUE}"
echo "  print_every: ${PRINT_EVERY_VALUE}"
echo

run_index=0
for spec in ${SWEEP_SPECS}; do
    run_index=$((run_index + 1))
    model="${spec%%:*}"
    batch="${spec##*:}"
    if [ -z "${model}" ] || [ -z "${batch}" ] || [ "${model}" = "${batch}" ]; then
        echo "Skipping invalid spec '${spec}', expected MODEL:BATCH" >&2
        continue
    fi

    run_name="$(printf "%02d_%s_b%s" "${run_index}" "${model}" "${batch}")"
    traindir="${SWEEP_ROOT}/${run_name}"
    log_path="${SWEEP_ROOT}/logs/${run_name}.log"

    echo "=== ${run_name} ==="
    set +e
    if [ "${SWEEP_TEE}" != "0" ]; then
        (
            MODEL_KIND="${model}" \
            BATCH_SIZE="${batch}" \
            MAX_TRAINING_SAMPLES="${MAX_TRAINING_SAMPLES_VALUE}" \
            WARMUP_SAMPLES="${WARMUP_SAMPLES_VALUE}" \
            PRINT_EVERY="${PRINT_EVERY_VALUE}" \
            SAVE_EVERY_SAMPLES="${SAVE_EVERY_SAMPLES_VALUE}" \
            VAL_EVERY_SAMPLES="${VAL_EVERY_SAMPLES_VALUE}" \
            MAX_VAL_BATCHES="${MAX_VAL_BATCHES_VALUE}" \
            ACTIVATION_DTYPE="${ACTIVATION_DTYPE:-bf16}" \
            PARAM_DTYPE="${PARAM_DTYPE:-bf16}" \
            OPT_STATE_DTYPE="${OPT_STATE_DTYPE:-bf16}" \
            OPT_UPDATE_DTYPE="${OPT_UPDATE_DTYPE:-float32}" \
            COMPONENT_PROFILE="${SWEEP_COMPONENT_PROFILE}" \
            COMPONENT_PROFILE_GRAD="${SWEEP_COMPONENT_PROFILE_GRAD}" \
            COMPONENT_PROFILE_REPEATS="${COMPONENT_PROFILE_REPEATS:-3}" \
            NO_RESUME=1 \
            NO_FINAL_SAVE=1 \
            TRAINDIR="${traindir}" \
            bash train_jax.sh
        ) 2>&1 | tee "${log_path}"
        status=${PIPESTATUS[0]}
    else
        MODEL_KIND="${model}" \
        BATCH_SIZE="${batch}" \
        MAX_TRAINING_SAMPLES="${MAX_TRAINING_SAMPLES_VALUE}" \
        WARMUP_SAMPLES="${WARMUP_SAMPLES_VALUE}" \
        PRINT_EVERY="${PRINT_EVERY_VALUE}" \
        SAVE_EVERY_SAMPLES="${SAVE_EVERY_SAMPLES_VALUE}" \
        VAL_EVERY_SAMPLES="${VAL_EVERY_SAMPLES_VALUE}" \
        MAX_VAL_BATCHES="${MAX_VAL_BATCHES_VALUE}" \
        ACTIVATION_DTYPE="${ACTIVATION_DTYPE:-bf16}" \
        PARAM_DTYPE="${PARAM_DTYPE:-bf16}" \
        OPT_STATE_DTYPE="${OPT_STATE_DTYPE:-bf16}" \
        OPT_UPDATE_DTYPE="${OPT_UPDATE_DTYPE:-float32}" \
        COMPONENT_PROFILE="${SWEEP_COMPONENT_PROFILE}" \
        COMPONENT_PROFILE_GRAD="${SWEEP_COMPONENT_PROFILE_GRAD}" \
        COMPONENT_PROFILE_REPEATS="${COMPONENT_PROFILE_REPEATS:-3}" \
        NO_RESUME=1 \
        NO_FINAL_SAVE=1 \
        TRAINDIR="${traindir}" \
        bash train_jax.sh >"${log_path}" 2>&1
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
component_re = re.compile(
    r"COMPONENT_TIME name=(?P<name>\S+) .*?min=(?P<min>[0-9.]+)s .*?"
    r"(?:est_TFLOPS_min=(?P<tflops>[0-9.]+) est_MFU_min=(?P<mfu>[0-9.]+)%)?"
)


def parse_args_line(text):
    for line in text.splitlines():
        marker = "Args: "
        if marker in line:
            payload = line.split(marker, 1)[1]
            try:
                return ast.literal_eval(payload)
            except Exception:
                return {}
    return {}


rows = []
for log in logs:
    text = log.read_text(errors="replace")
    args = parse_args_line(text)
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
    components = {}
    for m in component_re.finditer(text):
        components[m.group("name")] = {
            "min": float(m.group("min")),
            "tflops": float(m.group("tflops")) if m.group("tflops") else None,
            "mfu": float(m.group("mfu")) if m.group("mfu") else None,
        }

    status = "ok" if "Training complete" in text else "failed"
    if stable:
        mfus = [x["mfu"] for x in stable]
        tflops = [x["tflops"] for x in stable]
        sps = [x["sps"] for x in stable]
        best_idx = max(range(len(stable)), key=lambda i: stable[i]["mfu"])
        row = {
            "status": status,
            "name": log.stem,
            "model": args.get("model_kind", "?"),
            "batch": args.get("batch_size", "?"),
            "best_mfu": stable[best_idx]["mfu"],
            "median_mfu": statistics.median(mfus),
            "last_mfu": stable[-1]["mfu"],
            "best_tflops": stable[best_idx]["tflops"],
            "median_tflops": statistics.median(tflops),
            "last_tflops": stable[-1]["tflops"],
            "median_sps": statistics.median(sps),
            "windows": len(train_rows),
            "stable_windows": len(stable),
            "trunk_fwd_mfu": (components.get("trunk_all_blocks_fwd") or {}).get("mfu"),
            "full_fwd_mfu": (components.get("full_forward") or {}).get("mfu"),
            "log": str(log),
        }
    else:
        row = {
            "status": status,
            "name": log.stem,
            "model": args.get("model_kind", "?"),
            "batch": args.get("batch_size", "?"),
            "best_mfu": 0.0,
            "median_mfu": 0.0,
            "last_mfu": 0.0,
            "best_tflops": 0.0,
            "median_tflops": 0.0,
            "last_tflops": 0.0,
            "median_sps": 0.0,
            "windows": 0,
            "stable_windows": 0,
            "trunk_fwd_mfu": None,
            "full_fwd_mfu": None,
            "log": str(log),
        }
    rows.append(row)

rows.sort(key=lambda x: (x["best_mfu"], x["median_mfu"]), reverse=True)
fields = [
    "status", "model", "batch", "best_mfu", "median_mfu", "last_mfu",
    "best_tflops", "median_tflops", "median_sps", "windows",
    "trunk_fwd_mfu", "full_fwd_mfu", "log",
]
summary_path = root / "summary.tsv"
with summary_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

print("=== Sweep summary (sorted by best stable MFU) ===")
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
