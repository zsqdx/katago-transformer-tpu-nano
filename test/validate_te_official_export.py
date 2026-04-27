#!/usr/bin/env python3
"""Generate a random TransformerEngine checkpoint and export ONNX in multiple modes.

Usage:
    python test/validate_te_official_export.py
    python test/validate_te_official_export.py --config b24c1024 --verify-onnxruntime
"""

import argparse
import collections
import os
import random
import re
import shlex
import shutil
import subprocess
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import configs
from export_onnx import DEFAULT_ONNX_OPSET, convert_trt_fp8_to_standard_qdq, export, export_per_block, verify

DEFAULT_MODES = ["te-official", "te-decomposed", "legacy"]
ALL_MODES = DEFAULT_MODES + ["fp8-manual"]
_TE_MODES = {"te-official", "te-decomposed"}
TRT_PRECISIONS = ["fp32", "fp16", "bf16", "fp8"]
DEFAULT_TRTEXEC_CANDIDATES = [
    "trtexec",
    "/usr/src/tensorrt/bin/trtexec",
    "/usr/local/tensorrt/bin/trtexec",
    "/usr/local/TensorRT/bin/trtexec",
    "/opt/tensorrt/bin/trtexec",
]


def _mode_slug(mode):
    return mode.replace("-", "_")


def _resolve_mode_artifact_path(base_path, output_dir, config_name, mode, ext, single_mode):
    mode_suffix = _mode_slug(mode)
    if base_path is None:
        return os.path.join(output_dir, f"{config_name}_{mode_suffix}{ext}")

    root, current_ext = os.path.splitext(base_path)
    if not current_ext:
        root = base_path
        current_ext = ext
    if single_mode:
        return root + current_ext
    return f"{root}_{mode_suffix}{current_ext}"


def _append_artifact_suffix(path, suffix):
    root, ext = os.path.splitext(path)
    if not ext:
        return path + suffix
    return f"{root}{suffix}{ext}"


def _make_export_args(args, checkpoint_path, onnx_path, method, enable_nested_fallbacks):
    return argparse.Namespace(
        checkpoint=checkpoint_path,
        output=onnx_path,
        method=method,
        device=args.device,
        pos_len=args.pos_len,
        score_mode=args.score_mode,
        export_scope=args.export_scope,
        opset=args.opset,
        dynamic_batch=args.dynamic_batch,
        verify=False,
        ort_provider=args.ort_provider,
        fallback_to_te_decomposed_on_te_export_error=(
            enable_nested_fallbacks and args.fallback_to_te_decomposed_on_te_export_error
        ),
        fallback_to_legacy_on_te_export_error=(
            enable_nested_fallbacks and args.fallback_to_legacy_on_te_export_error
        ),
        use_fp8=args.use_fp8 if method != "fp8-manual" else False,
        fp8_recipe=args.fp8_recipe,
        use_te=(method in ("legacy", "fp8-manual")),
        use_ema=False,
    )


def _save_random_te_checkpoint(args, checkpoint_path):
    try:
        from model_te import Model as TEModel
    except ImportError as exc:
        print(
            "ERROR: failed to import model_te / transformer_engine.pytorch for TE validation.\n"
            f"Original import error: {exc}\n"
            "Hint: top-level `import transformer_engine` is not enough. "
            "This script needs `import transformer_engine.pytorch as te` to work."
        )
        raise SystemExit(1) from exc

    model_config = configs.config_of_name[args.config]
    print(f"Config: {args.config} -> {model_config}")
    print(f"Seed: {args.seed}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = TEModel(model_config, args.pos_len, score_mode=args.score_mode, use_fp8=args.use_fp8)
    model.initialize(init_std=args.init_std)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters (TE): {num_params:,}")

    torch.save({"model": model.state_dict(), "config": model_config}, checkpoint_path)
    print(f"Saved random TE checkpoint: {checkpoint_path}")

    return model_config


def _save_random_checkpoint(args, checkpoint_path):
    """Generate a random model.py checkpoint (no Transformer Engine dependency)."""
    from model import Model

    model_config = configs.config_of_name[args.config]
    print(f"Config: {args.config} -> {model_config}")
    print(f"Seed: {args.seed}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = Model(model_config, args.pos_len, score_mode=args.score_mode)
    model.initialize(init_std=args.init_std)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    torch.save({"model": model.state_dict(), "config": model_config}, checkpoint_path)
    print(f"Saved random checkpoint: {checkpoint_path}")

    return model_config


def _shape_spec(args, model_config):
    batch = args.batch_size
    if args.export_scope == "blocks":
        seq_len = args.pos_len * args.pos_len
        hidden_size = model_config["hidden_size"]
        return f"input_stem:{batch}x{seq_len}x{hidden_size}"

    num_bin = configs.get_num_bin_input_features(model_config)
    num_global = configs.get_num_global_input_features(model_config)
    return (
        f"input_spatial:{batch}x{num_bin}x{args.pos_len}x{args.pos_len},"
        f"input_global:{batch}x{num_global}"
    )


def _resolve_trtexec_path(trtexec_bin):
    if os.path.isabs(trtexec_bin):
        return trtexec_bin if os.path.isfile(trtexec_bin) and os.access(trtexec_bin, os.X_OK) else None

    direct = shutil.which(trtexec_bin)
    if direct is not None:
        return direct

    if trtexec_bin != "trtexec":
        return None

    for candidate in DEFAULT_TRTEXEC_CANDIDATES[1:]:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _mode_trt_precision(args, mode):
    return {
        "te-official": args.te_official_trt_precision,
        "te-decomposed": args.te_decomposed_trt_precision,
        "legacy": args.legacy_trt_precision,
        "fp8-manual": args.fp8_manual_trt_precision,
    }[mode]


def _trtexec_precision_flags(precision, has_trt_fp8_qdq=False, has_standard_qdq=False):
    if precision == "fp32":
        return []
    if precision == "fp8":
        if has_trt_fp8_qdq:
            # TRT custom FP8 ops (trt::TRT_FP8*) need --stronglyTyped to
            # force TRT to respect the explicit type annotations exactly.
            # --fp8 conflicts with --stronglyTyped (TRT ignores it with a
            # warning), so we use --stronglyTyped alone.
            return ["--stronglyTyped"]
        if has_standard_qdq:
            # Standard ONNX QuantizeLinear/DequantizeLinear with FP8 types.
            # Explicit quantization uses --stronglyTyped so TRT respects
            # the Q/DQ type annotations and fuses DQL->MatMul<-DQL into
            # FP8 GEMM kernels.
            return ["--stronglyTyped"]
        # No Q/DQ nodes at all; just request FP8 and hope for the best.
        return ["--fp8"]
    return [f"--{precision}"]


def _run_trtexec(cmd, description):
    print(description)
    print("  " + " ".join(shlex.quote(part) for part in cmd))
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.returncode != 0:
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        lines = [f"trtexec failed with exit code {completed.returncode}"]
        if stdout:
            lines.append("stdout:")
            lines.append(stdout)
        if stderr:
            lines.append("stderr:")
            lines.append(stderr)
        raise RuntimeError("\n".join(lines))
    return completed.stdout + ("\n" + completed.stderr if completed.stderr else "")


def _parse_trtexec_benchmark(output_text):
    metrics = {}

    throughput_match = re.search(r"Throughput:\s*([0-9.]+)\s*qps", output_text)
    if throughput_match:
        metrics["throughput_qps"] = float(throughput_match.group(1))

    latency_match = re.search(
        r"Latency:\s*min =\s*([0-9.]+)\s*ms,\s*max =\s*([0-9.]+)\s*ms,\s*"
        r"mean =\s*([0-9.]+)\s*ms,\s*median =\s*([0-9.]+)\s*ms,\s*"
        r"percentile\(99%\) =\s*([0-9.]+)\s*ms",
        output_text,
    )
    if latency_match:
        metrics["latency_min_ms"] = float(latency_match.group(1))
        metrics["latency_max_ms"] = float(latency_match.group(2))
        metrics["latency_mean_ms"] = float(latency_match.group(3))
        metrics["latency_median_ms"] = float(latency_match.group(4))
        metrics["latency_p99_ms"] = float(latency_match.group(5))

    gpu_match = re.search(
        r"GPU Compute Time:\s*min =\s*([0-9.]+)\s*ms,\s*max =\s*([0-9.]+)\s*ms,\s*"
        r"mean =\s*([0-9.]+)\s*ms,\s*median =\s*([0-9.]+)\s*ms,\s*"
        r"percentile\(99%\) =\s*([0-9.]+)\s*ms",
        output_text,
    )
    if gpu_match:
        metrics["gpu_mean_ms"] = float(gpu_match.group(3))
        metrics["gpu_p99_ms"] = float(gpu_match.group(5))

    return metrics


def _parse_trtexec_build_precision(output_text):
    match = re.search(r"Precision:\s*([A-Za-z0-9+, ]+)", output_text)
    if match:
        return match.group(1).strip()
    return None


def _format_trtexec_benchmark(metrics):
    if not metrics:
        return "benchmark completed (metrics not parsed)"

    parts = []
    if "throughput_qps" in metrics:
        parts.append(f"{metrics['throughput_qps']:.2f} qps")
    if "latency_mean_ms" in metrics:
        parts.append(f"latency_mean={metrics['latency_mean_ms']:.3f} ms")
    if "latency_p99_ms" in metrics:
        parts.append(f"latency_p99={metrics['latency_p99_ms']:.3f} ms")
    if "gpu_mean_ms" in metrics:
        parts.append(f"gpu_mean={metrics['gpu_mean_ms']:.3f} ms")
    return ", ".join(parts)


def _is_trtexec_static_shape_error(error_text):
    return "Static model does not take explicit shapes" in error_text


def _is_trtexec_myelin_ssa_error(error_text):
    return "MyelinCheckException" in error_text and "ssa_validation()" in error_text


def _inspect_onnx_quantization(onnx_path):
    try:
        import onnx
    except ImportError:
        return None

    model = onnx.load(onnx_path, load_external_data=False)
    op_counts = collections.Counter()
    for node in model.graph.node:
        key = f"{node.domain}::{node.op_type}" if node.domain else node.op_type
        op_counts[key] += 1

    standard_qdq = op_counts.get("QuantizeLinear", 0) + op_counts.get("DequantizeLinear", 0)
    trt_fp8_qdq = sum(
        count for key, count in op_counts.items()
        if key.startswith("trt::TRT_FP8")
    )
    custom_domain_ops = {
        key: count for key, count in op_counts.items()
        if "::" in key
    }
    return {
        "standard_qdq": standard_qdq,
        "trt_fp8_qdq": trt_fp8_qdq,
        "custom_domain_ops": custom_domain_ops,
    }


def _format_onnx_quantization_info(info):
    if info is None:
        return "onnx package not available; quantization nodes not inspected"
    parts = []
    if info["trt_fp8_qdq"] > 0:
        parts.append(f"{info['trt_fp8_qdq']} TRT FP8 Q/DQ nodes")
    if info["standard_qdq"] > 0:
        parts.append(f"{info['standard_qdq']} standard Q/DQ nodes")
    if info["custom_domain_ops"]:
        op_summary = ", ".join(f"{k}({v})" for k, v in sorted(info["custom_domain_ops"].items()))
        parts.append(f"custom ops: [{op_summary}]")
    if not parts:
        return "no Q/DQ nodes detected"
    return "; ".join(parts)


def _maybe_run_trtexec(args, onnx_path, engine_path, model_config, mode):
    if args.skip_trtexec:
        print(f"Skipping TensorRT build/benchmark for {mode} because --skip-trtexec was set")
        return None

    trtexec_path = _resolve_trtexec_path(args.trtexec_bin)
    if trtexec_path is None:
        print(
            f"Skipping TensorRT build/benchmark for {mode} because {args.trtexec_bin!r} was not found. "
            "Add it to PATH or pass --trtexec-bin /abs/path/to/trtexec"
        )
        return None

    precision = _mode_trt_precision(args, mode)
    quantization_info = _inspect_onnx_quantization(onnx_path)
    print(f"Requested TensorRT precision for {mode}: {precision}")
    print(f"ONNX quantization inspection for {mode}: {_format_onnx_quantization_info(quantization_info)}")
    if precision == "fp8" and (
        quantization_info is None
        or (quantization_info["trt_fp8_qdq"] == 0 and quantization_info["standard_qdq"] == 0)
    ):
        print(
            "WARNING: FP8 was requested, but no FP8 Q/DQ nodes were detected in the ONNX graph. "
            "TensorRT documents FP8 as an explicit-quantization workflow, so this build may fail or fall back to higher precision."
        )

    has_trt_fp8_qdq = quantization_info is not None and quantization_info["trt_fp8_qdq"] > 0
    has_standard_qdq = quantization_info is not None and quantization_info["standard_qdq"] > 0
    shapes = _shape_spec(args, model_config)
    build_cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes={shapes}",
        f"--optShapes={shapes}",
        f"--maxShapes={shapes}",
        "--skipInference",
    ]
    build_cmd.extend(_trtexec_precision_flags(precision, has_trt_fp8_qdq=has_trt_fp8_qdq, has_standard_qdq=has_standard_qdq))
    for plugin_path in getattr(args, "trt_plugins", []):
        build_cmd.append(f"--plugins={plugin_path}")

    build_uses_explicit_shapes = True
    try:
        build_output = _run_trtexec(build_cmd, f"Running TensorRT engine build for {mode}:")
    except RuntimeError as exc:
        if not _is_trtexec_static_shape_error(str(exc)):
            raise
        print(f"Retrying TensorRT engine build for {mode} without explicit shapes because the ONNX model is static.")
        build_uses_explicit_shapes = False
        static_build_cmd = [
            trtexec_path,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--skipInference",
        ]
        static_build_cmd.extend(_trtexec_precision_flags(precision, has_trt_fp8_qdq=has_trt_fp8_qdq, has_standard_qdq=has_standard_qdq))
        for plugin_path in getattr(args, "trt_plugins", []):
            static_build_cmd.append(f"--plugins={plugin_path}")
        build_output = _run_trtexec(static_build_cmd, f"Running TensorRT engine build for {mode} (static-shape retry):")
    actual_precision = _parse_trtexec_build_precision(build_output)
    if actual_precision is not None:
        print(f"TensorRT reported build precision for {mode}: {actual_precision}")
    print(f"TensorRT engine saved to: {engine_path}")

    if args.skip_trtexec_benchmark:
        print(f"Skipping TensorRT benchmark for {mode} because --skip-trtexec-benchmark was set")
        return None

    benchmark_cmd = [
        trtexec_path,
        f"--loadEngine={engine_path}",
        f"--warmUp={args.trtexec_warmup_ms}",
        f"--duration={args.trtexec_duration_s}",
        f"--avgRuns={args.trtexec_avg_runs}",
    ]
    if build_uses_explicit_shapes:
        benchmark_cmd.insert(2, f"--shapes={shapes}")

    try:
        benchmark_output = _run_trtexec(benchmark_cmd, f"Running TensorRT benchmark for {mode}:")
    except RuntimeError as exc:
        if not build_uses_explicit_shapes or not _is_trtexec_static_shape_error(str(exc)):
            raise
        print(f"Retrying TensorRT benchmark for {mode} without explicit shapes because the engine is static.")
        benchmark_cmd = [
            trtexec_path,
            f"--loadEngine={engine_path}",
            f"--warmUp={args.trtexec_warmup_ms}",
            f"--duration={args.trtexec_duration_s}",
            f"--avgRuns={args.trtexec_avg_runs}",
        ]
        benchmark_output = _run_trtexec(benchmark_cmd, f"Running TensorRT benchmark for {mode} (static-shape retry):")
    metrics = _parse_trtexec_benchmark(benchmark_output)
    summary = _format_trtexec_benchmark(metrics)
    if actual_precision is not None:
        summary = f"{summary}, build_precision={actual_precision}"
    print(f"TensorRT benchmark for {mode}: {summary}")
    return summary


def _maybe_retry_trtexec_with_non_fp8_export(
    args,
    mode,
    checkpoint_path,
    onnx_path,
    engine_path,
    model_config,
    enable_nested_fallbacks,
    exc,
):
    error_text = str(exc)
    if not getattr(args, "fallback_disable_fp8_on_trt_internal_error", False):
        return None
    if mode != "te-decomposed" or not args.use_fp8:
        return None
    if not _is_trtexec_myelin_ssa_error(error_text):
        return None

    fallback_onnx_path = _append_artifact_suffix(onnx_path, "_trt_nofp8")
    fallback_engine_path = _append_artifact_suffix(engine_path, "_trt_nofp8")
    print(
        "Detected a TensorRT Myelin SSA internal error while building the FP8 te-decomposed graph. "
        "Keeping the original FP8 ONNX artifact and retrying TensorRT validation with a non-FP8 re-export."
    )
    print(f"  Original FP8 ONNX: {onnx_path}")
    print(f"  Fallback TRT ONNX: {fallback_onnx_path}")

    fallback_export_args = _make_export_args(
        args,
        checkpoint_path,
        fallback_onnx_path,
        mode,
        enable_nested_fallbacks,
    )
    fallback_export_args.use_fp8 = False

    fallback_onnx_path, model, input_spatial, input_global = export(fallback_export_args)
    _verify_export(args, mode, fallback_onnx_path, model, input_spatial, input_global)
    benchmark_summary = _maybe_run_trtexec(args, fallback_onnx_path, fallback_engine_path, model_config, mode)

    detail = (
        f"{onnx_path} | FP8 TRT unsupported in current TRT/Myelin path; "
        f"fallback_non_fp8={fallback_onnx_path}"
    )
    if benchmark_summary is not None:
        detail = f"{detail} | TRT {benchmark_summary}"
    return detail


def _verify_export(args, mode, onnx_path, model, input_spatial, input_global):
    if not args.verify_onnxruntime:
        return

    if mode == "fp8-manual":
        atol, rtol = 0.5, 0.1
    elif mode == "legacy":
        atol, rtol = 1e-5, 1e-5
    else:
        atol, rtol = 1e-4, 1e-4
    verify(
        onnx_path,
        model,
        input_spatial,
        input_global,
        provider=args.ort_provider,
        atol=atol,
        rtol=rtol,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate a random TE checkpoint and export ONNX with multiple methods"
    )
    parser.add_argument("--config", default="b24c1024", choices=list(configs.config_of_name.keys()),
                        help="Model config to generate (default: b24c1024)")
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "test", "models"),
                        help="Directory for generated checkpoint/onnx/engine files")
    parser.add_argument("--checkpoint", default=None,
                        help="Output checkpoint path (default: <output-dir>/<config>_te_random.ckpt)")
    parser.add_argument("--output", default=None,
                        help="Output ONNX path. With multiple modes, mode suffixes are appended automatically")
    parser.add_argument("--engine", default=None,
                        help="TensorRT engine path. With multiple modes, mode suffixes are appended automatically")
    parser.add_argument("--modes", nargs="+", choices=ALL_MODES, default=DEFAULT_MODES,
                        help="Export modes to run in order (default: te-official te-decomposed legacy). "
                             "fp8-manual uses pure PyTorch with manual FP8 Q/DQ (no TE dependency)")
    parser.add_argument("--device", default="cuda",
                        help="Torch device for official TE export (default: cuda)")
    parser.add_argument("--pos-len", type=int, default=19, help="Board size (default: 19)")
    parser.add_argument("--score-mode", type=str, default="simple",
                        choices=["mixop", "mix", "simple"], help="Score belief head mode")
    parser.add_argument("--export-scope", type=str, default="full", choices=["full", "stem", "blocks", "trunk"],
                        help="Export the full model, stem-only, blocks-only, or stem+blocks (default: full)")
    parser.add_argument("--use-fp8", action="store_true",
                        help="Build the random TE checkpoint and TE-based exports with FP8-enabled module layout")
    parser.add_argument("--fp8-recipe", type=str, default="float8-current-scaling",
                        choices=["float8-current-scaling"],
                        help="FP8 recipe passed through to export_onnx.py when --use-fp8 is enabled")
    parser.add_argument("--opset", type=int, default=DEFAULT_ONNX_OPSET,
                        help=f"ONNX opset version passed to export_onnx.py (default: {DEFAULT_ONNX_OPSET})")
    parser.add_argument("--dynamic-batch", action="store_true",
                        help="Enable dynamic batch shapes during te-official export")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed (default: 1234)")
    parser.add_argument("--init-std", type=float, default=0.02,
                        help="Initialization std passed to model.initialize() (default: 0.02)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="TensorRT build/benchmark batch size (default: 1)")
    parser.add_argument("--verify-onnxruntime", action="store_true",
                        help="Also compare PyTorch and ONNX outputs with onnxruntime")
    parser.add_argument("--ort-provider", default="CPUExecutionProvider",
                        help="onnxruntime provider used by --verify-onnxruntime")
    parser.add_argument("--trtexec-bin", default="trtexec",
                        help="trtexec binary name or absolute path (default: trtexec)")
    parser.add_argument("--te-official-trt-precision", choices=TRT_PRECISIONS, default="fp32",
                        help="TensorRT precision requested for te-official mode (default: fp32)")
    parser.add_argument("--te-decomposed-trt-precision", choices=TRT_PRECISIONS, default="fp8",
                        help="TensorRT precision requested for te-decomposed mode (default: fp8)")
    parser.add_argument("--legacy-trt-precision", choices=TRT_PRECISIONS, default="bf16",
                        help="TensorRT precision requested for legacy mode (default: bf16)")
    parser.add_argument("--fp8-manual-trt-precision", choices=TRT_PRECISIONS, default="fp8",
                        help="TensorRT precision requested for fp8-manual mode (default: fp8)")
    parser.add_argument("--skip-trtexec", action="store_true",
                        help="Do not run TensorRT build or benchmark")
    parser.add_argument("--skip-trtexec-benchmark", action="store_true",
                        help="Build TensorRT engines but skip runtime benchmarking")
    parser.add_argument("--trtexec-warmup-ms", type=int, default=200,
                        help="Warmup time passed to trtexec --warmUp (default: 200)")
    parser.add_argument("--trtexec-duration-s", type=int, default=10,
                        help="Benchmark duration passed to trtexec --duration (default: 10)")
    parser.add_argument("--trtexec-avg-runs", type=int, default=100,
                        help="Average runs passed to trtexec --avgRuns (default: 100)")
    parser.add_argument("--trt-plugins", nargs="*", default=[],
                        help="Extra TRT plugin shared libraries to load via trtexec --plugins (e.g. libtransformer_engine.so)")
    parser.add_argument("--convert-fp8-qdq", action="store_true",
                        help="Convert trt::TRT_FP8 custom ops to standard ONNX QuantizeLinear/DequantizeLinear before TRT build "
                             "(workaround for TRT Myelin compiler bug with TE's custom FP8 ops)")
    parser.add_argument("--split-blocks", action="store_true",
                        help="Export each transformer block as a separate ONNX model and build individual TRT engines "
                             "(workaround for TRT Myelin compiler bug with large FP8 graphs)")
    parser.add_argument("--fallback-disable-fp8-on-trt-internal-error", action="store_true",
                        help="If TensorRT hits a Myelin internal error on te-decomposed FP8 export, keep the FP8 ONNX artifact "
                             "but re-export that mode without FP8 for TensorRT validation")
    parser.add_argument("--fallback-to-te-decomposed-on-te-export-error", action="store_true",
                        help="If the official TE export fails, retry with a decomposed TE export path before considering legacy export")
    parser.add_argument("--fallback-to-legacy-on-te-export-error", action="store_true",
                        help="If TE-based export fails, fall back to legacy export so TensorRT validation can continue")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    needs_te = any(m in _TE_MODES for m in args.modes)
    if needs_te:
        checkpoint_path = args.checkpoint or os.path.join(args.output_dir, f"{args.config}_te_random.ckpt")
        model_config = _save_random_te_checkpoint(args, checkpoint_path)
    else:
        checkpoint_path = args.checkpoint or os.path.join(args.output_dir, f"{args.config}_random.ckpt")
        model_config = _save_random_checkpoint(args, checkpoint_path)
    enable_nested_fallbacks = len(args.modes) == 1

    results = []
    for mode in args.modes:
        onnx_path = _resolve_mode_artifact_path(
            args.output, args.output_dir, args.config, mode, ".onnx", single_mode=enable_nested_fallbacks
        )
        engine_path = _resolve_mode_artifact_path(
            args.engine, args.output_dir, args.config, mode, ".plan", single_mode=enable_nested_fallbacks
        )

        print(f"\n=== Export mode: {mode} ===")
        export_args = _make_export_args(args, checkpoint_path, onnx_path, mode, enable_nested_fallbacks)

        # Per-block export path: export each transformer block as a separate
        # ONNX model and build individual TRT engines.
        if args.split_blocks and mode == "te-decomposed":
            try:
                block_paths = export_per_block(export_args)
                block_ok = 0
                for i, block_path in enumerate(block_paths):
                    print(f"\n--- Block {i}/{len(block_paths)} ---")
                    if args.convert_fp8_qdq:
                        convert_trt_fp8_to_standard_qdq(block_path)
                    block_engine = os.path.splitext(block_path)[0] + ".plan"
                    _maybe_run_trtexec(args, block_path, block_engine, model_config, mode)
                    block_ok += 1
                detail = f"{block_ok}/{len(block_paths)} blocks built"
                results.append((mode, "OK", detail))
            except RuntimeError as exc:
                error_text = str(exc)
                if args.use_fp8 and _is_trtexec_myelin_ssa_error(error_text):
                    # FP8 Q/DQ triggers a Myelin GVN bug in TRT 10.15.1
                    # even on single blocks.  Fall through to the normal
                    # export path which supports --fallback-disable-fp8-on-trt-internal-error.
                    print(
                        "Split-blocks FP8 build hit Myelin SSA error. "
                        "Falling through to standard export path for fallback handling."
                    )
                else:
                    print(f"ERROR: mode {mode} (split-blocks) failed")
                    print(error_text)
                    results.append((mode, "FAILED", error_text))
                    continue

        try:
            onnx_path, model, input_spatial, input_global = export(export_args)
            _verify_export(args, mode, onnx_path, model, input_spatial, input_global)
            if args.convert_fp8_qdq:
                convert_trt_fp8_to_standard_qdq(onnx_path)
            benchmark_summary = _maybe_run_trtexec(args, onnx_path, engine_path, model_config, mode)
            detail = onnx_path
            if benchmark_summary is not None:
                detail = f"{onnx_path} | TRT {benchmark_summary}"
            results.append((mode, "OK", detail))
        except RuntimeError as exc:
            fallback_detail = _maybe_retry_trtexec_with_non_fp8_export(
                args,
                mode,
                checkpoint_path,
                onnx_path,
                engine_path,
                model_config,
                enable_nested_fallbacks,
                exc,
            )
            if fallback_detail is not None:
                results.append((mode, "OK", fallback_detail))
                continue
            print(f"ERROR: mode {mode} failed")
            error_text = str(exc)
            if mode == "te-decomposed" and args.use_fp8 and _is_trtexec_myelin_ssa_error(error_text):
                error_text = (
                    f"{error_text}\n"
                    "Hint: this looks like a TensorRT/Myelin internal FP8 build bug on the te-decomposed path. "
                    "Try rerunning without --use-fp8, or pass --fallback-disable-fp8-on-trt-internal-error "
                    "to keep the FP8 ONNX artifact but validate TensorRT with a non-FP8 re-export."
                )
            print(error_text)
            results.append((mode, "FAILED", error_text))

    print("\nExport summary:")
    any_success = False
    for mode, status, detail in results:
        print(f"  {mode:14s} {status:7s} {detail}")
        if status == "OK":
            any_success = True

    if not any_success:
        raise SystemExit(1)

    print("Validation flow finished")


if __name__ == "__main__":
    main()
