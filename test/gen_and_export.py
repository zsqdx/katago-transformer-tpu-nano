#!/usr/bin/env python3
"""Generate a randomly initialized KataGo nano model and export it to ONNX.

Supports both predefined configs and fully custom model structures.

Usage:
    # Predefined config (outputs to test/models/)
    python test/gen_and_export.py --config b12c192

    # Custom model structure
    python test/gen_and_export.py --num-layers 6 --hidden-size 128 --num-heads 4 --ffn-dim 384

    # With onnxruntime verification
    python test/gen_and_export.py --config b12c192 --verify
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

import configs

# ---------------------------------------------------------------------------
# Patch nn.RMSNorm.forward so that ONNX export sees only basic math ops
# ---------------------------------------------------------------------------
_original_rms_norm_forward = None
if hasattr(nn, "RMSNorm"):
    _original_rms_norm_forward = nn.RMSNorm.forward

    def _manual_rms_norm_forward(self, x):
        x_f32 = x.float()
        mean_sq = (x_f32 * x_f32).mean(-1, keepdim=True)
        inv_rms = torch.rsqrt(mean_sq + torch.tensor(self.eps, dtype=x_f32.dtype, device=x_f32.device))
        return (self.weight * (x_f32 * inv_rms)).type_as(x)

    nn.RMSNorm.forward = _manual_rms_norm_forward


OUTPUT_NAMES = [
    "out_policy",
    "out_value",
    "out_misc",
    "out_moremisc",
    "out_ownership",
    "out_scoring",
    "out_futurepos",
    "out_seki",
    "out_scorebelief",
]


def main():
    parser = argparse.ArgumentParser(description="Generate randomly initialized model and export to ONNX")

    # Predefined config
    parser.add_argument("--config", default=None, choices=list(configs.config_of_name.keys()),
                        help="Predefined config name (mutually exclusive with custom params)")

    # Custom model structure
    parser.add_argument("--num-layers", type=int, default=None, help="Number of transformer blocks")
    parser.add_argument("--hidden-size", type=int, default=None, help="Hidden dimension")
    parser.add_argument("--num-heads", type=int, default=None, help="Number of attention heads")
    parser.add_argument("--ffn-dim", type=int, default=None, help="SwiGLU FFN intermediate dim (default: hidden_size * 8 // 3)")
    parser.add_argument("--num-scorebeliefs", type=int, default=8, help="Score belief mixtures (default: 8)")

    # Export options
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    parser.add_argument("--output", default=None, help="Output .onnx path (default: test/models/<name>.onnx)")
    parser.add_argument("--checkpoint", default=None, help="Also save checkpoint (default: test/models/<name>.ckpt)")
    parser.add_argument("--pos-len", type=int, default=19, help="Board size (default: 19)")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version (default: 18)")
    parser.add_argument("--verify", action="store_true", help="Verify with onnxruntime")
    parser.add_argument("--use-te", action="store_true",
                        help="Use TransformerEngine model for checkpoint generation (ONNX export still uses model.py)")
    args = parser.parse_args()

    # Resolve config
    custom_params = [args.num_layers, args.hidden_size, args.num_heads]
    has_custom = any(p is not None for p in custom_params)

    if args.config and has_custom:
        parser.error("--config and custom params (--num-layers/--hidden-size/--num-heads) are mutually exclusive")
    if not args.config and not has_custom:
        parser.error("specify either --config or custom params (--num-layers, --hidden-size, --num-heads)")
    if has_custom and not all(p is not None for p in custom_params):
        parser.error("--num-layers, --hidden-size, and --num-heads are all required for custom config")

    if args.config:
        model_config = configs.config_of_name[args.config]
        model_name = args.config
        print(f"Config: {args.config} -> {model_config}")
    else:
        model_config = configs.make_config(
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            ffn_dim=args.ffn_dim,
            num_scorebeliefs=args.num_scorebeliefs,
        )
        model_name = f"b{args.num_layers}c{args.hidden_size}"
        print(f"Custom config ({model_name}): {model_config}")

    # Create and initialize random model
    if args.use_te:
        from model_te import Model as TEModel
        te_model = TEModel(model_config, args.pos_len, score_mode="simple")
        te_model.initialize(init_std=0.02)
        te_model.eval()
        num_params = sum(p.numel() for p in te_model.parameters())
        print(f"Parameters (TE): {num_params:,}")
    else:
        te_model = None

    from model import Model
    model = Model(model_config, args.pos_len, score_mode="simple")
    if args.use_te:
        # Convert TE weights to model.py format for ONNX export
        from model_te import convert_checkpoint_te_to_model
        model.load_state_dict(convert_checkpoint_te_to_model(te_model.state_dict()))
    else:
        model.initialize(init_std=0.02)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Resolve output paths
    os.makedirs(models_dir, exist_ok=True)
    onnx_path = args.output or os.path.join(models_dir, f"{model_name}.onnx")
    ckpt_path = args.checkpoint or os.path.join(models_dir, f"{model_name}.ckpt")

    # Save checkpoint (TE format if --use-te, model.py format otherwise)
    if args.use_te:
        torch.save({"model": te_model.state_dict(), "config": model_config}, ckpt_path)
    else:
        torch.save({"model": model.state_dict(), "config": model_config}, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    # Dummy inputs
    num_bin = configs.get_num_bin_input_features(model_config)
    num_global = configs.get_num_global_input_features(model_config)
    H = W = args.pos_len

    input_spatial = torch.randn(1, num_bin, H, W)
    input_global = torch.randn(1, num_global)

    # Export ONNX (always uses model.py's Model for compatibility)
    dynamic_axes = {"input_spatial": {0: "batch"}, "input_global": {0: "batch"}}
    for name in OUTPUT_NAMES:
        dynamic_axes[name] = {0: "batch"}

    print(f"Exporting ONNX (opset {args.opset}) ...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_spatial, input_global),
            onnx_path,
            input_names=["input_spatial", "input_global"],
            output_names=OUTPUT_NAMES,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
        )

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"Saved: {onnx_path} ({size_mb:.1f} MB)")

    # Verify
    if args.verify:
        import numpy as np
        import onnxruntime as ort

        if _original_rms_norm_forward is not None:
            nn.RMSNorm.forward = _original_rms_norm_forward

        print("\nVerifying with onnxruntime ...")
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        with torch.no_grad():
            pt_outputs = model(input_spatial, input_global)

        ort_inputs = {
            "input_spatial": input_spatial.numpy(),
            "input_global": input_global.numpy(),
        }
        ort_outputs = sess.run(None, ort_inputs)

        all_close = True
        for i, name in enumerate(OUTPUT_NAMES):
            pt_arr = pt_outputs[i].numpy()
            ort_arr = ort_outputs[i]
            max_diff = np.max(np.abs(pt_arr - ort_arr))
            ok = np.allclose(pt_arr, ort_arr, atol=1e-5)
            status = "OK" if ok else "MISMATCH"
            print(f"  {name:20s} shape={str(pt_arr.shape):20s} max_diff={max_diff:.2e}  {status}")
            if not ok:
                all_close = False

        if all_close:
            print("All outputs match!")
        else:
            print("WARNING: some outputs have significant numerical differences")
            sys.exit(1)


if __name__ == "__main__":
    main()
