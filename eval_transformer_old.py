"""Evaluate train/-exported KataGo Transformer ONNX model on validation data.

Extracts weights from ONNX initializers into train/model_pytorch.py Model,
then evaluates using PyTorch inference.

Usage:
    python eval_transformer_old.py --model /path/to/b18c384h12tfrs.onnx --data-dir /path/to/val_npz
    python eval_transformer_old.py --model /path/to/b18c384h12tfrs.onnx --data-dir /path/to/val_npz --device cuda
"""

import argparse
import ast
import glob
import os
import sys
import time

import torch
import torch.amp
import torch.nn.functional as F

from model import SoftPlusWithGradientFloor, cross_entropy
from data import read_npz_training_data


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate train/ ONNX model on validation data")
    parser.add_argument("--model", required=True, help="Path to .onnx model file")
    parser.add_argument("--data-dir", required=True, help="Directory containing .npz validation files")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size (default: 256)")
    parser.add_argument("--pos-len", type=int, default=19, help="Board size (default: 19)")
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto)")
    parser.add_argument("--max-files", type=int, default=None, help="Max number of npz files to process")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    return parser.parse_args()


def _read_onnx_metadata(onnx_path):
    """Read model_config and other metadata from ONNX model."""
    import onnx

    model = onnx.load(onnx_path)
    metadata = {}
    for prop in model.metadata_props:
        metadata[prop.key] = prop.value
    return metadata, model


def _extract_onnx_to_state_dict(onnx_model):
    """Extract ONNX initializers into a PyTorch state_dict.

    Handles three categories:
    1. Named initializers (model.*) → strip prefix, use directly
    2. onnx::MatMul_* → sorted by ID, mapped sequentially to known layers, transposed
    3. onnx::Sub/Div/Mul (norm_trunkfinal bnorm) → mapped to running_mean/std/gamma
    """
    from onnx import numpy_helper

    state_dict = {}
    matmul_inits = []  # (id, tensor)
    norm_constants = {}  # op_type → tensor

    for init in onnx_model.graph.initializer:
        arr = numpy_helper.to_array(init)

        if init.name.startswith("model."):
            key = init.name[len("model."):]
            state_dict[key] = torch.from_numpy(arr.copy())

        elif init.name.startswith("onnx::MatMul_"):
            idx = int(init.name.split("_")[-1])
            matmul_inits.append((idx, arr))

        elif init.name.startswith("onnx::"):
            # Non-MatMul onnx:: constants (Sub/Div/Mul from norm_trunkfinal bnorm)
            op_type = init.name.split("::")[1].split("_")[0]
            norm_constants[op_type] = arr

    # Sort MatMul by ID
    matmul_inits.sort(key=lambda x: x[0])

    # Determine block count from named initializers
    block_indices = set()
    for key in state_dict:
        if key.startswith("blocks.") and ".norm1.weight" in key:
            block_indices.add(int(key.split(".")[1]))
    num_blocks = len(block_indices)

    expected_matmul = 1 + num_blocks * 7 + 2  # linear_global + blocks + policy head
    if len(matmul_inits) != expected_matmul:
        print(f"WARNING: expected {expected_matmul} MatMul initializers, got {len(matmul_inits)}")

    # Map MatMul weights (ONNX stores as transposed: (in, out) → PyTorch: (out, in))
    idx = 0

    # linear_global
    state_dict["linear_global.weight"] = torch.from_numpy(matmul_inits[idx][1].T.copy())
    idx += 1

    # Transformer blocks
    block_params = ["q_proj.weight", "k_proj.weight", "v_proj.weight", "out_proj.weight",
                    "ffn_linear1.weight", "ffn_linear_gate.weight", "ffn_linear2.weight"]
    for b in range(num_blocks):
        for param_name in block_params:
            state_dict[f"blocks.{b}.{param_name}"] = torch.from_numpy(matmul_inits[idx][1].T.copy())
            idx += 1

    # Policy head: linear_pass2 then linear_g
    state_dict["policy_head.linear_pass2.weight"] = torch.from_numpy(matmul_inits[idx][1].T.copy())
    idx += 1
    state_dict["policy_head.linear_g.weight"] = torch.from_numpy(matmul_inits[idx][1].T.copy())
    idx += 1

    # norm_trunkfinal bnorm constants
    if "Sub" in norm_constants:
        state_dict["norm_trunkfinal.running_mean"] = torch.from_numpy(
            norm_constants["Sub"].squeeze().copy()
        )
    if "Div" in norm_constants:
        state_dict["norm_trunkfinal.running_std"] = torch.from_numpy(
            norm_constants["Div"].squeeze().copy()
        )
    if "Mul" in norm_constants:
        # ONNX stores (gamma + 1), recover gamma
        gamma_plus_one = torch.from_numpy(norm_constants["Mul"].copy())
        state_dict["norm_trunkfinal.gamma"] = gamma_plus_one - 1.0

    return state_dict


def load_model_from_onnx(onnx_path, pos_len):
    """Load train/ Model from ONNX file by extracting weights."""
    # Add train/ to path for model_pytorch import
    train_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train")
    if train_dir not in sys.path:
        sys.path.insert(0, train_dir)
    from model_pytorch import Model

    # Read metadata and extract config
    metadata, onnx_model = _read_onnx_metadata(onnx_path)
    config_str = metadata.get("model_config")
    if config_str is None:
        print("ERROR: ONNX model has no model_config metadata")
        sys.exit(1)

    config = ast.literal_eval(config_str)
    print(f"  Config from ONNX metadata: norm_kind={config.get('norm_kind')}, "
          f"activation={config.get('activation', 'relu')}, "
          f"blocks={len(config.get('block_kind', []))}, "
          f"channels={config.get('trunk_num_channels')}")

    # Extract weights
    print("  Extracting weights from ONNX initializers...")
    state_dict = _extract_onnx_to_state_dict(onnx_model)
    print(f"  Extracted {len(state_dict)} weight tensors")

    # Build model
    model = Model(config, pos_len)

    # Load weights (strict=False for missing head weights)
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"  Missing keys ({len(result.missing_keys)}):")
        for k in result.missing_keys:
            print(f"    {k}")
    if result.unexpected_keys:
        print(f"  Unexpected keys ({len(result.unexpected_keys)}):")
        for k in result.unexpected_keys:
            print(f"    {k}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    return model, config


@torch.no_grad()
def evaluate(model, npz_files, batch_size, pos_len, device, use_amp=True):
    """Run evaluation and return metrics dict."""
    model.eval()
    model.to(device)

    model_config = {"version": 15}
    pos_area = pos_len * pos_len

    # AMP setup
    if device.type == "cuda":
        amp_device = "cuda"
        amp_dtype = torch.bfloat16
    elif device.type == "mps":
        amp_device = "mps"
        amp_dtype = torch.bfloat16
    else:
        amp_device = "cpu"
        amp_dtype = torch.bfloat16
        use_amp = False

    # Accumulators
    total_p0loss = 0.0
    total_vloss = 0.0
    total_oloss = 0.0
    total_smloss = 0.0
    total_leadloss = 0.0
    total_vtimeloss = 0.0
    total_tdvloss1 = 0.0
    total_tdvloss2 = 0.0
    total_tdvloss3 = 0.0
    total_tdsloss = 0.0
    total_evstloss = 0.0
    total_esstloss = 0.0
    total_pacc1 = 0.0
    total_wsum = 0.0
    total_samples = 0
    num_batches = 0

    t0 = time.time()

    data_gen = read_npz_training_data(
        npz_files=npz_files,
        batch_size=batch_size,
        world_size=1,
        rank=0,
        pos_len=pos_len,
        device=device,
        symmetry_type=None,
        include_meta=False,
        enable_history_matrices=False,
        model_config=model_config,
    )

    for batch in data_gen:
        input_spatial = batch["binaryInputNCHW"]
        input_global = batch["globalInputNC"]
        target_policy = batch["policyTargetsNCMove"]
        target_global = batch["globalTargetsNC"]
        target_value_nchw = batch["valueTargetsNCHW"]

        N = input_spatial.shape[0]

        with torch.amp.autocast(amp_device, dtype=amp_dtype, enabled=use_amp):
            # train/ Model returns ((outputs...),)
            outputs_by_heads = model(input_spatial, input_global, disable_mask=False)
            outputs = outputs_by_heads[0]

        (out_policy, out_value, out_miscvalue, out_moremiscvalue,
         out_ownership, _out_scoring, _out_futurepos, _out_seki,
         _out_scorebelief) = outputs

        # --- Targets ---
        target_policy_player = target_policy[:, 0, :]
        target_policy_player = target_policy_player / torch.sum(target_policy_player, dim=1, keepdim=True)

        target_value = target_global[:, 0:3]
        target_scoremean = target_global[:, 3]
        target_td_value = torch.stack(
            (target_global[:, 4:7], target_global[:, 8:11], target_global[:, 12:15]), dim=1
        )  # (N, 3, 3)
        target_td_score = torch.cat(
            (target_global[:, 7:8], target_global[:, 11:12], target_global[:, 15:16]), dim=1
        )  # (N, 3)
        target_lead = target_global[:, 21]
        target_variance_time = target_global[:, 22]
        target_ownership = target_value_nchw[:, 0, :, :]

        global_weight = target_global[:, 25]
        target_weight_policy_player = target_global[:, 26]
        target_weight_ownership = target_global[:, 27]
        target_weight_lead = target_global[:, 29]
        target_weight_value = 1.0 - target_global[:, 35]
        target_weight_td_value = 1.0 - target_global[:, 24]

        w_policy = global_weight * target_weight_policy_player
        w_value = global_weight * target_weight_value
        w_ownership = global_weight * target_weight_ownership
        w_lead = global_weight * target_weight_lead
        w_td = global_weight * target_weight_td_value

        # --- Postprocess model outputs ---
        policy_logits = out_policy
        value_logits = out_value
        ownership_pretanh = out_ownership

        pred_scoremean = out_miscvalue[:, 0] * 20.0
        pred_lead = out_miscvalue[:, 2] * 20.0
        pred_variance_time = SoftPlusWithGradientFloor.apply(out_miscvalue[:, 3], 0.05, False) * 40.0

        td_value_logits = torch.stack(
            (out_miscvalue[:, 4:7], out_miscvalue[:, 7:10], out_moremiscvalue[:, 2:5]), dim=1
        )  # (N, 3, 3)
        pred_td_score = out_moremiscvalue[:, 5:8] * 20.0  # (N, 3)
        pred_shortterm_value_error = SoftPlusWithGradientFloor.apply(out_moremiscvalue[:, 0], 0.05, True) * 0.25
        pred_shortterm_score_error = SoftPlusWithGradientFloor.apply(out_moremiscvalue[:, 1], 0.05, True) * 150.0

        # --- p0loss ---
        p0loss = (w_policy * cross_entropy(policy_logits[:, 0, :], target_policy_player, dim=1)).sum()

        # --- vloss ---
        vloss = 1.20 * (w_value * cross_entropy(value_logits, target_value, dim=1)).sum()

        # --- oloss ---
        pred_own_logits = torch.cat([ownership_pretanh, -ownership_pretanh], dim=1).view(N, 2, pos_area)
        target_own_probs = torch.stack([
            (1.0 + target_ownership) / 2.0,
            (1.0 - target_ownership) / 2.0,
        ], dim=1).view(N, 2, pos_area)
        oloss = 1.5 * (w_ownership * (
            torch.sum(cross_entropy(pred_own_logits, target_own_probs, dim=1), dim=1) / pos_area
        )).sum()

        # --- smloss ---
        smloss = 0.0015 * (w_ownership * F.huber_loss(
            pred_scoremean, target_scoremean, reduction="none", delta=12.0
        )).sum()

        # --- leadloss ---
        leadloss = 0.0060 * (w_lead * F.huber_loss(
            pred_lead, target_lead, reduction="none", delta=8.0
        )).sum()

        # --- vtimeloss ---
        vtimeloss = 0.0003 * (w_ownership * F.huber_loss(
            pred_variance_time, target_variance_time + 1e-5, reduction="none", delta=50.0
        )).sum()

        # --- tdvloss (temporal difference value loss) ---
        td_loss_raw = cross_entropy(td_value_logits, target_td_value, dim=2) - cross_entropy(
            torch.log(target_td_value + 1e-30), target_td_value, dim=2
        )
        td_loss_weighted = 1.20 * w_td.unsqueeze(1) * td_loss_raw
        tdvloss1 = td_loss_weighted[:, 0].sum()
        tdvloss2 = td_loss_weighted[:, 1].sum()
        tdvloss3 = td_loss_weighted[:, 2].sum()

        # --- tdsloss (temporal difference score loss) ---
        tdsloss = 0.0004 * (w_ownership * torch.sum(
            F.huber_loss(pred_td_score, target_td_score, reduction="none", delta=12.0), dim=1
        )).sum()

        # --- evstloss (shortterm value error loss) ---
        td_val_pred_probs = torch.softmax(td_value_logits[:, 2, :], dim=1)
        predvalue = (td_val_pred_probs[:, 0] - td_val_pred_probs[:, 1]).detach()
        realvalue = target_td_value[:, 2, 0] - target_td_value[:, 2, 1]
        sqerror_v = torch.square(predvalue - realvalue) + 1e-8
        evstloss = 2.0 * (w_ownership * F.huber_loss(
            pred_shortterm_value_error, sqerror_v, reduction="none", delta=0.4
        )).sum()

        # --- esstloss (shortterm score error loss) ---
        predscore = pred_td_score[:, 2].detach()
        realscore = target_td_score[:, 2]
        sqerror_s = torch.square(predscore - realscore) + 1e-4
        esstloss = 0.00002 * (w_ownership * F.huber_loss(
            pred_shortterm_score_error, sqerror_s, reduction="none", delta=100.0
        )).sum()

        # --- pacc1 ---
        pacc1 = (w_policy * (
            torch.argmax(policy_logits[:, 0, :], dim=1) == torch.argmax(target_policy_player, dim=1)
        ).float()).sum()

        wsum = global_weight.sum()

        # Accumulate
        total_p0loss += p0loss.item()
        total_vloss += vloss.item()
        total_oloss += oloss.item()
        total_smloss += smloss.item()
        total_leadloss += leadloss.item()
        total_vtimeloss += vtimeloss.item()
        total_tdvloss1 += tdvloss1.item()
        total_tdvloss2 += tdvloss2.item()
        total_tdvloss3 += tdvloss3.item()
        total_tdsloss += tdsloss.item()
        total_evstloss += evstloss.item()
        total_esstloss += esstloss.item()
        total_pacc1 += pacc1.item()
        total_wsum += wsum.item()
        total_samples += N
        num_batches += 1

        if num_batches % 50 == 0:
            elapsed = time.time() - t0
            print(f"  batch {num_batches}: {total_samples} samples, {total_samples / elapsed:.0f} samp/s, "
                  f"pacc1={total_pacc1 / total_wsum:.4f}")

    elapsed = time.time() - t0

    if total_wsum < 1e-8:
        print("WARNING: no valid samples found")
        return {}

    metrics = {
        "pacc1": total_pacc1 / total_wsum,
        "p0loss": total_p0loss / total_wsum,
        "vloss": total_vloss / total_wsum,
        "oloss": total_oloss / total_wsum,
        "smloss": total_smloss / total_wsum,
        "leadloss": total_leadloss / total_wsum,
        "vtimeloss": total_vtimeloss / total_wsum,
        "tdvloss1": total_tdvloss1 / total_wsum,
        "tdvloss2": total_tdvloss2 / total_wsum,
        "tdvloss3": total_tdvloss3 / total_wsum,
        "tdsloss": total_tdsloss / total_wsum,
        "evstloss": total_evstloss / total_wsum,
        "esstloss": total_esstloss / total_wsum,
        "wsum": total_wsum,
        "samples": total_samples,
        "batches": num_batches,
        "time": elapsed,
    }
    return metrics


def main():
    args = parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model from ONNX
    print(f"\nLoading ONNX model: {args.model}")
    model, config = load_model_from_onnx(args.model, args.pos_len)

    # Shape check
    print("\nRunning shape check...")
    model.eval()
    model.to(device)
    with torch.no_grad():
        dummy_spatial = torch.ones(1, 22, args.pos_len, args.pos_len, device=device)
        dummy_global = torch.zeros(1, 19, device=device)
        outputs = model(dummy_spatial, dummy_global, disable_mask=False)
        out = outputs[0]
        L = args.pos_len * args.pos_len
        assert out[0].shape == (1, 6, L + 1), f"Policy shape: {out[0].shape}"
        assert out[1].shape == (1, 3), f"Value shape: {out[1].shape}"
        assert out[4].shape == (1, 1, args.pos_len, args.pos_len), f"Ownership shape: {out[4].shape}"
        print("  Shape check passed!")

    # Find npz files
    npz_files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if not npz_files:
        print(f"ERROR: No .npz files found in {args.data_dir}")
        return

    if args.max_files is not None:
        npz_files = npz_files[:args.max_files]
    print(f"\nValidation files: {len(npz_files)}")

    # Run evaluation
    use_amp = not args.no_amp
    print(f"\nEvaluating with batch_size={args.batch_size}, amp={'on' if use_amp else 'off'}...")
    metrics = evaluate(model, npz_files, args.batch_size, args.pos_len, device, use_amp=use_amp)

    if not metrics:
        return

    # Print results
    print("\n" + "=" * 60)
    print(f"Results for: {os.path.basename(args.model)}")
    print("=" * 60)
    print(f"  pacc1     = {metrics['pacc1']:.4f}")
    print(f"  p0loss    = {metrics['p0loss']:.4f}")
    print(f"  vloss     = {metrics['vloss']:.4f}")
    print(f"  oloss     = {metrics['oloss']:.4f}")
    print(f"  smloss    = {metrics['smloss']:.6f}")
    print(f"  leadloss  = {metrics['leadloss']:.6f}")
    print(f"  vtimeloss = {metrics['vtimeloss']:.6f}")
    print(f"  tdvloss1  = {metrics['tdvloss1']:.6f}")
    print(f"  tdvloss2  = {metrics['tdvloss2']:.6f}")
    print(f"  tdvloss3  = {metrics['tdvloss3']:.6f}")
    print(f"  tdsloss   = {metrics['tdsloss']:.6f}")
    print(f"  evstloss  = {metrics['evstloss']:.6f}")
    print(f"  esstloss  = {metrics['esstloss']:.6f}")
    print(f"  ----")
    print(f"  samples   = {metrics['samples']}")
    print(f"  wsum      = {metrics['wsum']:.1f}")
    print(f"  time      = {metrics['time']:.1f}s")
    print(f"  speed     = {metrics['samples'] / metrics['time']:.0f} samp/s")
    print("=" * 60)


if __name__ == "__main__":
    main()
