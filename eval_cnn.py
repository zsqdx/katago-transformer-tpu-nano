"""Evaluate KataGo CNN (bin.gz) models on validation data.

Computes only metrics available from bin.gz outputs:
  pacc1, p0loss, vloss, oloss, smloss, leadloss, vtimeloss

Usage:
    python eval_cnn.py --model ~/.katrain/kata1-b28c512nbt-*.bin.gz --data-dir /path/to/val_npz
"""

import argparse
import glob
import os
import time

import torch
import torch.nn.functional as F

from model import SoftPlusWithGradientFloor, cross_entropy
from load_bin_gz import load_bin_gz
from data import read_npz_training_data


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate KataGo CNN model on validation data")
    parser.add_argument("--model", required=True, help="Path to .bin.gz model file")
    parser.add_argument("--data-dir", required=True, help="Directory containing .npz validation files")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size (default: 256)")
    parser.add_argument("--pos-len", type=int, default=19, help="Board size (default: 19)")
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto)")
    parser.add_argument("--max-files", type=int, default=None, help="Max number of npz files to process")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, npz_files, batch_size, pos_len, device):
    """Run evaluation and return metrics dict."""
    model.eval()
    model.to(device)

    # Minimal config for data loader (v15 format: 22 bin features, 19 global features)
    model_config = {"version": 15}
    pos_area = pos_len * pos_len

    # Accumulators
    total_p0loss = 0.0
    total_vloss = 0.0
    total_oloss = 0.0
    total_smloss = 0.0
    total_leadloss = 0.0
    total_vtimeloss = 0.0
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

        # Forward
        outputs = model(input_spatial, input_global)
        (out_policy, out_value, out_misc, out_moremisc,
         out_ownership, _out_scoring, _out_futurepos, _out_seki,
         _out_scorebelief) = outputs

        # --- Targets ---
        target_policy_player = target_policy[:, 0, :]
        target_policy_player = target_policy_player / torch.sum(target_policy_player, dim=1, keepdim=True)

        target_value = target_global[:, 0:3]
        target_scoremean = target_global[:, 3]
        target_lead = target_global[:, 21]
        target_variance_time = target_global[:, 22]
        target_ownership = target_value_nchw[:, 0, :, :]

        global_weight = target_global[:, 25]
        target_weight_policy_player = target_global[:, 26]
        target_weight_ownership = target_global[:, 27]
        target_weight_lead = target_global[:, 29]
        target_weight_value = 1.0 - target_global[:, 35]

        w_policy = global_weight * target_weight_policy_player
        w_value = global_weight * target_weight_value
        w_ownership = global_weight * target_weight_ownership
        w_lead = global_weight * target_weight_lead

        # --- Postprocess model outputs ---
        policy_logits = out_policy
        value_logits = out_value
        ownership_pretanh = out_ownership

        pred_scoremean = out_misc[:, 0] * 20.0
        pred_lead = out_misc[:, 2] * 20.0
        pred_variance_time = SoftPlusWithGradientFloor.apply(out_misc[:, 3], 0.05, False) * 40.0

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
        total_pacc1 += pacc1.item()
        total_wsum += wsum.item()
        total_samples += N
        num_batches += 1

        if num_batches % 50 == 0:
            elapsed = time.time() - t0
            samples_per_sec = total_samples / elapsed
            print(f"  batch {num_batches}: {total_samples} samples, {samples_per_sec:.0f} samp/s, "
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
        "wsum": total_wsum,
        "samples": total_samples,
        "batches": num_batches,
        "time": elapsed,
    }
    return metrics


def main():
    args = parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model = load_bin_gz(args.model, pos_len=args.pos_len)

    # Find npz files
    npz_files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if not npz_files:
        print(f"ERROR: No .npz files found in {args.data_dir}")
        return

    if args.max_files is not None:
        npz_files = npz_files[:args.max_files]
    print(f"\nValidation files: {len(npz_files)}")

    # Quick shape check with one batch
    print("\nRunning shape check...")
    model.eval()
    model.to(device)
    with torch.no_grad():
        dummy_spatial = torch.ones(1, 22, args.pos_len, args.pos_len, device=device)
        dummy_global = torch.zeros(1, 19, device=device)
        outputs = model(dummy_spatial, dummy_global)
        L = args.pos_len * args.pos_len
        assert outputs[0].shape == (1, 6, L + 1), f"Policy shape: {outputs[0].shape}"
        assert outputs[1].shape == (1, 3), f"Value shape: {outputs[1].shape}"
        assert outputs[4].shape == (1, 1, args.pos_len, args.pos_len), f"Ownership shape: {outputs[4].shape}"
        print("  Shape check passed!")

    # Run evaluation
    print(f"\nEvaluating with batch_size={args.batch_size}...")
    metrics = evaluate(model, npz_files, args.batch_size, args.pos_len, device)

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
    print(f"  ----")
    print(f"  samples   = {metrics['samples']}")
    print(f"  wsum      = {metrics['wsum']:.1f}")
    print(f"  time      = {metrics['time']:.1f}s")
    print(f"  speed     = {metrics['samples'] / metrics['time']:.0f} samp/s")
    print("=" * 60)


if __name__ == "__main__":
    main()
