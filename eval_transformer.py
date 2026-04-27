"""Evaluate KataGo Transformer checkpoint on validation data.

Computes the full training loss decomposition (all 26 metrics from losses.py):
  loss, p0loss, p1loss, vloss, oloss, smloss, leadloss, vtimeloss, pacc1, ...

Usage:
    python eval_transformer.py --checkpoint /path/to/checkpoint.ckpt --data-dir /path/to/val_npz
    python eval_transformer.py --checkpoint /path/to/checkpoint.ckpt --data-dir /path/to/val_npz --use-ema
"""

import argparse
import glob
import os
import time

import torch
import torch.amp

import configs
from data import read_npz_training_data
from losses import _METRIC_KEYS, postprocess_and_loss_core


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate KataGo Transformer on validation data")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt checkpoint file")
    parser.add_argument("--data-dir", required=True, help="Directory containing .npz validation files")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size (default: 256)")
    parser.add_argument("--pos-len", type=int, default=19, help="Board size (default: 19)")
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto)")
    parser.add_argument("--max-files", type=int, default=None, help="Max number of npz files to process")
    parser.add_argument("--use-ema", action="store_true", help="Use EMA shadow weights if available")
    parser.add_argument("--score-mode", type=str, default="simple", choices=["mixop", "mix", "simple"],
                        help="Score belief head mode (default: simple)")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    return parser.parse_args()


def _convert_te_to_pt(state_dict):
    """Convert TE-format state dict to PT format if needed."""
    try:
        from model_te import detect_checkpoint_format, convert_checkpoint_te_to_model
        if detect_checkpoint_format(state_dict) == "te":
            print("  Converting TE checkpoint to model.py format")
            return convert_checkpoint_te_to_model(state_dict)
    except ImportError:
        pass
    return state_dict


def _detect_score_mode(state_dict):
    """Auto-detect score_mode from checkpoint keys."""
    for key in state_dict:
        if "linear_s_mix" in key:
            return "mixop"
        if "linear_s_simple" in key:
            return "simple"
    return "simple"


def load_model_from_checkpoint(checkpoint_path, pos_len, score_mode, use_ema=False):
    """Load model and config from a training checkpoint."""
    from model import Model

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_config = configs.migrate_config(state.get("config", {}))

    # Auto-detect score_mode from checkpoint weights
    ref_state = state.get("ema_shadow", state["model"]) if use_ema else state["model"]
    detected_score_mode = _detect_score_mode(ref_state)
    if score_mode != detected_score_mode:
        print(f"  Auto-detected score_mode='{detected_score_mode}' from checkpoint (overriding '{score_mode}')")
        score_mode = detected_score_mode

    varlen = state.get("varlen", False)
    model = Model(model_config, pos_len, score_mode=score_mode, varlen=varlen)

    # Choose weights: EMA shadow or regular model
    if use_ema and "ema_shadow" in state:
        print("  Using EMA shadow weights")
        ema_state = _convert_te_to_pt(state["ema_shadow"])
        # EMA shadow only contains trainable params; merge into full state dict
        model_state = model.state_dict()
        matched = 0
        for k, v in ema_state.items():
            if k in model_state:
                model_state[k] = v
                matched += 1
        print(f"  Loaded {matched}/{len(ema_state)} EMA params")
        model.load_state_dict(model_state)
    elif use_ema:
        print("  WARNING: --use-ema requested but no EMA state in checkpoint, using regular weights")
        model.load_state_dict(_convert_te_to_pt(state["model"]))
    else:
        model.load_state_dict(_convert_te_to_pt(state["model"]))

    # Restore seki moving average state
    model.moving_unowned_proportion_sum = state.get("moving_unowned_proportion_sum", 0.0)
    model.moving_unowned_proportion_weight = state.get("moving_unowned_proportion_weight", 0.0)

    info = {
        "config": model_config,
        "global_step": state.get("global_step", "?"),
        "total_samples_trained": state.get("total_samples_trained", "?"),
        "varlen": varlen,
    }
    return model, model_config, info


@torch.no_grad()
def evaluate(model, model_config, npz_files, batch_size, pos_len, device, use_amp=True, varlen=False):
    """Run evaluation and return metrics dict."""
    model.eval()
    model.to(device)

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

    # Seki moving average (eval uses fixed scale=7.0, but we still need the tensors)
    moving_sum_t = torch.tensor(model.moving_unowned_proportion_sum, device=device)
    moving_weight_t = torch.tensor(model.moving_unowned_proportion_weight, device=device)

    # Accumulators
    accum = {k: 0.0 for k in _METRIC_KEYS}
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
        varlen=varlen,
    )

    for batch in data_gen:
        N = batch["binaryInputNCHW"].shape[0]

        with torch.amp.autocast(amp_device, dtype=amp_dtype, enabled=use_amp):
            outputs = model(batch["binaryInputNCHW"], batch["globalInputNC"])

        eval_mask = batch["binaryInputNCHW"][:, 0:1, :, :].contiguous() if varlen else None
        _, metrics_stack, _, _ = postprocess_and_loss_core(
            outputs,
            model.value_head.score_belief_offset_vector,
            batch["policyTargetsNCMove"],
            batch["globalTargetsNC"],
            batch["scoreDistrN"],
            batch["valueTargetsNCHW"],
            pos_len,
            moving_sum_t,
            moving_weight_t,
            is_training=False,
            mask=eval_mask,
        )

        batch_metrics = dict(zip(_METRIC_KEYS, metrics_stack.tolist()))
        for k in _METRIC_KEYS:
            accum[k] += batch_metrics[k]

        total_samples += N
        num_batches += 1

        if num_batches % 50 == 0:
            elapsed = time.time() - t0
            wsum = max(accum["wsum"], 1e-10)
            print(f"  batch {num_batches}: {total_samples} samples, "
                  f"{total_samples / elapsed:.0f} samp/s, "
                  f"pacc1={accum['pacc1'] / wsum:.4f}, "
                  f"p0loss={accum['p0loss'] / wsum:.4f}")

    elapsed = time.time() - t0

    if accum["wsum"] < 1e-8:
        print("WARNING: no valid samples found")
        return {}

    wsum = accum["wsum"]
    metrics = {}
    for k in _METRIC_KEYS:
        if k == "loss":
            metrics[k] = accum[k] / max(num_batches, 1)
        elif k == "wsum":
            metrics[k] = wsum
        else:
            metrics[k] = accum[k] / wsum

    metrics["samples"] = total_samples
    metrics["batches"] = num_batches
    metrics["time"] = elapsed
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

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, model_config, info = load_model_from_checkpoint(
        args.checkpoint, args.pos_len, args.score_mode, use_ema=args.use_ema,
    )
    print(f"  Config: {model_config}")
    print(f"  Step: {info['global_step']}, Samples trained: {info['total_samples_trained']}")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

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
    metrics = evaluate(model, model_config, npz_files, args.batch_size, args.pos_len, device, use_amp=use_amp, varlen=info["varlen"])

    if not metrics:
        return

    # Print results
    print("\n" + "=" * 60)
    print(f"Results for: {os.path.basename(args.checkpoint)}")
    print("=" * 60)

    # Primary metrics (same as eval_cnn.py)
    print(f"  pacc1     = {metrics['pacc1']:.4f}")
    print(f"  p0loss    = {metrics['p0loss']:.4f}")
    print(f"  vloss     = {metrics['vloss']:.4f}")
    print(f"  oloss     = {metrics['oloss']:.4f}")
    print(f"  smloss    = {metrics['smloss']:.6f}")
    print(f"  leadloss  = {metrics['leadloss']:.6f}")
    print(f"  vtimeloss = {metrics['vtimeloss']:.6f}")

    # Additional transformer-specific metrics
    print(f"  ----")
    print(f"  loss      = {metrics['loss']:.4f}")
    print(f"  p1loss    = {metrics['p1loss']:.4f}")
    print(f"  p0softloss= {metrics['p0softloss']:.4f}")
    print(f"  p1softloss= {metrics['p1softloss']:.4f}")
    print(f"  p0lopt    = {metrics['p0lopt']:.4f}")
    print(f"  p0sopt    = {metrics['p0sopt']:.4f}")
    print(f"  tdvloss1  = {metrics['tdvloss1']:.4f}")
    print(f"  tdvloss2  = {metrics['tdvloss2']:.4f}")
    print(f"  tdvloss3  = {metrics['tdvloss3']:.4f}")
    print(f"  tdsloss   = {metrics['tdsloss']:.6f}")
    print(f"  sloss     = {metrics['sloss']:.4f}")
    print(f"  fploss    = {metrics['fploss']:.4f}")
    print(f"  skloss    = {metrics['skloss']:.4f}")
    print(f"  sbcdfloss = {metrics['sbcdfloss']:.4f}")
    print(f"  sbpdfloss = {metrics['sbpdfloss']:.4f}")
    print(f"  sdregloss = {metrics['sdregloss']:.6f}")
    print(f"  evstloss  = {metrics['evstloss']:.4f}")
    print(f"  esstloss  = {metrics['esstloss']:.6f}")

    print(f"  ----")
    print(f"  samples   = {metrics['samples']}")
    print(f"  wsum      = {metrics['wsum']:.1f}")
    print(f"  time      = {metrics['time']:.1f}s")
    print(f"  speed     = {metrics['samples'] / metrics['time']:.0f} samp/s")
    print("=" * 60)


if __name__ == "__main__":
    main()
