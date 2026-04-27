"""Visualize attention patterns of KataGo Transformer via TensorBoard.

Loads a checkpoint and validation data, then writes attention heatmaps,
entropy, and distance statistics into TensorBoard event files.

Usage:
    # 1. Generate event files
    python visualize_attention.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --data-dir /path/to/val_npz \
        --logdir ./attn_tb \
        --num-samples 4 \
        --query-pos max_policy

    # 2. Browse in TensorBoard
    tensorboard --logdir ./attn_tb
"""

import argparse
import glob
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.amp
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import configs
from data import read_npz_training_data
from eval_transformer import load_model_from_checkpoint
from model import apply_rotary_emb


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize KataGo Transformer attention via TensorBoard")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt checkpoint file")
    parser.add_argument("--data-dir", required=True, help="Directory containing .npz validation files")
    parser.add_argument("--logdir", type=str, default="./attn_tb", help="TensorBoard log directory")
    parser.add_argument("--pos-len", type=int, default=19, help="Board size (default: 19)")
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto)")
    parser.add_argument("--use-ema", action="store_true", help="Use EMA shadow weights if available")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to visualize")
    parser.add_argument("--query-pos", type=str, default="max_policy",
                        help="Query position: integer (row*19+col), or 'max_policy' to use top policy move")
    parser.add_argument("--score-mode", type=str, default="simple", choices=["mixop", "mix", "simple"])
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for loading data")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Attention weight extraction
# ---------------------------------------------------------------------------

def extract_attention_weights(model, x, rope_cos, rope_sin, attn_mask=None):
    """Run transformer blocks explicitly, returning attention weights for each layer.

    Returns:
        x_out: final output after all blocks + norm_final
        all_attn_weights: list of (B, H, L, L) float32 cpu tensors, one per layer
    """
    all_attn_weights = []

    for block in model.blocks:
        B, L, C = x.shape
        x_normed = block.norm1(x)

        q = block.q_proj(x_normed).view(B, L, block.num_heads, block.head_dim)
        k = block.k_proj(x_normed).view(B, L, block.num_heads, block.head_dim)
        v = block.v_proj(x_normed).view(B, L, block.num_heads, block.head_dim)

        q, k = apply_rotary_emb(q, k, rope_cos, rope_sin)

        q = q.permute(0, 2, 1, 3)  # (B, H, L, D)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(block.head_dim))
        if attn_mask is not None:
            scores = scores + attn_mask
        attn_weights = torch.softmax(scores, dim=-1)  # (B, H, L, L)
        all_attn_weights.append(attn_weights.detach().float().cpu())

        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, L, C)

        if block.use_gated_attn:
            attn_out = torch.sigmoid(block.attn_gate_proj(x_normed)) * attn_out
        x = x + block.out_proj(attn_out)

        x_normed = block.norm2(x)
        w1_out = F.silu(block.ffn_w1(x_normed))
        wgate_out = block.ffn_wgate(x_normed)
        with torch.amp.autocast(x.device.type, enabled=False):
            ffn_hidden = (w1_out.float() * wgate_out.float()).to(x.dtype)
        x = x + block.ffn_w2(ffn_hidden)

    x_out = model.norm_final(x)
    return x_out, all_attn_weights


# ---------------------------------------------------------------------------
# Board state helpers
# ---------------------------------------------------------------------------

def extract_board_state(binary_input, pos_len):
    """Extract black/white stone positions from binary input channels.

    Returns: black_stones (H, W), white_stones (H, W) as bool arrays.
    """
    own_stones = binary_input[1].cpu().numpy().astype(bool)
    opp_stones = binary_input[2].cpu().numpy().astype(bool)
    if binary_input.shape[0] > 18:
        is_black_turn = binary_input[18].cpu().numpy().mean() > 0.5
    else:
        is_black_turn = True
    if is_black_turn:
        return own_stones, opp_stones
    else:
        return opp_stones, own_stones


# ---------------------------------------------------------------------------
# Matplotlib figure builders (for add_figure)
# ---------------------------------------------------------------------------

def make_board_figure(black_stones, white_stones, pos_len, title="Board State"):
    """Render the board state as a matplotlib figure."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    board_color = np.full((pos_len, pos_len, 3), [0.86, 0.72, 0.53])
    ax.imshow(board_color, interpolation="nearest")

    for i in range(pos_len):
        ax.axhline(i, color="black", linewidth=0.5, alpha=0.6)
        ax.axvline(i, color="black", linewidth=0.5, alpha=0.6)

    rows, cols = np.where(black_stones)
    ax.scatter(cols, rows, c="black", s=80, edgecolors="gray", linewidths=0.5, zorder=3)
    rows, cols = np.where(white_stones)
    ax.scatter(cols, rows, c="white", s=80, edgecolors="black", linewidths=0.5, zorder=3)

    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(pos_len))
    ax.set_yticks(range(pos_len))
    ax.set_xticklabels([chr(ord('A') + i + (1 if i >= 8 else 0)) for i in range(pos_len)], fontsize=6)
    ax.set_yticklabels([str(pos_len - i) for i in range(pos_len)], fontsize=6)
    plt.tight_layout()
    return fig


def make_attention_figure(attn_map_2d, pos_len, query_rc, title,
                          black_stones=None, white_stones=None):
    """Render a single attention heatmap as a matplotlib figure."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    im = ax.imshow(attn_map_2d, cmap="YlOrRd", interpolation="nearest",
                   vmin=0, vmax=attn_map_2d.max() + 1e-9)

    for i in range(pos_len):
        ax.axhline(i, color="black", linewidth=0.2, alpha=0.2)
        ax.axvline(i, color="black", linewidth=0.2, alpha=0.2)

    if black_stones is not None:
        rows, cols = np.where(black_stones)
        ax.scatter(cols, rows, c="black", s=40, edgecolors="white", linewidths=0.3, zorder=3)
    if white_stones is not None:
        rows, cols = np.where(white_stones)
        ax.scatter(cols, rows, c="white", s=40, edgecolors="black", linewidths=0.3, zorder=3)

    if query_rc is not None:
        qr, qc = query_rc
        ax.scatter([qc], [qr], c="lime", s=100, marker="*", edgecolors="black", linewidths=0.8, zorder=4)

    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(pos_len))
    ax.set_yticks(range(pos_len))
    ax.set_xticklabels([chr(ord('A') + i + (1 if i >= 8 else 0)) for i in range(pos_len)], fontsize=5)
    ax.set_yticklabels([str(pos_len - i) for i in range(pos_len)], fontsize=5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def make_layer_heads_grid_figure(attn_maps_2d, num_heads, pos_len, query_rc, layer_idx,
                                 black_stones=None, white_stones=None):
    """Render all heads of a layer in one grid figure."""
    ncols = min(num_heads, 8)
    nrows = (num_heads + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.8 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    vmax = attn_maps_2d.max() + 1e-9

    for h in range(num_heads):
        r, c = divmod(h, ncols)
        ax = axes[r, c]
        ax.imshow(attn_maps_2d[h], cmap="YlOrRd", interpolation="nearest", vmin=0, vmax=vmax)

        for i in range(pos_len):
            ax.axhline(i, color="black", linewidth=0.15, alpha=0.15)
            ax.axvline(i, color="black", linewidth=0.15, alpha=0.15)

        if black_stones is not None:
            rows, cols = np.where(black_stones)
            ax.scatter(cols, rows, c="black", s=10, edgecolors="white", linewidths=0.2, zorder=3)
        if white_stones is not None:
            rows, cols = np.where(white_stones)
            ax.scatter(cols, rows, c="white", s=10, edgecolors="black", linewidths=0.2, zorder=3)

        if query_rc is not None:
            qr, qc = query_rc
            ax.scatter([qc], [qr], c="lime", s=25, marker="*", edgecolors="black", linewidths=0.4, zorder=4)

        ax.set_title(f"Head {h}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    for h in range(num_heads, nrows * ncols):
        r, c = divmod(h, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(f"Layer {layer_idx} — Query: ({query_rc[0]}, {query_rc[1]})" if query_rc
                 else f"Layer {layer_idx}", fontsize=11)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, model_config, info = load_model_from_checkpoint(
        args.checkpoint, args.pos_len, args.score_mode, use_ema=args.use_ema
    )
    model.eval()
    model.to(device)
    num_layers = model_config["num_layers"]
    num_heads = model_config["num_heads"]
    print(f"  Model: {num_layers} layers, {num_heads} heads, hidden={model_config['hidden_size']}")
    print(f"  Step: {info['global_step']}, samples: {info['total_samples_trained']}")

    # Load data
    npz_files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {args.data_dir}")
    print(f"  Found {len(npz_files)} npz files")

    varlen = info.get("varlen", False)
    batch_size = max(args.batch_size, args.num_samples)
    data_gen = read_npz_training_data(
        npz_files=npz_files[:1],
        batch_size=batch_size,
        world_size=1,
        rank=0,
        pos_len=args.pos_len,
        device=device,
        symmetry_type=None,
        include_meta=False,
        enable_history_matrices=False,
        model_config=model_config,
        varlen=varlen,
    )

    batch = next(iter(data_gen))
    spatial = batch["binaryInputNCHW"]
    global_input = batch["globalInputNC"]
    N = spatial.shape[0]
    num_samples = min(args.num_samples, N)
    print(f"  Batch size: {N}, visualizing {num_samples} sample(s)")

    # AMP setup
    pos_len = args.pos_len
    L = pos_len * pos_len
    use_amp = not args.no_amp
    if device.type == "cuda":
        amp_dtype = torch.bfloat16
    elif device.type == "mps":
        amp_dtype = torch.bfloat16
    else:
        use_amp = False
        amp_dtype = torch.float32

    # Precompute distance matrix for attention distance stats
    rows_idx = torch.arange(pos_len).float()
    cols_idx = torch.arange(pos_len).float()
    pos_r = rows_idx.repeat_interleave(pos_len)  # (L,)
    pos_c = cols_idx.repeat(pos_len)              # (L,)
    dist_matrix = (pos_r.unsqueeze(0) - pos_r.unsqueeze(1)).abs() + \
                  (pos_c.unsqueeze(0) - pos_c.unsqueeze(1)).abs()  # (L, L)

    # Forward pass
    with torch.no_grad():
        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            x, attn_mask, mask_flat = model._forward_stem_impl(spatial, global_input)
            x_out, all_attn_weights = extract_attention_weights(
                model, x, model.rope_cos, model.rope_sin, attn_mask=attn_mask
            )
            x_fp32 = x_out.float()
            out_policy = model.policy_head(x_fp32, mask=mask_flat)

    # Write to TensorBoard
    writer = SummaryWriter(log_dir=args.logdir)
    print(f"  Writing TensorBoard events to: {args.logdir}")

    for si in range(num_samples):
        print(f"\n--- Sample {si} ---")

        # Board state
        black_stones, white_stones = extract_board_state(spatial[si], pos_len)
        board_fig = make_board_figure(black_stones, white_stones, pos_len,
                                      title=f"Sample {si} Board State")
        writer.add_figure("board/state", board_fig, global_step=si)

        # Query position
        policy_board = out_policy[si, 0, :-1]  # (L,)
        if args.query_pos == "max_policy":
            query_pos = int(policy_board.argmax().item())
        else:
            query_pos = int(args.query_pos)
        qr, qc = query_pos // pos_len, query_pos % pos_len
        query_rc = (qr, qc)
        print(f"  Query position: index={query_pos}, row={qr}, col={qc}")

        # Per-layer visualizations
        for layer_idx, attn_w in enumerate(all_attn_weights):
            # attn_w: (B, H, L, L)
            aw = attn_w[si]  # (H, L, L)

            # --- Images: mean attention ---
            mean_attn = aw.mean(dim=0)  # (L, L)
            attn_from_query = mean_attn[query_pos].numpy().reshape(pos_len, pos_len)
            fig = make_attention_figure(attn_from_query, pos_len, query_rc,
                                        title=f"Layer {layer_idx} Mean Attn",
                                        black_stones=black_stones, white_stones=white_stones)
            writer.add_figure(f"attn_mean/layer_{layer_idx:02d}", fig, global_step=si)

            # --- Images: all heads grid ---
            heads_from_query = aw[:, query_pos, :].numpy().reshape(num_heads, pos_len, pos_len)
            fig = make_layer_heads_grid_figure(heads_from_query, num_heads, pos_len,
                                               query_rc, layer_idx,
                                               black_stones=black_stones, white_stones=white_stones)
            writer.add_figure(f"attn_heads/layer_{layer_idx:02d}", fig, global_step=si)

            # --- Scalars: entropy ---
            log_aw = torch.log2(aw + 1e-10)
            entropy = -(aw * log_aw).sum(dim=-1).mean(dim=-1)  # (H,)
            for h in range(num_heads):
                writer.add_scalar(f"entropy/layer_{layer_idx:02d}/head_{h:02d}", entropy[h].item(), global_step=si)
            writer.add_scalar(f"entropy/layer_{layer_idx:02d}/mean", entropy.mean().item(), global_step=si)

            # --- Scalars: attention distance ---
            mean_dist = (aw * dist_matrix.unsqueeze(0)).sum(dim=-1).mean(dim=-1)  # (H,)
            for h in range(num_heads):
                writer.add_scalar(f"attn_distance/layer_{layer_idx:02d}/head_{h:02d}", mean_dist[h].item(), global_step=si)
            writer.add_scalar(f"attn_distance/layer_{layer_idx:02d}/mean", mean_dist.mean().item(), global_step=si)

        print(f"  Sample {si} done.")

    writer.close()
    print(f"\nAll done! Run: tensorboard --logdir {args.logdir}")


if __name__ == "__main__":
    main()
