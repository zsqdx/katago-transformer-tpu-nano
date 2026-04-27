#!/usr/bin/python3
"""
Preprocess KataGo NPZ training data for faster loading.

Converts compressed NPZ files to uncompressed NPZ with:
- Pre-unpacked binary inputs (uint8 NCHW instead of packed bits)
- Optional pre-applied symmetry augmentations (expand or random)
- Embedded pos_len metadata for validation

Usage:
    python3 preprocess.py \
        --input-dir ../data/shuffleddata/xxx/train \
        --output-dir ../data/preprocessed/xxx/train \
        --pos-len 19 \
        --symmetry-type xyt \
        --symmetry-mode expand \
        --workers 8
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Symmetry helpers (copied from data.py to avoid import issues with torch in
# multiprocessing workers — these operate on CPU tensors)
# ---------------------------------------------------------------------------

def apply_symmetry(tensor, symm):
    """Apply a symmetry operation to tensor (..., W, W)."""
    if symm == 0:
        return tensor
    if symm == 1:
        return tensor.transpose(-2, -1).flip(-2)
    if symm == 2:
        return tensor.flip(-1).flip(-2)
    if symm == 3:
        return tensor.transpose(-2, -1).flip(-1)
    if symm == 4:
        return tensor.transpose(-2, -1)
    if symm == 5:
        return tensor.flip(-1)
    if symm == 6:
        return tensor.transpose(-2, -1).flip(-1).flip(-2)
    if symm == 7:
        return tensor.flip(-2)


def apply_symmetry_policy(tensor, symm, pos_len):
    """Apply symmetry to policy tensor (N, C, pos_len*pos_len+1) handling pass index."""
    batch_size = tensor.shape[0]
    channels = tensor.shape[1]
    tensor_without_pass = tensor[:, :, :-1].view(batch_size, channels, pos_len, pos_len)
    tensor_transformed = apply_symmetry(tensor_without_pass, symm)
    return torch.cat((
        tensor_transformed.reshape(batch_size, channels, pos_len * pos_len),
        tensor[:, :, -1:]
    ), dim=2)


def get_allowed_symmetries(symmetry_type):
    """Return list of allowed symmetry indices for a given symmetry type."""
    if symmetry_type == "xyt":
        return [0, 1, 2, 3, 4, 5, 6, 7]
    elif symmetry_type == "xy":
        return [0, 2, 5, 7]
    elif symmetry_type == "x":
        return [0, 5]
    elif symmetry_type == "x+y":
        return [0, 2]
    elif symmetry_type == "t":
        return [0, 4]
    elif symmetry_type == "none" or symmetry_type is None:
        return [0]
    else:
        raise ValueError(f"Unknown symmetry type: {symmetry_type}")


# ---------------------------------------------------------------------------
# History matrix helpers (from data.py — operates on CPU torch)
# ---------------------------------------------------------------------------

NUM_BIN_FEATURES = 22


def build_history_matrices():
    """Build h_base (1, 22, 22) and h_builder (5, 22, 22) on CPU."""
    h_base = torch.diag(torch.tensor([
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # 0-8
        0.0, 0.0, 0.0, 0.0, 0.0,  # 9-13: move history
        1.0,  # 14: ladder
        0.0, 0.0,  # 15-16: ladder history
        1.0, 1.0, 1.0, 1.0, 1.0,  # 17-21
    ]))
    h_base[14, 15] = 1.0
    h_base[14, 16] = 1.0

    h0 = torch.zeros(NUM_BIN_FEATURES, NUM_BIN_FEATURES)
    h0[9, 9] = 1.0
    h0[14, 15] = -1.0
    h0[14, 16] = -1.0
    h0[15, 15] = 1.0
    h0[15, 16] = 1.0

    h1 = torch.zeros(NUM_BIN_FEATURES, NUM_BIN_FEATURES)
    h1[10, 10] = 1.0
    h1[15, 16] = -1.0
    h1[16, 16] = 1.0

    h2 = torch.zeros(NUM_BIN_FEATURES, NUM_BIN_FEATURES)
    h2[11, 11] = 1.0

    h3 = torch.zeros(NUM_BIN_FEATURES, NUM_BIN_FEATURES)
    h3[12, 12] = 1.0

    h4 = torch.zeros(NUM_BIN_FEATURES, NUM_BIN_FEATURES)
    h4[13, 13] = 1.0

    h_base = h_base.reshape(1, NUM_BIN_FEATURES, NUM_BIN_FEATURES)
    h_builder = torch.stack((h0, h1, h2, h3, h4), dim=0)
    return h_base, h_builder


def apply_history_matrices_np(binaryInputNCHW, globalInputNC, globalTargetsNC,
                              num_global_features, h_base, h_builder):
    """Apply random history truncation. Inputs/outputs are numpy arrays.

    binaryInputNCHW must be float32. globalInputNC must be float32.
    Returns (binaryInputNCHW, globalInputNC) as float32 numpy arrays.
    """
    ref = globalTargetsNC[:, 36:41]  # (N, 5)
    rng = np.random.default_rng()
    should_stop = (rng.random(ref.shape, dtype=np.float32) >= 0.98)
    include_history = (np.cumsum(should_stop, axis=1) <= 0.1).astype(np.float32)  # (N, 5)

    inc_t = torch.from_numpy(include_history)
    h_matrix = h_base + torch.einsum("bi,ijk->bjk", inc_t, h_builder)  # (N, 22, 22)

    bin_t = torch.from_numpy(binaryInputNCHW)
    bin_t = torch.einsum("bijk,bil->bljk", bin_t, h_matrix)

    glob_t = torch.from_numpy(globalInputNC)
    pad_width = num_global_features - include_history.shape[1]
    mask = torch.nn.functional.pad(inc_t, (0, pad_width), value=1.0)
    glob_t = glob_t * mask

    return bin_t.numpy(), glob_t.numpy()


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------

def apply_symmetry_to_arrays(binaryInputNCHW, policyTargetsNCMove, valueTargetsNCHW,
                             qValueTargetsNCMove, symm, pos_len):
    """Apply a single symmetry transform to all spatial arrays (as torch tensors on CPU)."""
    bin_t = torch.from_numpy(binaryInputNCHW)
    pol_t = torch.from_numpy(policyTargetsNCMove)
    val_t = torch.from_numpy(valueTargetsNCHW)

    bin_t = apply_symmetry(bin_t, symm)
    pol_t = apply_symmetry_policy(pol_t, symm, pos_len)
    val_t = apply_symmetry(val_t, symm)

    result_bin = bin_t.numpy()
    result_pol = pol_t.numpy()
    result_val = val_t.numpy()

    result_q = None
    if qValueTargetsNCMove is not None:
        q_t = torch.from_numpy(qValueTargetsNCMove)
        q_t = apply_symmetry_policy(q_t, symm, pos_len)
        result_q = q_t.numpy()

    return result_bin, result_pol, result_val, result_q


def save_npz_atomic(output_path, arrays_dict):
    """Save NPZ with atomic write (write to .tmp then rename)."""
    # np.savez auto-appends .npz if the path doesn't end with it,
    # so the tmp file must also end with .npz to avoid a mismatch.
    assert output_path.endswith(".npz")
    tmp_path = output_path[:-4] + ".tmp.npz"
    np.savez(tmp_path, **arrays_dict)
    os.replace(tmp_path, output_path)


def build_output_dict(binaryInputNCHW, globalInputNC, policyTargetsNCMove,
                      globalTargetsNC, scoreDistrN, valueTargetsNCHW,
                      metadataInputNC, qValueTargetsNCMove, pos_len):
    """Build the dict to save as NPZ."""
    d = {
        "binaryInputNCHW": binaryInputNCHW,
        "globalInputNC": globalInputNC,
        "policyTargetsNCMove": policyTargetsNCMove,
        "globalTargetsNC": globalTargetsNC,
        "scoreDistrN": scoreDistrN,
        "valueTargetsNCHW": valueTargetsNCHW,
        "pos_len": np.array(pos_len, dtype=np.int32),
    }
    if metadataInputNC is not None:
        d["metadataInputNC"] = metadataInputNC
    if qValueTargetsNCMove is not None:
        d["qValueTargetsNCMove"] = qValueTargetsNCMove
    return d


def preprocess_single_file(input_path, output_dir, pos_len, symmetry_type, symmetry_mode,
                           enable_history_matrices=False, num_global_features=19,
                           h_base=None, h_builder=None):
    """
    Preprocess a single NPZ file.

    Returns list of output file paths created.
    """
    basename = os.path.splitext(os.path.basename(input_path))[0]

    # 1. Load compressed NPZ
    with np.load(input_path) as npz:
        binaryInputNCHWPacked = npz["binaryInputNCHWPacked"]
        globalInputNC = npz["globalInputNC"]
        policyTargetsNCMove = npz["policyTargetsNCMove"]
        globalTargetsNC = npz["globalTargetsNC"]
        scoreDistrN = npz["scoreDistrN"]
        valueTargetsNCHW = npz["valueTargetsNCHW"]
        metadataInputNC = npz["metadataInputNC"] if "metadataInputNC" in npz else None
        qValueTargetsNCMove = npz["qValueTargetsNCMove"] if "qValueTargetsNCMove" in npz else None

    # 2. Unpack binary inputs
    binaryInputNCHW = np.unpackbits(binaryInputNCHWPacked, axis=2)
    assert len(binaryInputNCHW.shape) == 3
    assert binaryInputNCHW.shape[2] >= pos_len * pos_len
    binaryInputNCHW = binaryInputNCHW[:, :, :pos_len * pos_len]
    binaryInputNCHW = binaryInputNCHW.reshape(
        binaryInputNCHW.shape[0], binaryInputNCHW.shape[1], pos_len, pos_len
    )  # uint8

    # 2.5. Apply history matrices (random history truncation)
    if enable_history_matrices:
        binaryInputNCHW = binaryInputNCHW.astype(np.float32)
        globalInputNC = globalInputNC.astype(np.float32)
        binaryInputNCHW, globalInputNC = apply_history_matrices_np(
            binaryInputNCHW, globalInputNC, globalTargetsNC,
            num_global_features, h_base, h_builder)

    # 3. Validate dimensions
    num_samples = binaryInputNCHW.shape[0]
    assert policyTargetsNCMove.shape[0] == num_samples
    assert policyTargetsNCMove.shape[2] == pos_len * pos_len + 1, \
        f"Policy dim {policyTargetsNCMove.shape[2]} != {pos_len * pos_len + 1}"
    assert valueTargetsNCHW.shape[2] == pos_len and valueTargetsNCHW.shape[3] == pos_len, \
        f"Value spatial dims {valueTargetsNCHW.shape[2:]}, expected ({pos_len}, {pos_len})"

    # 4. Apply symmetry
    allowed_symms = get_allowed_symmetries(symmetry_type)
    output_files = []

    if symmetry_mode == "expand":
        for idx, symm in enumerate(allowed_symms):
            if symm == 0:
                out_bin, out_pol, out_val, out_q = (
                    binaryInputNCHW, policyTargetsNCMove, valueTargetsNCHW, qValueTargetsNCMove
                )
            else:
                out_bin, out_pol, out_val, out_q = apply_symmetry_to_arrays(
                    binaryInputNCHW, policyTargetsNCMove, valueTargetsNCHW,
                    qValueTargetsNCMove, symm, pos_len
                )

            out_name = f"{basename}_s{idx}.npz"
            out_path = os.path.join(output_dir, out_name)
            d = build_output_dict(out_bin, globalInputNC, out_pol,
                                  globalTargetsNC, scoreDistrN, out_val,
                                  metadataInputNC, out_q, pos_len)
            save_npz_atomic(out_path, d)
            output_files.append(out_path)

    elif symmetry_mode == "random":
        # Per-sample random symmetry
        rng = np.random.default_rng()
        symm_indices = rng.integers(0, len(allowed_symms), size=num_samples)

        out_bin = np.empty_like(binaryInputNCHW)
        out_pol = np.empty_like(policyTargetsNCMove)
        out_val = np.empty_like(valueTargetsNCHW)
        out_q = np.empty_like(qValueTargetsNCMove) if qValueTargetsNCMove is not None else None

        # Group samples by symmetry for batch processing
        for symm_local_idx in range(len(allowed_symms)):
            mask = symm_indices == symm_local_idx
            if not np.any(mask):
                continue
            symm = allowed_symms[symm_local_idx]
            if symm == 0:
                out_bin[mask] = binaryInputNCHW[mask]
                out_pol[mask] = policyTargetsNCMove[mask]
                out_val[mask] = valueTargetsNCHW[mask]
                if out_q is not None:
                    out_q[mask] = qValueTargetsNCMove[mask]
            else:
                s_bin, s_pol, s_val, s_q = apply_symmetry_to_arrays(
                    binaryInputNCHW[mask], policyTargetsNCMove[mask],
                    valueTargetsNCHW[mask],
                    qValueTargetsNCMove[mask] if qValueTargetsNCMove is not None else None,
                    symm, pos_len
                )
                out_bin[mask] = s_bin
                out_pol[mask] = s_pol
                out_val[mask] = s_val
                if out_q is not None:
                    out_q[mask] = s_q

        out_name = f"{basename}.npz"
        out_path = os.path.join(output_dir, out_name)
        d = build_output_dict(out_bin, globalInputNC, out_pol,
                              globalTargetsNC, scoreDistrN, out_val,
                              metadataInputNC, out_q, pos_len)
        save_npz_atomic(out_path, d)
        output_files.append(out_path)

    else:
        # No symmetry — just save unpacked
        out_name = f"{basename}.npz"
        out_path = os.path.join(output_dir, out_name)
        d = build_output_dict(binaryInputNCHW, globalInputNC, policyTargetsNCMove,
                              globalTargetsNC, scoreDistrN, valueTargetsNCHW,
                              metadataInputNC, qValueTargetsNCMove, pos_len)
        save_npz_atomic(out_path, d)
        output_files.append(out_path)

    return output_files


def _worker(args):
    """Wrapper for ProcessPoolExecutor."""
    (input_path, output_dir, pos_len, symmetry_type, symmetry_mode,
     enable_history_matrices, num_global_features, h_base, h_builder) = args
    try:
        files = preprocess_single_file(
            input_path, output_dir, pos_len, symmetry_type, symmetry_mode,
            enable_history_matrices, num_global_features, h_base, h_builder)
        return input_path, files, None
    except Exception as e:
        return input_path, [], e


def main():
    parser = argparse.ArgumentParser(description="Preprocess KataGo NPZ data for faster training")
    parser.add_argument("--input-dir", required=True, help="Directory containing original compressed NPZ files")
    parser.add_argument("--output-dir", required=True, help="Directory to write preprocessed NPZ files")
    parser.add_argument("--pos-len", type=int, default=19, help="Board size (default: 19)")
    parser.add_argument("--symmetry-type", type=str, default="none",
                        choices=["xyt", "xy", "x", "x+y", "t", "none"],
                        help="Symmetry type (default: none)")
    parser.add_argument("--symmetry-mode", type=str, default="expand",
                        choices=["expand", "random"],
                        help="Symmetry mode: expand=generate all variants, random=one random per sample (default: expand)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    parser.add_argument("--enable-history-matrices", action="store_true",
                        help="Apply random history truncation (2%% per-step cutoff)")
    parser.add_argument("--num-global-features", type=int, default=19,
                        help="Number of global input features (default: 19)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )

    input_files = sorted(glob(os.path.join(args.input_dir, "*.npz")))
    if not input_files:
        logging.error(f"No NPZ files found in {args.input_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    logging.info(f"Input: {args.input_dir} ({len(input_files)} files)")
    logging.info(f"Output: {args.output_dir}")
    logging.info(f"pos_len={args.pos_len}, symmetry_type={args.symmetry_type}, "
                 f"symmetry_mode={args.symmetry_mode}, workers={args.workers}, "
                 f"history_matrices={args.enable_history_matrices}")

    h_base, h_builder = None, None
    if args.enable_history_matrices:
        h_base, h_builder = build_history_matrices()
        logging.info(f"History matrices enabled (num_global_features={args.num_global_features})")

    allowed_symms = get_allowed_symmetries(args.symmetry_type)
    if args.symmetry_mode == "expand":
        expected_output = len(input_files) * len(allowed_symms)
        logging.info(f"Expand mode: {len(allowed_symms)} symmetries per file → ~{expected_output} output files")
    else:
        logging.info(f"Random mode: 1 output file per input, {len(allowed_symms)} symmetries sampled")

    t0 = time.perf_counter()
    total_output_files = 0
    errors = 0

    tasks = [(f, args.output_dir, args.pos_len, args.symmetry_type, args.symmetry_mode,
              args.enable_history_matrices, args.num_global_features, h_base, h_builder)
             for f in input_files]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_worker, task): task[0] for task in tasks}
        for i, future in enumerate(as_completed(futures)):
            input_path, output_files, error = future.result()
            if error is not None:
                logging.error(f"FAILED {os.path.basename(input_path)}: {error}")
                errors += 1
            else:
                total_output_files += len(output_files)
                if (i + 1) % 10 == 0 or (i + 1) == len(input_files):
                    elapsed = time.perf_counter() - t0
                    logging.info(f"Progress: {i + 1}/{len(input_files)} files, "
                                 f"{total_output_files} outputs, {elapsed:.1f}s")

    elapsed = time.perf_counter() - t0
    logging.info(f"Done: {total_output_files} output files from {len(input_files)} inputs "
                 f"in {elapsed:.1f}s ({errors} errors)")


if __name__ == "__main__":
    main()
