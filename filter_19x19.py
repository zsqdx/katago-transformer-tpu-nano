#!/usr/bin/env python3
"""Filter NPZ training data to keep only 19x19 board positions.

For KataGo data stored with pos_len=19, smaller boards are padded and channel 0
(the on-board mask) has zeros in padded positions. True 19x19 boards have channel 0
all-ones. This script filters each NPZ file row-by-row, keeping only 19x19 rows.

Usage:
    python3 filter_19x19.py --input-dir <dir> --output-dir <dir> [--workers 8]
"""

import argparse
import multiprocessing
import os
import sys
import time

import numpy as np

# For pos_len=19: 19*19=361 bits packed into ceil(361/8)=46 bytes.
# Full 19x19 board: first 45 bytes = 0xFF (360 ones), byte 45 = 0x80 (bit 360 = 1, rest 0).
POS_LEN = 19
EXPECTED_PACKED_BYTES = (POS_LEN * POS_LEN + 7) // 8  # 46
FULL_BYTES = (POS_LEN * POS_LEN) // 8  # 45
LAST_BYTE_VAL = 1 << (8 - (POS_LEN * POS_LEN - FULL_BYTES * 8))  # 128

REQUIRED_KEYS = [
    "binaryInputNCHWPacked",
    "globalInputNC",
    "policyTargetsNCMove",
    "globalTargetsNC",
    "scoreDistrN",
    "valueTargetsNCHW",
]
OPTIONAL_KEYS = []


def is_19x19_mask(packed):
    """Fast check: does channel 0 of binaryInputNCHWPacked indicate a full 19x19 board?

    Args:
        packed: array of shape (N, C, 46), dtype uint8

    Returns:
        boolean array of shape (N,)
    """
    ch0 = packed[:, 0, :]  # (N, 46)
    return np.all(ch0[:, :FULL_BYTES] == 255, axis=1) & (ch0[:, FULL_BYTES] == LAST_BYTE_VAL)


def filter_one_file(args):
    """Filter a single NPZ file to keep only 19x19 rows.

    Returns: (input_path, total_rows, kept_rows, output_path_or_None)
    """
    input_path, output_path = args
    try:
        with np.load(input_path) as npz:
            if "binaryInputNCHWPacked" not in npz:
                return (input_path, 0, 0, None, "missing binaryInputNCHWPacked")

            packed = npz["binaryInputNCHWPacked"]
            if packed.shape[2] != EXPECTED_PACKED_BYTES:
                return (input_path, packed.shape[0], 0, None,
                        f"packed_bytes={packed.shape[2]}, expected {EXPECTED_PACKED_BYTES}")

            total = packed.shape[0]
            mask = is_19x19_mask(packed)
            kept = int(mask.sum())

            if kept == 0:
                return (input_path, total, 0, None, "no 19x19 rows")

            # Build output dict with only kept rows
            out = {}
            for key in REQUIRED_KEYS:
                out[key] = npz[key][mask]
            for key in OPTIONAL_KEYS:
                if key in npz:
                    out[key] = npz[key][mask]

            np.savez_compressed(output_path, **out)
            return (input_path, total, kept, output_path, None)

    except Exception as e:
        return (input_path, 0, 0, None, str(e))


def find_npz_files(input_dir):
    """Recursively find all .npz files under input_dir."""
    npz_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".npz"):
                npz_files.append(os.path.join(root, f))
    npz_files.sort()
    return npz_files


def main():
    parser = argparse.ArgumentParser(description="Filter NPZ files to keep only 19x19 board data.")
    parser.add_argument("--input-dir", required=True, help="Input directory with NPZ files (searched recursively)")
    parser.add_argument("--output-dir", required=True, help="Output directory for filtered NPZ files")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers (default: 8)")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    num_workers = args.workers

    if not os.path.isdir(input_dir):
        print(f"ERROR: input directory does not exist: {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Scanning {input_dir} for NPZ files...")
    npz_files = find_npz_files(input_dir)
    print(f"Found {len(npz_files)} NPZ files")

    if not npz_files:
        print("No NPZ files found, exiting.")
        sys.exit(0)

    # Build (input_path, output_path) pairs. Output uses basename only (flat directory).
    # Handle duplicate basenames by appending a counter.
    seen_names = {}
    tasks = []
    for fpath in npz_files:
        basename = os.path.basename(fpath)
        if basename in seen_names:
            seen_names[basename] += 1
            name, ext = os.path.splitext(basename)
            basename = f"{name}_{seen_names[basename]}{ext}"
        else:
            seen_names[basename] = 0
        out_path = os.path.join(output_dir, basename)
        tasks.append((fpath, out_path))

    print(f"Filtering with {num_workers} workers...")
    t0 = time.time()

    total_files = len(tasks)
    total_rows_all = 0
    kept_rows_all = 0
    skipped_files = 0
    error_files = 0

    with multiprocessing.Pool(num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(filter_one_file, tasks, chunksize=32)):
            input_path, total, kept, out_path, err = result
            total_rows_all += total
            kept_rows_all += kept
            if err:
                if "no 19x19 rows" in err:
                    skipped_files += 1
                else:
                    error_files += 1
                    print(f"  WARN: {os.path.basename(input_path)}: {err}")
            if (i + 1) % 5000 == 0 or (i + 1) == total_files:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  [{i+1}/{total_files}] {rate:.0f} files/s | "
                      f"kept {kept_rows_all}/{total_rows_all} rows "
                      f"({100*kept_rows_all/max(total_rows_all,1):.1f}%)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Total files:   {total_files}")
    print(f"  Output files:  {total_files - skipped_files - error_files}")
    print(f"  Skipped (no 19x19): {skipped_files}")
    print(f"  Errors:        {error_files}")
    print(f"  Total rows:    {total_rows_all}")
    print(f"  Kept rows:     {kept_rows_all} ({100*kept_rows_all/max(total_rows_all,1):.1f}%)")
    print(f"  Filtered rows: {total_rows_all - kept_rows_all}")


if __name__ == "__main__":
    main()
