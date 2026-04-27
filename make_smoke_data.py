#!/usr/bin/env python3
"""Create tiny synthetic KataGo nano training data for smoke tests."""

import argparse
import os

import numpy as np

import configs


def _one_hot_rows(rng, rows, width):
    arr = np.zeros((rows, width), dtype=np.float32)
    arr[np.arange(rows), rng.integers(0, width, size=rows)] = 1.0
    return arr


def _make_file(path, samples, pos_len, model_config, rng):
    num_bin = configs.get_num_bin_input_features(model_config)
    num_global = configs.get_num_global_input_features(model_config)
    board_area = pos_len * pos_len
    moves = board_area + 1
    score_len = (board_area + 60) * 2

    binary = np.zeros((samples, num_bin, pos_len, pos_len), dtype=np.uint8)
    binary[:, 0, :, :] = 1
    binary[:, 1:min(num_bin, 9), :, :] = rng.integers(
        0, 2, size=(samples, max(0, min(num_bin, 9) - 1), pos_len, pos_len), dtype=np.uint8
    )

    global_input = rng.normal(0.0, 0.1, size=(samples, num_global)).astype(np.float32)
    global_input[:, -1] = rng.integers(0, 2, size=samples).astype(np.float32)

    policy = np.zeros((samples, 2, moves), dtype=np.float32)
    policy[:, 0, :] = _one_hot_rows(rng, samples, moves)
    policy[:, 1, :] = _one_hot_rows(rng, samples, moves)

    global_targets = np.zeros((samples, 41), dtype=np.float32)
    global_targets[:, 0:3] = _one_hot_rows(rng, samples, 3)
    global_targets[:, 3] = rng.normal(0.0, 5.0, size=samples)
    global_targets[:, 4:7] = _one_hot_rows(rng, samples, 3)
    global_targets[:, 7] = rng.normal(0.0, 5.0, size=samples)
    global_targets[:, 8:11] = _one_hot_rows(rng, samples, 3)
    global_targets[:, 11] = rng.normal(0.0, 5.0, size=samples)
    global_targets[:, 12:15] = _one_hot_rows(rng, samples, 3)
    global_targets[:, 15] = rng.normal(0.0, 5.0, size=samples)
    global_targets[:, 21] = rng.normal(0.0, 5.0, size=samples)
    global_targets[:, 22] = rng.uniform(1.0, 20.0, size=samples)
    global_targets[:, 24] = 0.0
    global_targets[:, 25] = 1.0
    global_targets[:, 26] = 1.0
    global_targets[:, 27] = 1.0
    global_targets[:, 28] = 1.0
    global_targets[:, 29] = 1.0
    global_targets[:, 33] = 1.0
    global_targets[:, 34] = 1.0
    global_targets[:, 35] = 0.0
    global_targets[:, 36:41] = 1.0

    score_distr = np.zeros((samples, score_len), dtype=np.float32)
    score_distr[:, score_len // 2] = 100.0

    value_targets = np.zeros((samples, 5, pos_len, pos_len), dtype=np.float32)

    np.savez_compressed(
        path,
        binaryInputNCHW=binary,
        globalInputNC=global_input,
        policyTargetsNCMove=policy,
        globalTargetsNC=global_targets,
        scoreDistrN=score_distr,
        valueTargetsNCHW=value_targets,
        pos_len=np.array(pos_len, dtype=np.int32),
    )


def main():
    parser = argparse.ArgumentParser(description="Generate tiny synthetic nano training data")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-kind", default="b12c192", choices=list(configs.config_of_name.keys()))
    parser.add_argument("--pos-len", type=int, default=9)
    parser.add_argument("--samples-per-file", type=int, default=8)
    parser.add_argument("--train-files", type=int, default=2)
    parser.add_argument("--val-files", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    model_config = configs.config_of_name[args.model_kind].copy()
    rng = np.random.default_rng(args.seed)

    train_dir = os.path.join(args.out_dir, "train")
    val_dir = os.path.join(args.out_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for i in range(args.train_files):
        _make_file(os.path.join(train_dir, f"smoke_train_{i:03d}.npz"),
                   args.samples_per_file, args.pos_len, model_config, rng)
    for i in range(args.val_files):
        _make_file(os.path.join(val_dir, f"smoke_val_{i:03d}.npz"),
                   args.samples_per_file, args.pos_len, model_config, rng)

    print(f"Wrote smoke data to {args.out_dir}")


if __name__ == "__main__":
    main()
