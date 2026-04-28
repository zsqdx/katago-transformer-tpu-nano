"""Numpy data loader for the JAX TPU training path.

This intentionally avoids importing torch so that a JAX-only Colab runtime can
run the loader after installing jax[tpu].
"""

import glob
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import configs


def list_npz_files(data_dir, split):
    return sorted(glob.glob(os.path.join(data_dir, split, "*.npz")))


def sample_symmetry(symmetry_type, rand):
    if symmetry_type == "xyt":
        allowed = [0, 1, 2, 3, 4, 5, 6, 7]
    elif symmetry_type == "x":
        allowed = [0, 5]
    elif symmetry_type == "xy":
        allowed = [0, 2, 5, 7]
    elif symmetry_type == "x+y":
        allowed = [0, 2]
    elif symmetry_type == "t":
        allowed = [0, 4]
    else:
        raise ValueError(f"Unknown symmetry type {symmetry_type}")
    return allowed[int(rand.integers(0, len(allowed)))]


def apply_symmetry_np(array, symm):
    """Apply one of KataGo's eight board symmetries to an array ending in H,W."""
    if symm == 0:
        return array
    if symm == 1:
        return np.flip(np.swapaxes(array, -2, -1), axis=-2)
    if symm == 2:
        return np.flip(np.flip(array, axis=-1), axis=-2)
    if symm == 3:
        return np.flip(np.swapaxes(array, -2, -1), axis=-1)
    if symm == 4:
        return np.swapaxes(array, -2, -1)
    if symm == 5:
        return np.flip(array, axis=-1)
    if symm == 6:
        return np.flip(np.flip(np.swapaxes(array, -2, -1), axis=-1), axis=-2)
    if symm == 7:
        return np.flip(array, axis=-2)
    raise ValueError(f"Unknown symmetry {symm}")


def apply_symmetry_policy_np(array, symm, pos_len):
    batch_size = array.shape[0]
    channels = array.shape[1]
    no_pass = array[:, :, :-1].reshape((batch_size, channels, pos_len, pos_len))
    transformed = apply_symmetry_np(no_pass, symm)
    return np.concatenate(
        (transformed.reshape(batch_size, channels, pos_len * pos_len), array[:, :, -1:]),
        axis=2,
    )


def build_history_matrices_np(model_config):
    num_bin_features = configs.get_num_bin_input_features(model_config)
    if num_bin_features != 22:
        raise ValueError("History matrices are hardcoded for 22 binary input features")

    h_base = np.diag(np.array(
        [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
        dtype=np.float32,
    ))
    h_base[14, 15] = 1.0
    h_base[14, 16] = 1.0

    h0 = np.zeros((num_bin_features, num_bin_features), dtype=np.float32)
    h0[9, 9] = 1.0
    h0[14, 15] = -1.0
    h0[14, 16] = -1.0
    h0[15, 15] = 1.0
    h0[15, 16] = 1.0

    h1 = np.zeros((num_bin_features, num_bin_features), dtype=np.float32)
    h1[10, 10] = 1.0
    h1[15, 16] = -1.0
    h1[16, 16] = 1.0

    h2 = np.zeros((num_bin_features, num_bin_features), dtype=np.float32)
    h2[11, 11] = 1.0

    h3 = np.zeros((num_bin_features, num_bin_features), dtype=np.float32)
    h3[12, 12] = 1.0

    h4 = np.zeros((num_bin_features, num_bin_features), dtype=np.float32)
    h4[13, 13] = 1.0

    return h_base.reshape((1, num_bin_features, num_bin_features)), np.stack(
        (h0, h1, h2, h3, h4), axis=0
    )


def apply_history_matrices_np(
    model_config,
    binary_input,
    global_input,
    global_targets,
    h_base,
    h_builder,
    rand,
):
    num_global_features = configs.get_num_global_input_features(model_config)
    ref = global_targets[:, 36:41]
    should_stop_history = rand.random(ref.shape) >= 0.98
    include_history = (np.cumsum(should_stop_history, axis=1, dtype=np.float32) <= 0.1).astype(np.float32)

    h_matrix = h_base + np.einsum("bi,ijk->bjk", include_history, h_builder)
    binary_input = np.einsum("bijk,bil->bljk", binary_input, h_matrix).astype(np.float32, copy=False)

    pad_width = num_global_features - include_history.shape[1]
    include_padded = np.pad(include_history, ((0, 0), (0, pad_width)), mode="constant", constant_values=1.0)
    global_input = global_input * include_padded
    return binary_input, global_input


def _load_npz_file(npz_file, model_config, pos_len, allow_nonfull_mask):
    num_bin_features = configs.get_num_bin_input_features(model_config)
    num_global_features = configs.get_num_global_input_features(model_config)
    include_qvalues = model_config["version"] >= 16 and model_config["version"] < 100

    with np.load(npz_file) as npz:
        if "binaryInputNCHW" in npz:
            binary_input = npz["binaryInputNCHW"]
            stored_pos_len = int(npz["pos_len"]) if "pos_len" in npz else None
            if stored_pos_len is not None and stored_pos_len != pos_len:
                raise ValueError(f"{npz_file}: pos_len={stored_pos_len}, expected {pos_len}")
        else:
            packed = npz["binaryInputNCHWPacked"]
            binary_input = np.unpackbits(packed, axis=2)
            binary_input = binary_input[:, :, : pos_len * pos_len]
            binary_input = binary_input.reshape(binary_input.shape[0], binary_input.shape[1], pos_len, pos_len)

        global_input = npz["globalInputNC"]
        policy_targets = npz["policyTargetsNCMove"]
        global_targets = npz["globalTargetsNC"]
        score_distr = npz["scoreDistrN"]
        value_targets = npz["valueTargetsNCHW"]
        qvalue_targets = npz["qValueTargetsNCMove"] if include_qvalues else None

    if binary_input.shape[1] != num_bin_features:
        raise ValueError(f"{npz_file}: binary features={binary_input.shape[1]}, expected {num_bin_features}")
    if global_input.shape[1] != num_global_features:
        raise ValueError(f"{npz_file}: global features={global_input.shape[1]}, expected {num_global_features}")
    if not allow_nonfull_mask and not np.all(binary_input[:, 0, :, :] == 1):
        logging.warning(
            "Channel 0 mask is not all 1 in %s; continuing in fixed-board legacy mode",
            npz_file,
        )

    return {
        "binaryInputNCHW": binary_input,
        "globalInputNC": global_input,
        "policyTargetsNCMove": policy_targets,
        "globalTargetsNC": global_targets,
        "scoreDistrN": score_distr,
        "valueTargetsNCHW": value_targets,
        "qValueTargetsNCMove": qvalue_targets,
    }


def read_npz_batches(
    npz_files,
    batch_size,
    pos_len,
    model_config,
    symmetry_type="xyt",
    enable_history_matrices=True,
    seed=None,
    allow_nonfull_mask=True,
):
    """Yield float32 numpy batches compatible with the JAX training path."""
    if seed is None:
        rand = np.random.default_rng(seed=list(os.urandom(12)))
    else:
        rand = np.random.default_rng(seed=seed)
    npz_files = list(npz_files)
    if not npz_files:
        return

    if enable_history_matrices:
        h_base, h_builder = build_history_matrices_np(model_config)
    else:
        h_base, h_builder = None, None

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_load_npz_file, npz_files[0], model_config, pos_len, allow_nonfull_mask)
        for next_file in npz_files[1:] + [None]:
            arrays = future.result()
            if next_file is not None:
                future = executor.submit(_load_npz_file, next_file, model_config, pos_len, allow_nonfull_mask)

            binary_input = arrays["binaryInputNCHW"]
            num_samples = binary_input.shape[0]
            is_symmetry_all = symmetry_type == "all"
            read_batch_size = batch_size // 8 if is_symmetry_all else batch_size
            num_steps = num_samples // read_batch_size

            for n in range(num_steps):
                start = n * read_batch_size
                end = start + read_batch_size
                batch = {
                    k: v[start:end] if v is not None else None
                    for k, v in arrays.items()
                }

                if enable_history_matrices:
                    batch["binaryInputNCHW"], batch["globalInputNC"] = apply_history_matrices_np(
                        model_config,
                        batch["binaryInputNCHW"],
                        batch["globalInputNC"],
                        batch["globalTargetsNC"],
                        h_base,
                        h_builder,
                        rand,
                    )

                if symmetry_type and symmetry_type != "none":
                    if is_symmetry_all:
                        binary_parts = []
                        policy_parts = []
                        value_parts = []
                        for symm in range(8):
                            binary_parts.append(apply_symmetry_np(batch["binaryInputNCHW"], symm))
                            policy_parts.append(apply_symmetry_policy_np(batch["policyTargetsNCMove"], symm, pos_len))
                            value_parts.append(apply_symmetry_np(batch["valueTargetsNCHW"], symm))
                        batch["binaryInputNCHW"] = np.concatenate(binary_parts, axis=0)
                        batch["policyTargetsNCMove"] = np.concatenate(policy_parts, axis=0)
                        batch["valueTargetsNCHW"] = np.concatenate(value_parts, axis=0)
                        batch["globalInputNC"] = np.tile(batch["globalInputNC"], (8, 1))
                        batch["globalTargetsNC"] = np.tile(batch["globalTargetsNC"], (8, 1))
                        batch["scoreDistrN"] = np.tile(batch["scoreDistrN"], (8, 1))
                    else:
                        symm = sample_symmetry(symmetry_type, rand)
                        batch["binaryInputNCHW"] = apply_symmetry_np(batch["binaryInputNCHW"], symm)
                        batch["policyTargetsNCMove"] = apply_symmetry_policy_np(batch["policyTargetsNCMove"], symm, pos_len)
                        batch["valueTargetsNCHW"] = apply_symmetry_np(batch["valueTargetsNCHW"], symm)

                yield {
                    "binaryInputNCHW": np.ascontiguousarray(batch["binaryInputNCHW"], dtype=np.float32),
                    "globalInputNC": np.ascontiguousarray(batch["globalInputNC"], dtype=np.float32),
                    "policyTargetsNCMove": np.ascontiguousarray(batch["policyTargetsNCMove"], dtype=np.float32),
                    "globalTargetsNC": np.ascontiguousarray(batch["globalTargetsNC"], dtype=np.float32),
                    "scoreDistrN": np.ascontiguousarray(batch["scoreDistrN"], dtype=np.float32),
                    "valueTargetsNCHW": np.ascontiguousarray(batch["valueTargetsNCHW"], dtype=np.float32),
                }
