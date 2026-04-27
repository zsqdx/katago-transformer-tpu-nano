"""Data loading and symmetry augmentation for KataGo nano training."""

import logging
import os
import threading
import queue

import numpy as np
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional

import configs

_XLA_SEND_CPU_DATA_TO_DEVICE = None
_XLA_SEND_LOOKED_UP = False
_XLA_SEND_FAILED = False


def prefetch_generator(gen, prefetch_batches):
    """Prefetch batches in a background thread to overlap data prep with GPU compute."""
    if prefetch_batches <= 0:
        yield from gen
        return

    q = queue.Queue(maxsize=prefetch_batches)
    _sentinel = object()

    def _producer():
        try:
            for item in gen:
                q.put(item)
        except Exception as e:
            q.put(e)
        finally:
            q.put(_sentinel)

    t = threading.Thread(target=_producer, daemon=True)
    t.start()

    while True:
        item = q.get()
        if item is _sentinel:
            break
        if isinstance(item, Exception):
            raise item
        yield item

    t.join()


def _float_cpu_tensor(array):
    tensor = torch.from_numpy(array)
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    return tensor


def _get_xla_cpu_sender():
    global _XLA_SEND_CPU_DATA_TO_DEVICE, _XLA_SEND_LOOKED_UP
    if not _XLA_SEND_LOOKED_UP:
        _XLA_SEND_LOOKED_UP = True
        try:
            import torch_xla.core.xla_model as xm
            _XLA_SEND_CPU_DATA_TO_DEVICE = getattr(xm, "send_cpu_data_to_device", None)
        except Exception:
            _XLA_SEND_CPU_DATA_TO_DEVICE = None
    return _XLA_SEND_CPU_DATA_TO_DEVICE


def _send_cpu_batch_to_xla(batch, device):
    global _XLA_SEND_FAILED
    sender = None if _XLA_SEND_FAILED else _get_xla_cpu_sender()
    if sender is not None:
        try:
            return sender(batch, device)
        except Exception as exc:
            if not _XLA_SEND_FAILED:
                logging.warning("XLA batched CPU transfer failed, falling back to per-tensor .to(device): %s", exc)
            _XLA_SEND_FAILED = True
    return {k: v.to(device) for k, v in batch.items()}


def read_npz_training_data(
    npz_files,
    batch_size: int,
    world_size: int,
    rank: int,
    pos_len: int,
    device,
    symmetry_type: str,
    include_meta: bool,
    enable_history_matrices: bool,
    model_config: configs.ModelConfig,
    use_pin_memory: bool = False,
    seed=None,
    varlen: bool = False,
    allow_nonfull_mask: bool = False,
    xla_batched_transfer: bool = True,
):
    if seed is not None:
        rand = np.random.default_rng(seed=seed)
    else:
        rand = np.random.default_rng(seed=list(os.urandom(12)))
    num_bin_features = configs.get_num_bin_input_features(model_config)
    num_global_features = configs.get_num_global_input_features(model_config)
    prepare_on_host = device.type == "xla"
    if enable_history_matrices:
        if prepare_on_host:
            h_base, h_builder = None, None
            (h_base_np, h_builder_np) = build_history_matrices_np(model_config)
        else:
            (h_base, h_builder) = build_history_matrices(model_config, device)
            h_base_np, h_builder_np = None, None

    include_qvalues = model_config["version"] >= 16 and model_config["version"] < 100

    def load_npz_file(npz_file):
        with np.load(npz_file) as npz:
            is_preprocessed = "binaryInputNCHW" in npz

            if is_preprocessed:
                # Preprocessed format: already unpacked uint8 NCHW
                binaryInputNCHW = npz["binaryInputNCHW"]
                stored_pos_len = int(npz["pos_len"]) if "pos_len" in npz else None
                if stored_pos_len is not None:
                    assert stored_pos_len == pos_len, \
                        f"pos_len mismatch: file has {stored_pos_len}, expected {pos_len}"
                assert binaryInputNCHW.shape[2] == pos_len and binaryInputNCHW.shape[3] == pos_len, \
                    f"Spatial dims {binaryInputNCHW.shape[2:]}, expected ({pos_len}, {pos_len})"
            else:
                # Original format: packed bits, need unpackbits
                binaryInputNCHWPacked = npz["binaryInputNCHWPacked"]
                binaryInputNCHW = np.unpackbits(binaryInputNCHWPacked, axis=2)
                assert len(binaryInputNCHW.shape) == 3
                assert binaryInputNCHW.shape[2] == ((pos_len * pos_len + 7) // 8) * 8
                binaryInputNCHW = binaryInputNCHW[:, :, :pos_len * pos_len]
                binaryInputNCHW = np.reshape(binaryInputNCHW, (
                    binaryInputNCHW.shape[0], binaryInputNCHW.shape[1], pos_len, pos_len
                ))  # uint8, no float32 conversion here

            globalInputNC = npz["globalInputNC"]
            policyTargetsNCMove = npz["policyTargetsNCMove"]
            globalTargetsNC = npz["globalTargetsNC"]
            scoreDistrN = npz["scoreDistrN"]
            valueTargetsNCHW = npz["valueTargetsNCHW"]
            if include_meta:
                metadataInputNC = npz["metadataInputNC"]
            else:
                metadataInputNC = None
            if include_qvalues:
                qValueTargetsNCMove = npz["qValueTargetsNCMove"]
            else:
                qValueTargetsNCMove = None

        assert binaryInputNCHW.shape[1] == num_bin_features
        assert globalInputNC.shape[1] == num_global_features
        # Channel 0 is the on-board mask. Some older real datasets have legacy
        # mask contents even for fixed-size 19x19 data, so allow compatibility.
        if not varlen and not allow_nonfull_mask and not np.all(binaryInputNCHW[:, 0, :, :] == 1):
            logging.warning(
                "Channel 0 (on-board mask) is not all 1 in %s. Continuing in legacy "
                "fixed-board mode; use --allow-nonfull-mask to silence this warning.",
                npz_file,
            )
        return (npz_file, binaryInputNCHW, globalInputNC, policyTargetsNCMove, globalTargetsNC, scoreDistrN, valueTargetsNCHW, metadataInputNC, qValueTargetsNCMove)

    if not npz_files:
        return

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(load_npz_file, npz_files[0])

        for next_file in (npz_files[1:] + [None]):
            (npz_file, binaryInputNCHW, globalInputNC, policyTargetsNCMove, globalTargetsNC, scoreDistrN, valueTargetsNCHW, metadataInputNC, qValueTargetsNCMove) = future.result()

            num_samples = binaryInputNCHW.shape[0]
            is_symmetry_all = (symmetry_type == "all")
            if is_symmetry_all:
                assert batch_size % 8 == 0, "batch_size must be divisible by 8 when symmetry_type='all'"
                read_batch_size = batch_size // 8
            else:
                read_batch_size = batch_size
            num_whole_steps = num_samples // (read_batch_size * world_size)

            if next_file is not None:
                future = executor.submit(load_npz_file, next_file)

            for n in range(num_whole_steps):
                start = (n * world_size + rank) * read_batch_size
                end = start + read_batch_size

                batch_binaryInputNCHW_np = binaryInputNCHW[start:end]
                batch_globalInputNC_np = globalInputNC[start:end]
                batch_policyTargetsNCMove_np = policyTargetsNCMove[start:end]
                batch_globalTargetsNC_np = globalTargetsNC[start:end]
                batch_scoreDistrN_np = scoreDistrN[start:end]
                batch_valueTargetsNCHW_np = valueTargetsNCHW[start:end]
                if include_meta:
                    batch_metadataInputNC_np = metadataInputNC[start:end]
                if include_qvalues:
                    batch_qValueTargetsNCMove_np = qValueTargetsNCMove[start:end]

                if prepare_on_host:
                    if enable_history_matrices:
                        (batch_binaryInputNCHW_np, batch_globalInputNC_np) = apply_history_matrices_np(
                            model_config, batch_binaryInputNCHW_np, batch_globalInputNC_np,
                            batch_globalTargetsNC_np, h_base_np, h_builder_np, rand,
                        )

                    if symmetry_type is not None and symmetry_type != "" and symmetry_type != "none":
                        if is_symmetry_all:
                            sym_binary_parts = []
                            sym_policy_parts = []
                            sym_value_parts = []
                            if include_qvalues:
                                sym_qvalue_parts = []
                            for symm in range(8):
                                sym_binary_parts.append(apply_symmetry_np(batch_binaryInputNCHW_np, symm))
                                sym_policy_parts.append(apply_symmetry_policy_np(batch_policyTargetsNCMove_np, symm, pos_len))
                                sym_value_parts.append(apply_symmetry_np(batch_valueTargetsNCHW_np, symm))
                                if include_qvalues:
                                    sym_qvalue_parts.append(apply_symmetry_policy_np(batch_qValueTargetsNCMove_np, symm, pos_len))
                            batch_binaryInputNCHW_np = np.concatenate(sym_binary_parts, axis=0)
                            batch_policyTargetsNCMove_np = np.concatenate(sym_policy_parts, axis=0)
                            batch_valueTargetsNCHW_np = np.concatenate(sym_value_parts, axis=0)
                            if include_qvalues:
                                batch_qValueTargetsNCMove_np = np.concatenate(sym_qvalue_parts, axis=0)
                            batch_globalInputNC_np = np.tile(batch_globalInputNC_np, (8, 1))
                            batch_globalTargetsNC_np = np.tile(batch_globalTargetsNC_np, (8, 1))
                            batch_scoreDistrN_np = np.tile(batch_scoreDistrN_np, (8, 1))
                            if include_meta:
                                batch_metadataInputNC_np = np.tile(batch_metadataInputNC_np, (8, 1))
                        else:
                            symm = sample_symmetry(symmetry_type, rand)
                            batch_binaryInputNCHW_np = apply_symmetry_np(batch_binaryInputNCHW_np, symm)
                            batch_policyTargetsNCMove_np = apply_symmetry_policy_np(batch_policyTargetsNCMove_np, symm, pos_len)
                            batch_valueTargetsNCHW_np = apply_symmetry_np(batch_valueTargetsNCHW_np, symm)
                            if include_qvalues:
                                batch_qValueTargetsNCMove_np = apply_symmetry_policy_np(batch_qValueTargetsNCMove_np, symm, pos_len)

                    batch_binaryInputNCHW_np = np.ascontiguousarray(batch_binaryInputNCHW_np)
                    batch_globalInputNC_np = np.ascontiguousarray(batch_globalInputNC_np)
                    batch_policyTargetsNCMove_np = np.ascontiguousarray(batch_policyTargetsNCMove_np)
                    batch_globalTargetsNC_np = np.ascontiguousarray(batch_globalTargetsNC_np)
                    batch_scoreDistrN_np = np.ascontiguousarray(batch_scoreDistrN_np)
                    batch_valueTargetsNCHW_np = np.ascontiguousarray(batch_valueTargetsNCHW_np)
                    if include_meta:
                        batch_metadataInputNC_np = np.ascontiguousarray(batch_metadataInputNC_np)
                    if include_qvalues:
                        batch_qValueTargetsNCMove_np = np.ascontiguousarray(batch_qValueTargetsNCMove_np)

                if prepare_on_host and xla_batched_transfer:
                    batch = dict(
                        binaryInputNCHW=_float_cpu_tensor(batch_binaryInputNCHW_np),
                        globalInputNC=_float_cpu_tensor(batch_globalInputNC_np),
                        policyTargetsNCMove=_float_cpu_tensor(batch_policyTargetsNCMove_np),
                        globalTargetsNC=_float_cpu_tensor(batch_globalTargetsNC_np),
                        scoreDistrN=_float_cpu_tensor(batch_scoreDistrN_np),
                        valueTargetsNCHW=_float_cpu_tensor(batch_valueTargetsNCHW_np),
                    )
                    if include_meta:
                        batch["metadataInputNC"] = _float_cpu_tensor(batch_metadataInputNC_np)
                    if include_qvalues:
                        batch["qValueTargetsNCMove"] = _float_cpu_tensor(batch_qValueTargetsNCMove_np)
                    yield _send_cpu_batch_to_xla(batch, device)
                    continue

                if use_pin_memory:
                    batch_binaryInputNCHW = torch.from_numpy(batch_binaryInputNCHW_np).pin_memory().to(device, non_blocking=True).float()
                    batch_globalInputNC = torch.from_numpy(batch_globalInputNC_np).pin_memory().to(device, non_blocking=True).float()
                    batch_policyTargetsNCMove = torch.from_numpy(batch_policyTargetsNCMove_np).pin_memory().to(device, non_blocking=True).float()
                    batch_globalTargetsNC = torch.from_numpy(batch_globalTargetsNC_np).pin_memory().to(device, non_blocking=True).float()
                    batch_scoreDistrN = torch.from_numpy(batch_scoreDistrN_np).pin_memory().to(device, non_blocking=True).float()
                    batch_valueTargetsNCHW = torch.from_numpy(batch_valueTargetsNCHW_np).pin_memory().to(device, non_blocking=True).float()
                    if include_meta:
                        batch_metadataInputNC = torch.from_numpy(batch_metadataInputNC_np).pin_memory().to(device, non_blocking=True).float()
                    if include_qvalues:
                        batch_qValueTargetsNCMove = torch.from_numpy(batch_qValueTargetsNCMove_np).pin_memory().to(device, non_blocking=True).float()
                else:
                    batch_binaryInputNCHW = torch.from_numpy(batch_binaryInputNCHW_np).to(device).float()
                    batch_globalInputNC = torch.from_numpy(batch_globalInputNC_np).to(device).float()
                    batch_policyTargetsNCMove = torch.from_numpy(batch_policyTargetsNCMove_np).to(device).float()
                    batch_globalTargetsNC = torch.from_numpy(batch_globalTargetsNC_np).to(device).float()
                    batch_scoreDistrN = torch.from_numpy(batch_scoreDistrN_np).to(device).float()
                    batch_valueTargetsNCHW = torch.from_numpy(batch_valueTargetsNCHW_np).to(device).float()
                    if include_meta:
                        batch_metadataInputNC = torch.from_numpy(batch_metadataInputNC_np).to(device).float()
                    if include_qvalues:
                        batch_qValueTargetsNCMove = torch.from_numpy(batch_qValueTargetsNCMove_np).to(device).float()

                if not prepare_on_host:
                    if enable_history_matrices:
                        (batch_binaryInputNCHW, batch_globalInputNC) = apply_history_matrices(
                            model_config, batch_binaryInputNCHW, batch_globalInputNC, batch_globalTargetsNC, h_base, h_builder
                        )

                    if symmetry_type is not None and symmetry_type != "" and symmetry_type != "none":
                        if is_symmetry_all:
                            # Apply all 8 symmetries to each sample, expanding read_batch_size -> batch_size
                            sym_binary_parts = []
                            sym_policy_parts = []
                            sym_value_parts = []
                            if include_qvalues:
                                sym_qvalue_parts = []
                            for symm in range(8):
                                sym_binary_parts.append(apply_symmetry(batch_binaryInputNCHW, symm))
                                sym_policy_parts.append(apply_symmetry_policy(batch_policyTargetsNCMove, symm, pos_len))
                                sym_value_parts.append(apply_symmetry(batch_valueTargetsNCHW, symm))
                                if include_qvalues:
                                    sym_qvalue_parts.append(apply_symmetry_policy(batch_qValueTargetsNCMove, symm, pos_len))
                            batch_binaryInputNCHW = torch.cat(sym_binary_parts, dim=0)
                            batch_policyTargetsNCMove = torch.cat(sym_policy_parts, dim=0)
                            batch_valueTargetsNCHW = torch.cat(sym_value_parts, dim=0)
                            if include_qvalues:
                                batch_qValueTargetsNCMove = torch.cat(sym_qvalue_parts, dim=0)
                            # Non-spatial tensors: repeat 8 times to match
                            batch_globalInputNC = batch_globalInputNC.repeat(8, 1)
                            batch_globalTargetsNC = batch_globalTargetsNC.repeat(8, 1)
                            batch_scoreDistrN = batch_scoreDistrN.repeat(8, 1)
                            if include_meta:
                                batch_metadataInputNC = batch_metadataInputNC.repeat(8, 1)
                        else:
                            symm = sample_symmetry(symmetry_type, rand)

                            batch_binaryInputNCHW = apply_symmetry(batch_binaryInputNCHW, symm)
                            batch_policyTargetsNCMove = apply_symmetry_policy(batch_policyTargetsNCMove, symm, pos_len)
                            batch_valueTargetsNCHW = apply_symmetry(batch_valueTargetsNCHW, symm)
                            if include_qvalues:
                                batch_qValueTargetsNCMove = apply_symmetry_policy(batch_qValueTargetsNCMove, symm, pos_len)

                batch_binaryInputNCHW = batch_binaryInputNCHW.contiguous()
                batch_policyTargetsNCMove = batch_policyTargetsNCMove.contiguous()
                batch_valueTargetsNCHW = batch_valueTargetsNCHW.contiguous()
                if include_qvalues:
                    batch_qValueTargetsNCMove = batch_qValueTargetsNCMove.contiguous()

                batch = dict(
                    binaryInputNCHW=batch_binaryInputNCHW,
                    globalInputNC=batch_globalInputNC,
                    policyTargetsNCMove=batch_policyTargetsNCMove,
                    globalTargetsNC=batch_globalTargetsNC,
                    scoreDistrN=batch_scoreDistrN,
                    valueTargetsNCHW=batch_valueTargetsNCHW,
                )
                if include_meta:
                    batch["metadataInputNC"] = batch_metadataInputNC
                if include_qvalues:
                    batch["qValueTargetsNCMove"] = batch_qValueTargetsNCMove

                yield batch


def apply_symmetry_policy(tensor, symm, pos_len):
    """Same as apply_symmetry but also handles the pass index."""
    batch_size = tensor.shape[0]
    channels = tensor.shape[1]
    tensor_without_pass = tensor[:, :, :-1].view((batch_size, channels, pos_len, pos_len))
    tensor_transformed = apply_symmetry(tensor_without_pass, symm)
    return torch.cat((
        tensor_transformed.reshape(batch_size, channels, pos_len * pos_len),
        tensor[:, :, -1:]
    ), dim=2)


def sample_symmetry(symmetry_type, rand):
    if symmetry_type == "xyt":
        allowed_symms = [0, 1, 2, 3, 4, 5, 6, 7]
    elif symmetry_type == "x":
        allowed_symms = [0, 5]
    elif symmetry_type == "xy":
        allowed_symms = [0, 2, 5, 7]
    elif symmetry_type == "x+y":
        allowed_symms = [0, 2]
    elif symmetry_type == "t":
        allowed_symms = [0, 4]
    else:
        assert False, f"Unknown data symmetry type {symmetry_type}"
    return allowed_symms[int(rand.integers(0, len(allowed_symms)))]


def apply_symmetry(tensor, symm):
    """
    Apply a symmetry operation to the given tensor.

    Args:
        tensor (torch.Tensor): Tensor to be rotated. (..., W, W)
        symm (int):
            0, 1, 2, 3: Rotation by symm * pi / 2 radians.
            4, 5, 6, 7: Mirror symmetry on top of rotation.
    """
    assert tensor.shape[-1] == tensor.shape[-2]

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


def apply_symmetry_policy_np(array, symm, pos_len):
    """Numpy version used for XLA host-side data augmentation."""
    batch_size = array.shape[0]
    channels = array.shape[1]
    array_without_pass = array[:, :, :-1].reshape((batch_size, channels, pos_len, pos_len))
    array_transformed = apply_symmetry_np(array_without_pass, symm)
    return np.concatenate((
        array_transformed.reshape(batch_size, channels, pos_len * pos_len),
        array[:, :, -1:],
    ), axis=2)


def apply_symmetry_np(array, symm):
    """Numpy version of apply_symmetry. Returns a view when possible."""
    assert array.shape[-1] == array.shape[-2]

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
    assert False, f"Unknown symmetry {symm}"


def build_history_matrices(model_config: configs.ModelConfig, device):
    num_bin_features = configs.get_num_bin_input_features(model_config)
    assert num_bin_features == 22, "Currently this code is hardcoded for this many features"

    h_base = torch.diag(
        torch.tensor(
            [
                1.0,  # 0
                1.0,  # 1
                1.0,  # 2
                1.0,  # 3
                1.0,  # 4
                1.0,  # 5
                1.0,  # 6
                1.0,  # 7
                1.0,  # 8
                0.0,  # 9   Location of move 1 turn ago
                0.0,  # 10  Location of move 2 turns ago
                0.0,  # 11  Location of move 3 turns ago
                0.0,  # 12  Location of move 4 turns ago
                0.0,  # 13  Location of move 5 turns ago
                1.0,  # 14  Ladder-threatened stone
                0.0,  # 15  Ladder-threatened stone, 1 turn ago
                0.0,  # 16  Ladder-threatened stone, 2 turns ago
                1.0,  # 17
                1.0,  # 18
                1.0,  # 19
                1.0,  # 20
                1.0,  # 21
            ],
            device=device,
            requires_grad=False,
        )
    )
    h_base[14, 15] = 1.0
    h_base[14, 16] = 1.0

    h0 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    h0[9, 9] = 1.0
    h0[14, 15] = -1.0
    h0[14, 16] = -1.0
    h0[15, 15] = 1.0
    h0[15, 16] = 1.0

    h1 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    h1[10, 10] = 1.0
    h1[15, 16] = -1.0
    h1[16, 16] = 1.0

    h2 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    h2[11, 11] = 1.0

    h3 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    h3[12, 12] = 1.0

    h4 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    h4[13, 13] = 1.0

    h_base = h_base.reshape((1, num_bin_features, num_bin_features))
    h_builder = torch.stack((h0, h1, h2, h3, h4), dim=0)

    return (h_base, h_builder)


def build_history_matrices_np(model_config: configs.ModelConfig):
    num_bin_features = configs.get_num_bin_input_features(model_config)
    assert num_bin_features == 22, "Currently this code is hardcoded for this many features"

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

    h_base = h_base.reshape((1, num_bin_features, num_bin_features))
    h_builder = np.stack((h0, h1, h2, h3, h4), axis=0)
    return (h_base, h_builder)


def apply_history_matrices(model_config, batch_binaryInputNCHW, batch_globalInputNC, batch_globalTargetsNC, h_base, h_builder):
    num_global_features = configs.get_num_global_input_features(model_config)
    # Generate random on CPU to avoid conflict with torch.compile CUDA graph capture
    ref = batch_globalTargetsNC[:, 36:41]
    should_stop_history = (torch.rand(ref.shape, dtype=ref.dtype, device="cpu") >= 0.98).to(ref.device)
    include_history = (torch.cumsum(should_stop_history, axis=1, dtype=torch.float32) <= 0.1).to(torch.float32)

    h_matrix = h_base + torch.einsum("bi,ijk->bjk", include_history, h_builder)

    batch_binaryInputNCHW = torch.einsum("bijk,bil->bljk", batch_binaryInputNCHW, h_matrix)

    batch_globalInputNC = batch_globalInputNC * torch.nn.functional.pad(
        include_history, ((0, num_global_features - include_history.shape[1])), value=1.0
    )
    return batch_binaryInputNCHW, batch_globalInputNC


def apply_history_matrices_np(
    model_config,
    batch_binaryInputNCHW,
    batch_globalInputNC,
    batch_globalTargetsNC,
    h_base,
    h_builder,
    rand,
):
    num_global_features = configs.get_num_global_input_features(model_config)
    ref = batch_globalTargetsNC[:, 36:41]
    should_stop_history = (rand.random(ref.shape) >= 0.98)
    include_history = (np.cumsum(should_stop_history, axis=1, dtype=np.float32) <= 0.1).astype(np.float32)

    h_matrix = h_base + np.einsum("bi,ijk->bjk", include_history, h_builder)
    batch_binaryInputNCHW = np.einsum("bijk,bil->bljk", batch_binaryInputNCHW, h_matrix).astype(np.float32, copy=False)

    pad_width = num_global_features - include_history.shape[1]
    include_padded = np.pad(include_history, ((0, 0), (0, pad_width)), mode="constant", constant_values=1.0)
    batch_globalInputNC = batch_globalInputNC * include_padded
    return batch_binaryInputNCHW, batch_globalInputNC
