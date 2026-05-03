"""JAX TPU training prototype for KataGo transformer nano.

This path is intentionally separate from train.py so that PyTorch/XLA remains a
known-good baseline while we test whether a direct JAX/XLA program improves TPU
utilization.
"""

import argparse
import json
import logging
import math
import os
import pickle
import time

import numpy as np

import configs
import jax_data

jax_losses = None
jax_model = None


def _tree_map(fn, *trees):
    ref = trees[0]
    if isinstance(ref, dict):
        return {k: _tree_map(fn, *(t[k] for t in trees)) for k in ref}
    if isinstance(ref, list):
        return [_tree_map(fn, *(t[i] for t in trees)) for i in range(len(ref))]
    if isinstance(ref, tuple):
        return tuple(_tree_map(fn, *(t[i] for t in trees)) for i in range(len(ref)))
    return fn(*trees)


def _tree_map_with_path(fn, tree, path=()):
    if isinstance(tree, dict):
        return {k: _tree_map_with_path(fn, v, path + (k,)) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_tree_map_with_path(fn, v, path + (str(i),)) for i, v in enumerate(tree)]
    if isinstance(tree, tuple):
        return tuple(_tree_map_with_path(fn, v, path + (str(i),)) for i, v in enumerate(tree))
    return fn(path, tree)


def _tree_sum_squares(tree):
    import jax.numpy as jnp

    leaves = []

    def collect(x):
        leaves.append(jnp.sum(jnp.square(x.astype(jnp.float32))))
        return x

    _tree_map(lambda x: collect(x), tree)
    return sum(leaves)


def _decay_mask_for_path(path):
    leaf = path[-1] if path else ""
    if leaf != "w":
        return False
    return not any("norm" in part for part in path)


def init_adam_state(params, dtype=None):
    import jax.numpy as jnp

    state_dtype = jnp.float32 if dtype is None else dtype
    return {
        "m": _tree_map(lambda p: jnp.zeros(p.shape, dtype=state_dtype), params),
        "v": _tree_map(lambda p: jnp.zeros(p.shape, dtype=state_dtype), params),
    }


def adamw_update(
    params,
    grads,
    state,
    step,
    lr,
    wd,
    state_dtype=None,
    param_dtype=None,
    update_dtype=None,
    beta1=0.9,
    beta2=0.95,
    eps=1e-8,
):
    import jax.numpy as jnp

    state_dtype = jnp.float32 if state_dtype is None else state_dtype
    param_dtype = jnp.float32 if param_dtype is None else param_dtype
    update_dtype = jnp.float32 if update_dtype is None else update_dtype
    beta1_v = jnp.asarray(beta1, dtype=update_dtype)
    beta2_v = jnp.asarray(beta2, dtype=update_dtype)
    one_v = jnp.asarray(1.0, dtype=update_dtype)
    new_m = _tree_map(
        lambda m, g: beta1_v * m.astype(update_dtype) + (one_v - beta1_v) * g.astype(update_dtype),
        state["m"],
        grads,
    )
    new_v = _tree_map(
        lambda v, g: beta2_v * v.astype(update_dtype) + (one_v - beta2_v) * jnp.square(g.astype(update_dtype)),
        state["v"],
        grads,
    )
    bc1 = jnp.asarray(1.0 - beta1 ** step, dtype=update_dtype)
    bc2 = jnp.asarray(1.0 - beta2 ** step, dtype=update_dtype)
    eps_v = jnp.asarray(eps, dtype=update_dtype)
    lr_v = jnp.asarray(lr, dtype=update_dtype)
    wd_v = jnp.asarray(wd, dtype=update_dtype)

    def update_leaf(path, p, m, v):
        m_hat = m / bc1
        v_hat = v / bc2
        p_update = m_hat / (jnp.sqrt(v_hat) + eps_v)
        p_new = p.astype(update_dtype) - lr_v * p_update
        if _decay_mask_for_path(path):
            p_new = p_new - lr_v * wd_v * p.astype(update_dtype)
        return p_new

    new_params = _tree_map_with_path(
        lambda path, p: update_leaf(
            path,
            p,
            _get_path(new_m, path),
            _get_path(new_v, path),
        ),
        params,
    )
    return _tree_map(lambda p: p.astype(param_dtype), new_params), {
        "m": _tree_map(lambda m: m.astype(state_dtype), new_m),
        "v": _tree_map(lambda v: v.astype(state_dtype), new_v),
    }


def sgd_update(
    params,
    grads,
    lr,
    wd,
    param_dtype=None,
    update_dtype=None,
):
    import jax.numpy as jnp

    param_dtype = jnp.float32 if param_dtype is None else param_dtype
    update_dtype = jnp.float32 if update_dtype is None else update_dtype
    lr_v = jnp.asarray(lr, dtype=update_dtype)
    wd_v = jnp.asarray(wd, dtype=update_dtype)

    def update_leaf(path, p, g):
        p_new = p.astype(update_dtype) - lr_v * g.astype(update_dtype)
        if _decay_mask_for_path(path):
            p_new = p_new - lr_v * wd_v * p.astype(update_dtype)
        return p_new.astype(param_dtype)

    return _tree_map_with_path(
        lambda path, p: update_leaf(path, p, _get_path(grads, path)),
        params,
    )


def _get_path(tree, path):
    cur = tree
    for part in path:
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, tuple):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


_POLAR_EXPRESS_COEFFS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


def _muon_target_matches_path(path, target):
    target = str(target).lower()
    if target == "all":
        return True
    if target == "none":
        return False
    if target == "attn":
        return any(part in ("q_proj", "k_proj", "v_proj", "qkv_proj", "out_proj") for part in path)
    if target == "ffn":
        return any(part in ("ffn_w1", "ffn_wgate", "ffn_w2", "ffn_upgate") for part in path)
    if target == "square":
        return True
    raise ValueError(f"Unknown Muon target: {target}")


def _is_muon_path(path, leaf, target="all"):
    if not path or path[0] != "blocks" or path[-1] != "w":
        return False
    if getattr(leaf, "ndim", 0) < 2:
        return False
    if str(target).lower() == "square" and leaf.shape[-2] != leaf.shape[-1]:
        return False
    return _muon_target_matches_path(path, target)


def _is_qkv_projection_path(path):
    return any(part in ("q_proj", "k_proj", "v_proj", "qkv_proj") for part in path)


def init_muon_adamw_state(params, dtype=None, muon_target="all"):
    import jax.numpy as jnp

    state_dtype = jnp.float32 if dtype is None else dtype

    def init_v(path, p):
        if _is_muon_path(path, p, muon_target):
            return jnp.zeros((), dtype=state_dtype)
        return jnp.zeros(p.shape, dtype=state_dtype)

    return {
        # AdamW first moment for non-Muon leaves; Muon momentum for block matrix leaves.
        "m": _tree_map(lambda p: jnp.zeros(p.shape, dtype=state_dtype), params),
        # AdamW second moment is unnecessary for Muon leaves, so store scalar sentinels there.
        "v": _tree_map_with_path(init_v, params),
    }


def _muon_split_for_path(path, tensor, num_heads, row_split_size=0):
    if tensor.ndim == 4:
        tensor = tensor.reshape((tensor.shape[0], -1))
    if _is_qkv_projection_path(path) and num_heads > 0:
        chunks = 3 * num_heads if any(part == "qkv_proj" for part in path) else num_heads
        if tensor.shape[-2] % chunks != 0:
            raise ValueError(f"Cannot head-split Muon param {'/'.join(path)} with shape {tensor.shape}")
        return tensor.reshape((-1, tensor.shape[-2] // chunks, tensor.shape[-1]))
    if row_split_size > 0 and tensor.shape[-2] > row_split_size and tensor.shape[-2] % row_split_size == 0:
        return tensor.reshape((-1, row_split_size, tensor.shape[-1]))
    return tensor


def _leaf_paths(tree, path=()):
    if isinstance(tree, dict):
        paths = []
        for k, v in tree.items():
            paths.extend(_leaf_paths(v, path + (k,)))
        return paths
    if isinstance(tree, list):
        paths = []
        for i, v in enumerate(tree):
            paths.extend(_leaf_paths(v, path + (str(i),)))
        return paths
    if isinstance(tree, tuple):
        paths = []
        for i, v in enumerate(tree):
            paths.extend(_leaf_paths(v, path + (str(i),)))
        return paths
    return [path]


def _tree_from_leaf_values(template, values, index=None, path=()):
    if isinstance(template, dict):
        return {
            k: _tree_from_leaf_values(v, values, index=index, path=path + (k,))
            for k, v in template.items()
        }
    if isinstance(template, list):
        return [
            _tree_from_leaf_values(v, values, index=index, path=path + (str(i),))
            for i, v in enumerate(template)
        ]
    if isinstance(template, tuple):
        return tuple(
            _tree_from_leaf_values(v, values, index=index, path=path + (str(i),))
            for i, v in enumerate(template)
        )
    value = values[path]
    if index is None:
        return value
    return value[index]


def polar_express_jax(g, steps=5):
    import jax.numpy as jnp

    x = g.astype(jnp.bfloat16)
    transposed = g.shape[-2] > g.shape[-1]
    if transposed:
        x = jnp.swapaxes(x, -1, -2)

    norm = jnp.sqrt(jnp.sum(jnp.square(x.astype(jnp.float32)), axis=(-2, -1), keepdims=True))
    x = (x / (norm.astype(jnp.float32) * 1.02 + 1e-6)).astype(jnp.bfloat16)

    for a, b, c in _POLAR_EXPRESS_COEFFS[:steps]:
        a_v = jnp.asarray(a, dtype=jnp.bfloat16)
        b_v = jnp.asarray(b, dtype=jnp.bfloat16)
        c_v = jnp.asarray(c, dtype=jnp.bfloat16)
        xx_t = jnp.swapaxes(x, -1, -2)
        a_mat = (x @ xx_t).astype(jnp.bfloat16)
        b_mat = (b_v * a_mat + c_v * (a_mat @ a_mat)).astype(jnp.bfloat16)
        x = (a_v * x + b_mat @ x).astype(jnp.bfloat16)

    if transposed:
        x = jnp.swapaxes(x, -1, -2)
    return x


def muon_adamw_update(
    params,
    grads,
    state,
    step,
    lr,
    wd,
    config,
    state_dtype=None,
    param_dtype=None,
    update_dtype=None,
    beta1=0.9,
    beta2=0.95,
    eps=1e-8,
    muon_lr_multiplier=0.2,
    muon_momentum=0.95,
    muon_row_split_size=0,
    muon_target="all",
    muon_polar_steps=5,
    muon_group_blocks=True,
):
    import jax.numpy as jnp

    if str(muon_target).lower() == "none":
        return adamw_update(
            params,
            grads,
            state,
            step,
            lr,
            wd,
            state_dtype=state_dtype,
            param_dtype=param_dtype,
            update_dtype=update_dtype,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
        )

    state_dtype = jnp.float32 if state_dtype is None else state_dtype
    param_dtype = jnp.float32 if param_dtype is None else param_dtype
    update_dtype = jnp.float32 if update_dtype is None else update_dtype
    beta1_v = jnp.asarray(beta1, dtype=update_dtype)
    beta2_v = jnp.asarray(beta2, dtype=update_dtype)
    one_v = jnp.asarray(1.0, dtype=update_dtype)
    bc1 = jnp.asarray(1.0 - beta1 ** step, dtype=update_dtype)
    bc2 = jnp.asarray(1.0 - beta2 ** step, dtype=update_dtype)
    eps_v = jnp.asarray(eps, dtype=update_dtype)
    lr_v = jnp.asarray(lr, dtype=update_dtype)
    wd_v = jnp.asarray(wd, dtype=update_dtype)
    muon_lr_v = lr_v * jnp.asarray(muon_lr_multiplier, dtype=update_dtype)
    muon_momentum_v = jnp.asarray(muon_momentum, dtype=update_dtype)
    num_heads = int(config["num_heads"])

    def adam_update_leaf(path, p, g, m, v):
        new_m_leaf = beta1_v * m.astype(update_dtype) + (one_v - beta1_v) * g.astype(update_dtype)
        new_v_leaf = beta2_v * v.astype(update_dtype) + (one_v - beta2_v) * jnp.square(g.astype(update_dtype))
        m_hat = new_m_leaf / bc1
        v_hat = new_v_leaf / bc2
        p_update = m_hat / (jnp.sqrt(v_hat) + eps_v)
        p_new = p.astype(update_dtype) - lr_v * p_update
        if _decay_mask_for_path(path):
            p_new = p_new - lr_v * wd_v * p.astype(update_dtype)
        return p_new.astype(param_dtype), new_m_leaf.astype(state_dtype), new_v_leaf.astype(state_dtype)

    def muon_update_leaf(path, p, g, m, v):
        new_m_leaf = muon_momentum_v * m.astype(update_dtype) + g.astype(update_dtype)
        split_update = _muon_split_for_path(path, new_m_leaf, num_heads, muon_row_split_size)
        split_update = polar_express_jax(split_update, steps=muon_polar_steps)
        scale = math.sqrt(max(split_update.shape[-2], split_update.shape[-1]))
        update = (split_update * jnp.asarray(scale, dtype=split_update.dtype)).reshape(p.shape)
        p_new = p.astype(update_dtype) * (one_v - lr_v * wd_v)
        p_new = p_new - muon_lr_v * update.astype(update_dtype)
        return p_new.astype(param_dtype), new_m_leaf.astype(state_dtype), v

    def update_leaf(path, p):
        g = _get_path(grads, path)
        m = _get_path(state["m"], path)
        v = _get_path(state["v"], path)
        if _is_muon_path(path, p, muon_target):
            return muon_update_leaf(path, p, g, m, v)
        return adam_update_leaf(path, p, g, m, v)

    def extract_updated(tree, idx):
        if isinstance(tree, dict):
            return {k: extract_updated(v, idx) for k, v in tree.items()}
        if isinstance(tree, list):
            return [extract_updated(v, idx) for v in tree]
        if isinstance(tree, tuple) and len(tree) == 3:
            return tree[idx]
        raise TypeError(f"Unexpected Muon update tree leaf: {type(tree)}")

    if muon_group_blocks and isinstance(params.get("blocks"), list):
        new_params = {}
        new_m = {}
        new_v = {}
        for key, subtree in params.items():
            if key == "blocks":
                continue
            updated_subtree = _tree_map_with_path(lambda path, p: update_leaf((key,) + path, p), subtree)
            new_params[key] = extract_updated(updated_subtree, 0)
            new_m[key] = extract_updated(updated_subtree, 1)
            new_v[key] = extract_updated(updated_subtree, 2)

        block_template = params["blocks"][0]
        block_param_values = {}
        block_m_values = {}
        block_v_values = {}
        for leaf_path in _leaf_paths(block_template):
            full_path = ("blocks",) + leaf_path
            p_stack = jnp.stack([_get_path(block, leaf_path) for block in params["blocks"]], axis=0)
            g_stack = jnp.stack([_get_path(block, leaf_path) for block in grads["blocks"]], axis=0)
            m_stack = jnp.stack([_get_path(block, leaf_path) for block in state["m"]["blocks"]], axis=0)
            if _is_muon_path(full_path, p_stack, muon_target):
                v_values = [_get_path(block, leaf_path) for block in state["v"]["blocks"]]
                p_new, m_new, _ = muon_update_leaf(
                    full_path,
                    p_stack,
                    g_stack,
                    m_stack,
                    v_values[0],
                )
                block_param_values[leaf_path] = p_new
                block_m_values[leaf_path] = m_new
                block_v_values[leaf_path] = v_values
            else:
                v_stack = jnp.stack([_get_path(block, leaf_path) for block in state["v"]["blocks"]], axis=0)
                p_new, m_new, v_new = adam_update_leaf(
                    full_path,
                    p_stack,
                    g_stack,
                    m_stack,
                    v_stack,
                )
                block_param_values[leaf_path] = p_new
                block_m_values[leaf_path] = m_new
                block_v_values[leaf_path] = v_new

        new_params["blocks"] = [
            _tree_from_leaf_values(block_template, block_param_values, index=i)
            for i in range(len(params["blocks"]))
        ]
        new_m["blocks"] = [
            _tree_from_leaf_values(block_template, block_m_values, index=i)
            for i in range(len(params["blocks"]))
        ]
        new_v["blocks"] = [
            _tree_from_leaf_values(block_template, block_v_values, index=i)
            for i in range(len(params["blocks"]))
        ]
        return new_params, {"m": new_m, "v": new_v}

    updated = _tree_map_with_path(update_leaf, params)
    return (
        extract_updated(updated, 0),
        {
            "m": extract_updated(updated, 1),
            "v": extract_updated(updated, 2),
        },
    )


def estimate_forward_flops(config, pos_len, score_mode="simple"):
    s = pos_len * pos_len
    d = config["hidden_size"]
    ff = config["ffn_dim"]
    blocks = config["num_layers"]
    attn_proj = 2 * s * d * d * 4
    attn_scores = 2 * s * s * d
    attn_values = 2 * s * s * d
    ffn = 3 * 2 * s * d * ff
    trunk = (attn_proj + attn_scores + attn_values + ffn) * blocks
    conv = 2 * configs.get_num_bin_input_features(config) * d * 9 * s
    global_flops = 2 * configs.get_num_global_input_features(config) * d
    policy = 2 * s * d * 6 + 2 * d * 6
    value = 2 * s * d * 29
    score_len = (s + 60) * 2
    score = 2 * d * score_len
    if score_mode != "simple":
        score = 2 * d * (score_len * config["num_scorebeliefs"] + config["num_scorebeliefs"])
    return trunk + conv + global_flops + policy + value + score


def estimate_forward_component_flops(config, pos_len, score_mode="simple"):
    s = pos_len * pos_len
    d = config["hidden_size"]
    ff = config["ffn_dim"]
    blocks = config["num_layers"]
    attn_qkv = 3 * 2 * s * d * d
    attn_scores = 2 * s * s * d
    attn_values = 2 * s * s * d
    attn_out_proj = 2 * s * d * d
    ffn_upgate = 2 * 2 * s * d * ff
    ffn_down = 2 * s * ff * d
    block = attn_qkv + attn_scores + attn_values + attn_out_proj + ffn_upgate + ffn_down
    conv = 2 * configs.get_num_bin_input_features(config) * d * 9 * s
    global_flops = 2 * configs.get_num_global_input_features(config) * d
    policy = 2 * s * d * 6 + 2 * d * 6
    value = 2 * s * d * 29
    score_len = (s + 60) * 2
    score = 2 * d * score_len
    if score_mode != "simple":
        score = 2 * d * (score_len * config["num_scorebeliefs"] + config["num_scorebeliefs"])
    heads = policy + value + score
    return {
        "stem_fwd": conv + global_flops,
        "block0_qkv_proj": attn_qkv,
        "block0_attention_core": attn_scores + attn_values,
        "block0_out_proj_residual": attn_out_proj,
        "block0_ffn_upgate": ffn_upgate,
        "block0_ffn_down_residual": ffn_down,
        "block0_total_fwd": block,
        "trunk_all_blocks_fwd": block * blocks,
        "heads_fwd": heads,
        "full_forward": block * blocks + conv + global_flops + heads,
    }


def estimate_muon_update_flops(config, row_split_size=0, fuse_projections=False, muon_target="all", polar_steps=5):
    d = config["hidden_size"]
    heads = config["num_heads"]
    ff = config["ffn_dim"]
    layers = config["num_layers"]
    target = str(muon_target).lower()

    def polar_flops(m, n):
        if m > n:
            m, n = n, m
        return polar_steps * (4 * m * m * n + 2 * m * m * m)

    def split_flops(m, n):
        if row_split_size > 0 and m > row_split_size and m % row_split_size == 0:
            return (m // row_split_size) * polar_flops(row_split_size, n)
        return polar_flops(m, n)

    def include(kind, m, n):
        if target == "none":
            return 0
        if target == "all":
            return split_flops(m, n)
        if target == "attn" and kind == "attn":
            return split_flops(m, n)
        if target == "ffn" and kind == "ffn":
            return split_flops(m, n)
        if target == "square" and m == n:
            return split_flops(m, n)
        return 0

    head_dim = d // heads
    if fuse_projections:
        qkv = 0 if target in ("none", "ffn", "square") else 3 * heads * polar_flops(head_dim, d)
        per_layer = qkv + include("attn", d, d) + include("ffn", 2 * ff, d) + include("ffn", d, ff)
    else:
        qkv = 0 if target in ("none", "ffn") else 3 * heads * polar_flops(head_dim, d)
        per_layer = (
            qkv
            + include("attn", d, d)
            + include("ffn", ff, d)
            + include("ffn", ff, d)
            + include("ffn", d, ff)
        )
    return layers * per_layer


def lr_wd_at_step(step, args, samples_per_step):
    if args.lr_schedule == "constant":
        return args.lr, args.wd
    warmup_steps = max(1, args.warmup_samples // samples_per_step)
    total_steps = max(1, args.max_training_samples // samples_per_step)
    if step < warmup_steps:
        mult = (step + 1) / warmup_steps
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        mult = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.lr * mult, args.wd


def save_checkpoint(path, params, opt_state, meta, ema_params=None):
    import jax

    os.makedirs(os.path.dirname(path), exist_ok=True)
    host_state = {
        "params": jax.device_get(params),
        "opt_state": jax.device_get(opt_state),
        "meta": meta,
    }
    if ema_params is not None:
        host_state["ema_params"] = jax.device_get(ema_params)
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(host_state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def stack_batch_list(batch_list):
    return {
        key: np.stack([batch[key] for batch in batch_list], axis=0)
        for key in batch_list[0]
    }


def shard_batch_for_data_parallel(batch, num_devices, per_device_batch):
    return [
        _tree_map(
            lambda x, device_idx=device_idx: np.asarray(x)[
                device_idx * per_device_batch:(device_idx + 1) * per_device_batch
            ],
            batch,
        )
        for device_idx in range(num_devices)
    ]


def shard_stacked_batches_for_data_parallel(batch, num_devices, per_device_batch):
    return [
        _tree_map(
            lambda x, device_idx=device_idx: np.asarray(x)[
                :,
                device_idx * per_device_batch:(device_idx + 1) * per_device_batch,
            ],
            batch,
        )
        for device_idx in range(num_devices)
    ]


def first_replica(tree):
    return _tree_map(lambda x: x[0], tree)


def main():
    global jax_losses, jax_model

    parser = argparse.ArgumentParser(description="JAX TPU prototype trainer for KataGo nano")
    parser.add_argument("--traindir", required=True)
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--pos-len", type=int, default=19)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-kind", type=str, default="b12c2048")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon", "sgd", "none"])
    parser.add_argument("--muon-lr-multiplier", type=float, default=0.2)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-row-split-size", type=int, default=0)
    parser.add_argument("--muon-target", type=str, default="all", choices=["all", "attn", "ffn", "square", "none"])
    parser.add_argument("--muon-polar-steps", type=int, default=5)
    parser.add_argument("--muon-group-blocks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loss-profile", type=str, default="full",
                        choices=["full", "policy_value", "policy_only", "value_only"])
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--lr-schedule", type=str, default="cosine", choices=["cosine", "constant"])
    parser.add_argument("--max-training-samples", type=int, default=32768)
    parser.add_argument("--warmup-samples", type=int, default=4096)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--steps-per-jit", type=int, default=1)
    parser.add_argument("--save-every-samples", type=int, default=0)
    parser.add_argument("--val-every-samples", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=16)
    parser.add_argument("--symmetry-type", type=str, default="xyt")
    parser.add_argument("--enable-history-matrices", action="store_true")
    parser.add_argument("--allow-nonfull-mask", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-final-save", action="store_true")
    parser.add_argument("--separate-projections", action="store_true")
    parser.add_argument("--fuse-projections", action="store_true")
    parser.add_argument("--attention-impl", type=str, default="manual", choices=["manual", "xla"])
    parser.add_argument("--activation-dtype", type=str, default="float32",
                        choices=["float32", "fp32", "bfloat16", "bf16"])
    parser.add_argument("--param-dtype", type=str, default="float32",
                        choices=["float32", "fp32", "bfloat16", "bf16"])
    parser.add_argument("--opt-state-dtype", type=str, default="float32",
                        choices=["float32", "fp32", "bfloat16", "bf16"])
    parser.add_argument("--opt-update-dtype", type=str, default="float32",
                        choices=["float32", "fp32", "bfloat16", "bf16"])
    parser.add_argument("--rope-dtype", type=str, default="float32",
                        choices=["float32", "fp32", "bfloat16", "bf16"])
    parser.add_argument("--ffn-mul-dtype", type=str, default="float32",
                        choices=["float32", "fp32", "bfloat16", "bf16"])
    parser.add_argument("--attention-logits-dtype", type=str, default="float32",
                        choices=["float32", "fp32", "bfloat16", "bf16"])
    parser.add_argument("--int8-train", action="store_true")
    parser.add_argument("--int8-target", type=str, default="ffn",
                        choices=["none", "ffn", "attn", "attn_proj", "attn_core", "heads", "stem", "trunk", "all"])
    parser.add_argument("--int8-fwd-bits", type=int, default=8)
    parser.add_argument("--int8-bwd-bits", type=int, default=8)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--log-grad-norm", action="store_true")
    parser.add_argument("--log-step-time", action="store_true")
    parser.add_argument("--component-profile", action="store_true")
    parser.add_argument("--component-profile-repeats", type=int, default=3)
    parser.add_argument("--component-profile-grad", action="store_true")
    parser.add_argument("--donate-train-buffers", action="store_true")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--muon-split-jit", action="store_true")
    parser.add_argument("--stack-blocks", action="store_true")
    parser.add_argument("--scan-blocks", action="store_true")
    parser.add_argument("--remat-blocks", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--score-mode", type=str, default="simple", choices=["simple"])
    parser.add_argument("--xla-peak-tflops", type=float, default=918.0)
    args = parser.parse_args()
    if args.steps_per_jit < 1:
        raise ValueError("--steps-per-jit must be >= 1")
    if args.component_profile_repeats < 1:
        raise ValueError("--component-profile-repeats must be >= 1")
    if args.muon_row_split_size < 0:
        raise ValueError("--muon-row-split-size must be >= 0")
    if args.muon_polar_steps < 1 or args.muon_polar_steps > len(_POLAR_EXPRESS_COEFFS):
        raise ValueError(f"--muon-polar-steps must be between 1 and {len(_POLAR_EXPRESS_COEFFS)}")
    if args.int8_fwd_bits < 2 or args.int8_fwd_bits > 8:
        raise ValueError("--int8-fwd-bits must be between 2 and 8")
    if args.int8_bwd_bits < 2 or args.int8_bwd_bits > 8:
        raise ValueError("--int8-bwd-bits must be between 2 and 8")
    if args.ema_decay < 0.0 or args.ema_decay >= 1.0:
        raise ValueError("--ema-decay must be in [0, 1)")
    if args.data_parallel and args.component_profile:
        raise ValueError("--component-profile is currently single-device only; disable it with DATA_PARALLEL=1")
    if args.data_parallel and args.muon_split_jit:
        raise ValueError("--muon-split-jit is not supported with --data-parallel")
    track_grad_norm = args.log_grad_norm or args.grad_clip_norm > 0

    try:
        import jax_losses as _jax_losses
        import jax_model as _jax_model
    except ModuleNotFoundError as exc:
        if exc.name == "jax":
            raise SystemExit(
                "JAX is not installed. On a Colab TPU runtime, run: "
                "bash colab_install_jax_tpu.sh"
            ) from exc
        raise
    jax_losses = _jax_losses
    jax_model = _jax_model

    import jax
    import jax.numpy as jnp

    activation_dtype = jax_model.dtype_from_name(args.activation_dtype)
    param_dtype = jax_model.dtype_from_name(args.param_dtype)
    opt_state_dtype = jax_model.dtype_from_name(args.opt_state_dtype)
    opt_update_dtype = jax_model.dtype_from_name(args.opt_update_dtype)
    rope_dtype = jax_model.dtype_from_name(args.rope_dtype)
    ffn_mul_dtype = jax_model.dtype_from_name(args.ffn_mul_dtype)
    attention_logits_dtype = jax_model.dtype_from_name(args.attention_logits_dtype)
    int8_dot_general = None
    if args.int8_train:
        int8_dot_general = jax_model.make_int8_dot_general(
            fwd_bits=args.int8_fwd_bits,
            bwd_bits=args.int8_bwd_bits,
        )

    os.makedirs(args.traindir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.traindir, "train_jax.log"), mode="a"),
        ],
    )
    logging.info("Args: %s", vars(args))
    logging.info("JAX devices: %s", jax.devices())
    data_parallel_devices = tuple(jax.local_devices()) if args.data_parallel else ()
    num_data_parallel_devices = len(data_parallel_devices)
    per_device_batch = args.batch_size
    if args.data_parallel:
        if num_data_parallel_devices < 1:
            raise RuntimeError("--data-parallel requested, but JAX reported no local devices")
        if args.batch_size % num_data_parallel_devices != 0:
            raise ValueError(
                f"--batch-size={args.batch_size} must be divisible by local device count "
                f"{num_data_parallel_devices} when --data-parallel is enabled"
            )
        per_device_batch = args.batch_size // num_data_parallel_devices
        logging.info(
            "Data parallel enabled: local_devices=%d, global_batch=%d, per_device_batch=%d",
            num_data_parallel_devices,
            args.batch_size,
            per_device_batch,
        )
        if num_data_parallel_devices > 1 and args.xla_peak_tflops <= 1000.0:
            logging.warning(
                "XLA_PEAK_TFLOPS=%.1f looks like a single-chip peak. "
                "For v6e-8 MFU, set XLA_PEAK_TFLOPS to the aggregate peak, e.g. 7344.",
                args.xla_peak_tflops,
            )

    if args.model_kind not in configs.config_of_name:
        raise ValueError(f"Unknown model kind {args.model_kind}")
    model_config = configs.config_of_name[args.model_kind]
    logging.info("Model config: %s", json.dumps(model_config, indent=2))

    rope_cache = tuple(jax.device_put(x) for x in jax_model.precompute_rope(
        model_config["hidden_size"] // model_config["num_heads"], args.pos_len
    ))
    score_offsets = jax.device_put(jax_model.score_belief_offsets(args.pos_len))
    checkpoint_path = os.path.join(args.traindir, "checkpoint_jax.pkl")

    moving_sum = jnp.asarray(0.0, dtype=jnp.float32)
    moving_weight = jnp.asarray(0.0, dtype=jnp.float32)
    ema_params = ()
    total_samples = 0
    step = 0

    if not args.no_resume and os.path.exists(checkpoint_path):
        state = load_checkpoint(checkpoint_path)
        meta = state.get("meta", {})
        if meta.get("model_config") != model_config:
            raise ValueError(f"{checkpoint_path}: checkpoint model_config does not match {args.model_kind}")
        if int(meta.get("pos_len", args.pos_len)) != args.pos_len:
            raise ValueError(f"{checkpoint_path}: checkpoint pos_len does not match {args.pos_len}")
        if meta.get("optimizer", args.optimizer) != args.optimizer:
            raise ValueError(
                f"{checkpoint_path}: checkpoint optimizer={meta.get('optimizer')} does not match "
                f"--optimizer={args.optimizer}; set NO_RESUME=1 when switching optimizers"
            )
        if args.optimizer == "muon" and meta.get("muon_target", "all") != args.muon_target:
            raise ValueError(
                f"{checkpoint_path}: checkpoint muon_target={meta.get('muon_target', 'all')} does not match "
                f"--muon-target={args.muon_target}; set NO_RESUME=1 when changing Muon target"
            )
        state_params = state["params"]
        expected_stacked_blocks = args.stack_blocks or args.scan_blocks
        if isinstance(state_params.get("blocks"), dict) != expected_stacked_blocks:
            raise ValueError(f"{checkpoint_path}: checkpoint block layout does not match --stack-blocks/--scan-blocks")
        if bool(meta.get("int8_train", False)) != args.int8_train:
            raise ValueError(
                f"{checkpoint_path}: checkpoint int8_train={meta.get('int8_train', False)} does not match "
                f"--int8-train={args.int8_train}; set NO_RESUME=1 when changing INT8 training"
            )
        if args.int8_train:
            if meta.get("int8_target", "ffn") != args.int8_target:
                raise ValueError(
                    f"{checkpoint_path}: checkpoint int8_target={meta.get('int8_target', 'ffn')} does not match "
                    f"--int8-target={args.int8_target}; set NO_RESUME=1 when changing INT8 target"
                )
            if int(meta.get("int8_fwd_bits", args.int8_fwd_bits)) != args.int8_fwd_bits:
                raise ValueError(f"{checkpoint_path}: checkpoint int8_fwd_bits does not match --int8-fwd-bits")
            if int(meta.get("int8_bwd_bits", args.int8_bwd_bits)) != args.int8_bwd_bits:
                raise ValueError(f"{checkpoint_path}: checkpoint int8_bwd_bits does not match --int8-bwd-bits")
        params = _tree_map(lambda x: jnp.asarray(x, dtype=param_dtype), state_params)
        opt_state = _tree_map(lambda x: jnp.asarray(x, dtype=opt_state_dtype), state["opt_state"])
        if args.ema_decay > 0:
            if "ema_params" in state:
                ema_params = _tree_map(lambda x: jnp.asarray(x, dtype=jnp.float32), state["ema_params"])
                logging.info("EMA state loaded")
            else:
                ema_params = _tree_map(lambda x: jnp.asarray(x, dtype=jnp.float32), params)
                logging.info("EMA enabled but checkpoint has no EMA state; initialized from params")
        step = int(meta.get("step", 0))
        total_samples = int(meta.get("samples", 0))
        moving_sum = jnp.asarray(float(meta.get("moving_sum", 0.0)), dtype=jnp.float32)
        moving_weight = jnp.asarray(float(meta.get("moving_weight", 0.0)), dtype=jnp.float32)
        logging.info("Resumed checkpoint at step %d, %d samples", step, total_samples)
    else:
        key = jax.random.PRNGKey(args.seed)
        params = jax_model.init_params(
            key,
            model_config,
            args.pos_len,
            init_std=args.init_std,
            score_mode=args.score_mode,
            fuse_projections=args.fuse_projections and not args.separate_projections,
            stack_blocks=args.stack_blocks or args.scan_blocks,
        )
        params = _tree_map(lambda x: x.astype(param_dtype), params)
        if args.optimizer == "muon":
            opt_state = init_muon_adamw_state(
                params,
                dtype=opt_state_dtype,
                muon_target=args.muon_target,
            )
        else:
            opt_state = init_adam_state(params, dtype=opt_state_dtype)
        if args.ema_decay > 0:
            ema_params = _tree_map(lambda x: jnp.asarray(x, dtype=jnp.float32), params)
            logging.info("EMA enabled: decay=%.6f", args.ema_decay)

    if args.data_parallel:
        params = jax.device_put_replicated(params, data_parallel_devices)
        opt_state = jax.device_put_replicated(opt_state, data_parallel_devices)
        if args.ema_decay > 0:
            ema_params = jax.device_put_replicated(ema_params, data_parallel_devices)
        moving_sum = jax.device_put_replicated(moving_sum, data_parallel_devices)
        moving_weight = jax.device_put_replicated(moving_weight, data_parallel_devices)
    else:
        params = jax.device_put(params)
        opt_state = jax.device_put(opt_state)
        if args.ema_decay > 0:
            ema_params = jax.device_put(ema_params)
        moving_sum = jax.device_put(moving_sum)
        moving_weight = jax.device_put(moving_weight)

    td_value_loss_scales = (0.6, 0.6, 0.6)

    def loss_fn(params_, batch_, moving_sum_, moving_weight_, is_training):
        outputs = jax_model.forward(params_, batch_["binaryInputNCHW"], batch_["globalInputNC"],
                                    model_config, args.pos_len, rope_cache,
                                    attention_impl=args.attention_impl,
                                    activation_dtype=activation_dtype,
                                    remat_blocks=args.remat_blocks,
                                    scan_blocks=args.scan_blocks,
                                    rope_dtype=rope_dtype,
                                    ffn_mul_dtype=ffn_mul_dtype,
                                    attention_logits_dtype=attention_logits_dtype,
                                    int8_dot_general=int8_dot_general,
                                    int8_target=args.int8_target)
        if args.loss_profile != "full":
            return jax_losses.profile_loss_core(
                outputs,
                batch_["policyTargetsNCMove"],
                batch_["globalTargetsNC"],
                moving_sum_,
                moving_weight_,
                profile=args.loss_profile,
                value_loss_scale=0.6,
            )
        return jax_losses.postprocess_and_loss_core(
            outputs,
            score_offsets,
            batch_["policyTargetsNCMove"],
            batch_["globalTargetsNC"],
            batch_["scoreDistrN"],
            batch_["valueTargetsNCHW"],
            args.pos_len,
            moving_sum_,
            moving_weight_,
            is_training=is_training,
            td_value_loss_scales=td_value_loss_scales,
        )

    def train_one_step(
        params_,
        opt_state_,
        batch_,
        moving_sum_,
        moving_weight_,
        opt_step,
        lr,
        wd,
        data_axis_name=None,
    ):
        def scalar_loss(p):
            loss, metrics, new_moving_sum, new_moving_weight = loss_fn(
                p, batch_, moving_sum_, moving_weight_, True
            )
            return loss, (metrics, new_moving_sum, new_moving_weight)

        (loss, (metrics, new_moving_sum, new_moving_weight)), grads = jax.value_and_grad(
            scalar_loss, has_aux=True
        )(params_)
        if data_axis_name is not None:
            grads = jax.lax.pmean(grads, data_axis_name)
            metrics_sum = jax.lax.psum(metrics, data_axis_name)
            metrics = metrics_sum.at[0].set(metrics_sum[0] / num_data_parallel_devices)
            new_moving_sum = jax.lax.pmean(new_moving_sum, data_axis_name)
            new_moving_weight = jax.lax.pmean(new_moving_weight, data_axis_name)
        if track_grad_norm:
            grad_norm = jnp.sqrt(_tree_sum_squares(grads))
        else:
            grad_norm = jnp.asarray(0.0, dtype=jnp.float32)
        if args.grad_clip_norm > 0:
            scale = jnp.minimum(1.0, args.grad_clip_norm / (grad_norm + 1e-6))
            grads = _tree_map(lambda g: g * scale, grads)
        if args.optimizer == "adamw":
            new_params, new_opt_state = adamw_update(
                params_, grads, opt_state_, opt_step, lr, wd,
                state_dtype=opt_state_dtype,
                param_dtype=param_dtype,
                update_dtype=opt_update_dtype,
            )
        elif args.optimizer == "muon":
            new_params, new_opt_state = muon_adamw_update(
                params_, grads, opt_state_, opt_step, lr, wd, model_config,
                state_dtype=opt_state_dtype,
                param_dtype=param_dtype,
                update_dtype=opt_update_dtype,
                muon_lr_multiplier=args.muon_lr_multiplier,
                muon_momentum=args.muon_momentum,
                muon_row_split_size=args.muon_row_split_size,
                muon_target=args.muon_target,
                muon_polar_steps=args.muon_polar_steps,
                muon_group_blocks=args.muon_group_blocks,
            )
        elif args.optimizer == "sgd":
            new_params = sgd_update(
                params_, grads, lr, wd,
                param_dtype=param_dtype,
                update_dtype=opt_update_dtype,
            )
            new_opt_state = opt_state_
        elif args.optimizer == "none":
            new_params = params_
            new_opt_state = opt_state_
        else:
            raise ValueError(f"Unknown optimizer: {args.optimizer}")
        return new_params, new_opt_state, new_moving_sum, new_moving_weight, metrics, grad_norm

    def update_ema_params(ema_params_, params_):
        decay = jnp.asarray(args.ema_decay, dtype=jnp.float32)
        one_minus_decay = jnp.asarray(1.0 - args.ema_decay, dtype=jnp.float32)
        return _tree_map(
            lambda ema, p: decay * ema.astype(jnp.float32) + one_minus_decay * p.astype(jnp.float32),
            ema_params_,
            params_,
        )

    def train_steps_impl(
        params_,
        opt_state_,
        ema_params_,
        batches_,
        moving_sum_,
        moving_weight_,
        opt_steps,
        lrs,
        wds,
        data_axis_name=None,
    ):
        def body(carry, xs):
            params_i, opt_state_i, ema_params_i, moving_sum_i, moving_weight_i = carry
            batch_i, opt_step_i, lr_i, wd_i = xs
            params_i, opt_state_i, moving_sum_i, moving_weight_i, metrics_i, grad_norm_i = train_one_step(
                params_i,
                opt_state_i,
                batch_i,
                moving_sum_i,
                moving_weight_i,
                opt_step_i,
                lr_i,
                wd_i,
                data_axis_name=data_axis_name,
            )
            if args.ema_decay > 0:
                ema_params_i = update_ema_params(ema_params_i, params_i)
            return (params_i, opt_state_i, ema_params_i, moving_sum_i, moving_weight_i), (metrics_i, grad_norm_i)

        (params_, opt_state_, ema_params_, moving_sum_, moving_weight_), (metrics_seq, grad_norm_seq) = jax.lax.scan(
            body,
            (params_, opt_state_, ema_params_, moving_sum_, moving_weight_),
            (batches_, opt_steps, lrs, wds),
        )
        return (
            params_,
            opt_state_,
            ema_params_,
            moving_sum_,
            moving_weight_,
            jnp.sum(metrics_seq, axis=0),
            jnp.sum(grad_norm_seq),
        )

    if args.donate_train_buffers:
        train_donate_argnums = (0, 1) if args.optimizer in ("adamw", "muon") else (0,)
        if args.ema_decay > 0:
            train_donate_argnums = train_donate_argnums + (2,)
    else:
        train_donate_argnums = ()
    train_steps = jax.jit(train_steps_impl, donate_argnums=train_donate_argnums)
    train_steps_dp = None
    if args.data_parallel:
        ema_in_axes = 0 if args.ema_decay > 0 else None
        train_steps_dp = jax.pmap(
            lambda params_, opt_state_, ema_params_, batches_, moving_sum_, moving_weight_, opt_steps, lrs, wds: train_steps_impl(
                params_,
                opt_state_,
                ema_params_,
                batches_,
                moving_sum_,
                moving_weight_,
                opt_steps,
                lrs,
                wds,
                data_axis_name="data",
            ),
            axis_name="data",
            in_axes=(0, 0, ema_in_axes, 0, 0, 0, None, None, None),
            donate_argnums=train_donate_argnums,
        )

    def loss_grad_step_impl(params_, batch_, moving_sum_, moving_weight_):
        def scalar_loss(p):
            loss, metrics, new_moving_sum, new_moving_weight = loss_fn(
                p, batch_, moving_sum_, moving_weight_, True
            )
            return loss, (metrics, new_moving_sum, new_moving_weight)

        (_, (metrics, new_moving_sum, new_moving_weight)), grads = jax.value_and_grad(
            scalar_loss, has_aux=True
        )(params_)
        if track_grad_norm:
            grad_norm = jnp.sqrt(_tree_sum_squares(grads))
        else:
            grad_norm = jnp.asarray(0.0, dtype=jnp.float32)
        if args.grad_clip_norm > 0:
            scale = jnp.minimum(1.0, args.grad_clip_norm / (grad_norm + 1e-6))
            grads = _tree_map(lambda g: g * scale, grads)
        return grads, new_moving_sum, new_moving_weight, metrics, grad_norm

    loss_grad_step = jax.jit(loss_grad_step_impl)
    ema_update_step = jax.jit(update_ema_params) if args.ema_decay > 0 else None

    def muon_update_step_impl(params_, opt_state_, grads_, opt_step, lr, wd):
        return muon_adamw_update(
            params_,
            grads_,
            opt_state_,
            opt_step,
            lr,
            wd,
            model_config,
            state_dtype=opt_state_dtype,
            param_dtype=param_dtype,
            update_dtype=opt_update_dtype,
            muon_lr_multiplier=args.muon_lr_multiplier,
            muon_momentum=args.muon_momentum,
            muon_row_split_size=args.muon_row_split_size,
            muon_target=args.muon_target,
            muon_polar_steps=args.muon_polar_steps,
            muon_group_blocks=args.muon_group_blocks,
        )

    muon_update_donate_argnums = (0, 1) if args.donate_train_buffers else ()
    muon_update_step = jax.jit(muon_update_step_impl, donate_argnums=muon_update_donate_argnums)

    @jax.jit
    def eval_step(params_, batch_, moving_sum_, moving_weight_):
        _, metrics, _, _ = loss_fn(params_, batch_, moving_sum_, moving_weight_, False)
        return metrics

    eval_step_dp = None
    if args.data_parallel:
        def eval_step_dp_impl(params_, batch_, moving_sum_, moving_weight_):
            _, metrics, _, _ = loss_fn(params_, batch_, moving_sum_, moving_weight_, False)
            metrics_sum = jax.lax.psum(metrics, "data")
            return metrics_sum.at[0].set(metrics_sum[0] / num_data_parallel_devices)

        eval_step_dp = jax.pmap(
            eval_step_dp_impl,
            axis_name="data",
            in_axes=(0, 0, 0, 0),
        )

    train_files = jax_data.list_npz_files(args.datadir, "train")
    if not train_files:
        raise RuntimeError(f"No training files found in {os.path.join(args.datadir, 'train')}")
    logging.info("Training files: %d", len(train_files))

    val_files = jax_data.list_npz_files(args.datadir, "val")
    logging.info("Validation files: %d", len(val_files))

    samples_per_step = args.batch_size
    forward_flops = estimate_forward_flops(model_config, args.pos_len, args.score_mode)
    train_flops_per_sample = forward_flops * 3
    logging.info(
        "FLOPs/sample fwd=%.2fG train=%.2fG",
        forward_flops / 1e9,
        train_flops_per_sample / 1e9,
    )
    if args.int8_train:
        logging.info(
            "AQT INT8 training enabled: target=%s, fwd_bits=%d, bwd_bits=%d. "
            "Parameters and optimizer state remain floating point; only selected dot_general ops are quantized. "
            "The standard TFLOPS/MFU logs still use model FLOPs and XLA_PEAK_TFLOPS, so compare samples/s too.",
            args.int8_target,
            args.int8_fwd_bits,
            args.int8_bwd_bits,
        )
    if args.optimizer == "muon":
        muon_update_flops = estimate_muon_update_flops(
            model_config,
            row_split_size=args.muon_row_split_size,
            fuse_projections=args.fuse_projections and not args.separate_projections,
            muon_target=args.muon_target,
            polar_steps=args.muon_polar_steps,
        )
        logging.info(
            "Muon update approx FLOPs/step=%.2fT (target=%s, polar_steps=%d, row_split=%d). "
            "The standard TFLOPS/MFU logs count model train FLOPs only.",
            muon_update_flops / 1e12,
            args.muon_target,
            args.muon_polar_steps,
            args.muon_row_split_size,
        )
        if args.muon_split_jit:
            logging.info(
                "Muon split-JIT fallback is enabled: loss/grad and optimizer update compile separately. "
                "This can avoid oversized HLOs, but it materializes gradients between programs and is slower."
            )
        else:
            logging.info(
                "Muon uses the single train-step JIT path. If compilation stalls, retry with MUON_SPLIT_JIT=1."
            )
    component_flops = estimate_forward_component_flops(model_config, args.pos_len, args.score_mode)

    def block_until_ready(value):
        return jax.block_until_ready(value)

    def first_transformer_block(params_):
        blocks = params_["blocks"]
        if isinstance(blocks, dict):
            return jax_model.tree_index(blocks, 0)
        return blocks[0]

    def run_component_profile():
        logging.info(
            "COMPONENT_PROFILE starting: repeats=%d grad=%s. "
            "These are separately-jitted microbenchmarks, so use them for bottleneck direction; "
            "XLA may fuse/schedule ops differently inside the full train step.",
            args.component_profile_repeats,
            args.component_profile_grad,
        )
        profile_batches = jax_data.read_npz_batches(
            train_files,
            args.batch_size,
            args.pos_len,
            model_config,
            symmetry_type=args.symmetry_type,
            enable_history_matrices=args.enable_history_matrices,
            seed=args.seed + 424242,
            allow_nonfull_mask=args.allow_nonfull_mask,
        )
        try:
            batch = jax.device_put(next(profile_batches))
        except StopIteration:
            logging.warning("COMPONENT_PROFILE skipped: no full training batch available")
            return

        rope_cos, rope_sin = rope_cache
        num_heads = model_config["num_heads"]

        def time_component(name, fn, *fn_args, flops_per_sample=0.0):
            compiled = jax.jit(fn)
            compile_start = time.perf_counter()
            result = block_until_ready(compiled(*fn_args))
            compile_first = time.perf_counter() - compile_start

            times = []
            for _ in range(args.component_profile_repeats):
                run_start = time.perf_counter()
                result = block_until_ready(compiled(*fn_args))
                times.append(time.perf_counter() - run_start)
            times_np = np.asarray(times, dtype=np.float64)
            mean_elapsed = float(times_np.mean())
            min_elapsed = float(times_np.min())
            max_elapsed = float(times_np.max())

            if flops_per_sample > 0 and min_elapsed > 0:
                best_tflops = args.batch_size * flops_per_sample / min_elapsed / 1e12
                best_mfu = best_tflops / args.xla_peak_tflops * 100.0 if args.xla_peak_tflops > 0 else 0.0
                perf_text = f" est_TFLOPS_min={best_tflops:.2f} est_MFU_min={best_mfu:.2f}%"
            else:
                perf_text = ""
            logging.info(
                "COMPONENT_TIME name=%s compile_first=%.4fs mean=%.6fs min=%.6fs max=%.6fs repeats=%d%s",
                name,
                compile_first,
                mean_elapsed,
                min_elapsed,
                max_elapsed,
                args.component_profile_repeats,
                perf_text,
            )
            return result

        def stem_fn(params_i, batch_i):
            return jax_model.forward_stem(
                params_i,
                batch_i["binaryInputNCHW"],
                batch_i["globalInputNC"],
                model_config,
                args.pos_len,
                activation_dtype,
                int8_dot_general=int8_dot_general,
                int8_target=args.int8_target,
            )

        x_stem = time_component(
            "stem_fwd",
            stem_fn,
            params,
            batch,
            flops_per_sample=component_flops["stem_fwd"],
        )
        block0 = first_transformer_block(params)

        x_norm1 = time_component(
            "block0_norm1",
            lambda block_i, x_i: jax_model.rms_norm(block_i["norm1"], x_i),
            block0,
            x_stem,
        )

        def qkv_fn(block_i, x_norm_i):
            bsz, seq_len, channels = x_norm_i.shape
            head_dim = channels // num_heads
            if "qkv_proj" in block_i:
                qkv = jax_model.linear(
                    block_i["qkv_proj"],
                    x_norm_i,
                    out_dtype=activation_dtype,
                    dot_general=jax_model.dot_for_target(
                        int8_dot_general, args.int8_target, "attn", "attn_proj", "trunk"
                    ),
                )
                q, k, v = jnp.split(qkv, 3, axis=-1)
            else:
                attn_proj_dot = jax_model.dot_for_target(
                    int8_dot_general, args.int8_target, "attn", "attn_proj", "trunk"
                )
                q = jax_model.linear(block_i["q_proj"], x_norm_i, out_dtype=activation_dtype, dot_general=attn_proj_dot)
                k = jax_model.linear(block_i["k_proj"], x_norm_i, out_dtype=activation_dtype, dot_general=attn_proj_dot)
                v = jax_model.linear(block_i["v_proj"], x_norm_i, out_dtype=activation_dtype, dot_general=attn_proj_dot)
            return (
                q.reshape(bsz, seq_len, num_heads, head_dim),
                k.reshape(bsz, seq_len, num_heads, head_dim),
                v.reshape(bsz, seq_len, num_heads, head_dim),
            )

        q, k, v = time_component(
            "block0_qkv_proj",
            qkv_fn,
            block0,
            x_norm1,
            flops_per_sample=component_flops["block0_qkv_proj"],
        )
        q, k = time_component(
            "block0_rope",
            lambda q_i, k_i: jax_model.apply_rope(q_i, k_i, rope_cos, rope_sin, compute_dtype=rope_dtype),
            q,
            k,
        )

        def attention_core_fn(q_i, k_i, v_i):
            bsz, seq_len, heads, head_dim = q_i.shape
            return jax_model.attention(
                q_i,
                k_i,
                v_i,
                attention_impl=args.attention_impl,
                out_dtype=activation_dtype,
                logits_dtype=attention_logits_dtype,
                dot_general=jax_model.dot_for_target(
                    int8_dot_general, args.int8_target, "attn", "attn_core", "trunk"
                ),
            ).reshape(bsz, seq_len, heads * head_dim)

        attn_out = time_component(
            "block0_attention_core",
            attention_core_fn,
            q,
            k,
            v,
            flops_per_sample=component_flops["block0_attention_core"],
        )
        x_attn = time_component(
            "block0_out_proj_residual",
            lambda block_i, x_i, attn_i: (
                x_i + jax_model.linear(
                    block_i["out_proj"],
                    attn_i,
                    out_dtype=activation_dtype,
                    dot_general=jax_model.dot_for_target(
                        int8_dot_general, args.int8_target, "attn", "attn_proj", "trunk"
                    ),
                )
            ).astype(activation_dtype),
            block0,
            x_stem,
            attn_out,
            flops_per_sample=component_flops["block0_out_proj_residual"],
        )
        x_norm2 = time_component(
            "block0_norm2",
            lambda block_i, x_i: jax_model.rms_norm(block_i["norm2"], x_i),
            block0,
            x_attn,
        )

        def ffn_upgate_fn(block_i, x_norm_i):
            ffn_dot = jax_model.dot_for_target(int8_dot_general, args.int8_target, "ffn", "trunk")
            if "ffn_upgate" in block_i:
                upgate = jax_model.linear(
                    block_i["ffn_upgate"],
                    x_norm_i,
                    out_dtype=activation_dtype,
                    dot_general=ffn_dot,
                )
                w1, wg = jnp.split(upgate, 2, axis=-1)
                w1 = jax_model.silu(w1)
            else:
                w1 = jax_model.silu(jax_model.linear(
                    block_i["ffn_w1"], x_norm_i, out_dtype=activation_dtype, dot_general=ffn_dot
                ))
                wg = jax_model.linear(block_i["ffn_wgate"], x_norm_i, out_dtype=activation_dtype, dot_general=ffn_dot)
            return (w1.astype(ffn_mul_dtype) * wg.astype(ffn_mul_dtype)).astype(activation_dtype)

        hidden = time_component(
            "block0_ffn_upgate",
            ffn_upgate_fn,
            block0,
            x_norm2,
            flops_per_sample=component_flops["block0_ffn_upgate"],
        )
        _ = time_component(
            "block0_ffn_down_residual",
            lambda block_i, x_i, hidden_i: (
                x_i + jax_model.linear(
                    block_i["ffn_w2"],
                    hidden_i,
                    out_dtype=activation_dtype,
                    dot_general=jax_model.dot_for_target(int8_dot_general, args.int8_target, "ffn", "trunk"),
                )
            ).astype(activation_dtype),
            block0,
            x_attn,
            hidden,
            flops_per_sample=component_flops["block0_ffn_down_residual"],
        )
        _ = time_component(
            "block0_total_fwd",
            lambda block_i, x_i: jax_model.transformer_block(
                block_i,
                x_i,
                rope_cos,
                rope_sin,
                num_heads,
                attention_impl=args.attention_impl,
                activation_dtype=activation_dtype,
                rope_dtype=rope_dtype,
                ffn_mul_dtype=ffn_mul_dtype,
                attention_logits_dtype=attention_logits_dtype,
                int8_dot_general=int8_dot_general,
                int8_target=args.int8_target,
            ),
            block0,
            x_stem,
            flops_per_sample=component_flops["block0_total_fwd"],
        )
        x_trunk = time_component(
            "trunk_all_blocks_fwd",
            lambda params_i, x_i: jax_model.forward_trunk(
                params_i,
                x_i,
                model_config,
                rope_cache,
                attention_impl=args.attention_impl,
                activation_dtype=activation_dtype,
                remat_blocks=args.remat_blocks,
                scan_blocks=args.scan_blocks,
                rope_dtype=rope_dtype,
                ffn_mul_dtype=ffn_mul_dtype,
                attention_logits_dtype=attention_logits_dtype,
                int8_dot_general=int8_dot_general,
                int8_target=args.int8_target,
            ),
            params,
            x_stem,
            flops_per_sample=component_flops["trunk_all_blocks_fwd"],
        )
        _ = time_component(
            "heads_fwd",
            lambda params_i, x_i: jax_model.forward_heads(
                params_i,
                x_i,
                args.pos_len,
                int8_dot_general=int8_dot_general,
                int8_target=args.int8_target,
            ),
            params,
            x_trunk,
            flops_per_sample=component_flops["heads_fwd"],
        )
        _ = time_component(
            "full_forward",
            lambda params_i, batch_i: jax_model.forward(
                params_i,
                batch_i["binaryInputNCHW"],
                batch_i["globalInputNC"],
                model_config,
                args.pos_len,
                rope_cache,
                attention_impl=args.attention_impl,
                activation_dtype=activation_dtype,
                remat_blocks=args.remat_blocks,
                scan_blocks=args.scan_blocks,
                rope_dtype=rope_dtype,
                ffn_mul_dtype=ffn_mul_dtype,
                attention_logits_dtype=attention_logits_dtype,
                int8_dot_general=int8_dot_general,
                int8_target=args.int8_target,
            ),
            params,
            batch,
            flops_per_sample=component_flops["full_forward"],
        )
        _ = time_component(
            "loss_forward",
            lambda params_i, batch_i, moving_sum_i, moving_weight_i: loss_fn(
                params_i,
                batch_i,
                moving_sum_i,
                moving_weight_i,
                True,
            )[0],
            params,
            batch,
            moving_sum,
            moving_weight,
            flops_per_sample=component_flops["full_forward"],
        )

        if not args.component_profile_grad:
            logging.info("COMPONENT_PROFILE grad/update skipped; set COMPONENT_PROFILE_GRAD=1 to include them")
            logging.info("COMPONENT_PROFILE complete")
            return

        def loss_grad_fn(params_i, batch_i, moving_sum_i, moving_weight_i):
            def scalar_loss(p):
                return loss_fn(p, batch_i, moving_sum_i, moving_weight_i, True)[0]

            return jax.value_and_grad(scalar_loss)(params_i)

        loss_value, grads = time_component(
            "loss_grad",
            loss_grad_fn,
            params,
            batch,
            moving_sum,
            moving_weight,
            flops_per_sample=train_flops_per_sample,
        )
        del loss_value
        if args.optimizer == "adamw":
            _ = time_component(
                "optimizer_adamw_update",
                lambda params_i, grads_i, opt_state_i: adamw_update(
                    params_i,
                    grads_i,
                    opt_state_i,
                    jnp.asarray(1.0, dtype=jnp.float32),
                    jnp.asarray(args.lr, dtype=jnp.float32),
                    jnp.asarray(args.wd, dtype=jnp.float32),
                    state_dtype=opt_state_dtype,
                    param_dtype=param_dtype,
                    update_dtype=opt_update_dtype,
                )[0],
                params,
                grads,
                opt_state,
            )
        elif args.optimizer == "muon":
            _ = time_component(
                "optimizer_muon_adamw_update",
                lambda params_i, grads_i, opt_state_i: muon_adamw_update(
                    params_i,
                    grads_i,
                    opt_state_i,
                    jnp.asarray(1.0, dtype=jnp.float32),
                    jnp.asarray(args.lr, dtype=jnp.float32),
                    jnp.asarray(args.wd, dtype=jnp.float32),
                    model_config,
                    state_dtype=opt_state_dtype,
                    param_dtype=param_dtype,
                    update_dtype=opt_update_dtype,
                    muon_lr_multiplier=args.muon_lr_multiplier,
                    muon_momentum=args.muon_momentum,
                    muon_row_split_size=args.muon_row_split_size,
                    muon_target=args.muon_target,
                    muon_polar_steps=args.muon_polar_steps,
                    muon_group_blocks=args.muon_group_blocks,
                )[0],
                params,
                grads,
                opt_state,
            )
        elif args.optimizer == "sgd":
            _ = time_component(
                "optimizer_sgd_update",
                lambda params_i, grads_i: sgd_update(
                    params_i,
                    grads_i,
                    jnp.asarray(args.lr, dtype=jnp.float32),
                    jnp.asarray(args.wd, dtype=jnp.float32),
                    param_dtype=param_dtype,
                    update_dtype=opt_update_dtype,
                ),
                params,
                grads,
            )
        else:
            logging.info("COMPONENT_PROFILE optimizer update skipped for optimizer=%s", args.optimizer)

        _ = time_component(
            "train_one_step_full",
            lambda params_i, opt_state_i, batch_i, moving_sum_i, moving_weight_i: train_one_step(
                params_i,
                opt_state_i,
                batch_i,
                moving_sum_i,
                moving_weight_i,
                jnp.asarray(1.0, dtype=jnp.float32),
                jnp.asarray(args.lr, dtype=jnp.float32),
                jnp.asarray(args.wd, dtype=jnp.float32),
            ),
            params,
            opt_state,
            batch,
            moving_sum,
            moving_weight,
            flops_per_sample=train_flops_per_sample,
        )
        logging.info("COMPONENT_PROFILE complete")

    if args.component_profile:
        profile_start = time.perf_counter()
        run_component_profile()
        logging.info("COMPONENT_PROFILE total_time=%.1fs", time.perf_counter() - profile_start)

    running_metrics = jnp.zeros((len(jax_losses.METRIC_KEYS),), dtype=jnp.float32)
    running_grad_norm = jnp.asarray(0.0, dtype=jnp.float32)
    last_print_time = time.perf_counter()
    last_print_step = step
    save_every = args.save_every_samples or args.max_training_samples
    val_every = args.val_every_samples or args.max_training_samples
    last_save_samples = total_samples
    last_val_samples = total_samples

    def checkpoint_meta():
        moving_sum_for_meta = first_replica(moving_sum) if args.data_parallel else moving_sum
        moving_weight_for_meta = first_replica(moving_weight) if args.data_parallel else moving_weight
        return {
            "step": step,
            "samples": total_samples,
            "model_config": model_config,
            "pos_len": args.pos_len,
            "optimizer": args.optimizer,
            "muon_lr_multiplier": args.muon_lr_multiplier,
            "muon_momentum": args.muon_momentum,
            "muon_row_split_size": args.muon_row_split_size,
            "muon_target": args.muon_target,
            "muon_polar_steps": args.muon_polar_steps,
            "muon_group_blocks": args.muon_group_blocks,
            "muon_split_jit": args.muon_split_jit,
            "loss_profile": args.loss_profile,
            "score_mode": args.score_mode,
            "log_step_time": args.log_step_time,
            "moving_sum": float(jax.device_get(moving_sum_for_meta)),
            "moving_weight": float(jax.device_get(moving_weight_for_meta)),
            "ema_decay": args.ema_decay,
            "ema_dtype": "float32" if args.ema_decay > 0 else "none",
            "data_parallel": args.data_parallel,
            "data_parallel_devices": num_data_parallel_devices,
            "per_device_batch": per_device_batch,
            "fuse_projections": args.fuse_projections and not args.separate_projections,
            "attention_impl": args.attention_impl,
            "activation_dtype": args.activation_dtype,
            "param_dtype": args.param_dtype,
            "opt_state_dtype": args.opt_state_dtype,
            "opt_update_dtype": args.opt_update_dtype,
            "rope_dtype": args.rope_dtype,
            "ffn_mul_dtype": args.ffn_mul_dtype,
            "attention_logits_dtype": args.attention_logits_dtype,
            "int8_train": args.int8_train,
            "int8_target": args.int8_target,
            "int8_fwd_bits": args.int8_fwd_bits,
            "int8_bwd_bits": args.int8_bwd_bits,
            "donate_train_buffers": args.donate_train_buffers,
            "stack_blocks": args.stack_blocks or args.scan_blocks,
            "scan_blocks": args.scan_blocks,
            "remat_blocks": args.remat_blocks,
        }

    def log_metric_summary(prefix, samples, metrics_host, batch_count):
        weight_sum = max(float(metrics_host[-1]), 1e-10)
        by_key = dict(zip(jax_losses.METRIC_KEYS, metrics_host.tolist()))
        logging.info(
            "%s [%d samples]: loss=%.4f, p0loss=%.4f, vloss=%.4f, "
            "oloss=%.4f, skloss=%.4f, pacc1=%.4f",
            prefix,
            samples,
            by_key["loss"] / max(batch_count, 1),
            by_key["p0loss"] / weight_sum,
            by_key["vloss"] / weight_sum,
            by_key["oloss"] / weight_sum,
            by_key["skloss"] / weight_sum,
            by_key["pacc1"] / weight_sum,
        )

    def run_validation():
        if not val_files:
            logging.warning("  VAL skipped: no validation files found")
            return
        if args.max_val_batches > 0:
            logging.info("  VAL limited to %d batches", args.max_val_batches)

        val_metrics = jnp.zeros((len(jax_losses.METRIC_KEYS),), dtype=jnp.float32)
        val_count = 0
        for val_batch_idx, batch_np in enumerate(jax_data.read_npz_batches(
            val_files,
            args.batch_size,
            args.pos_len,
            model_config,
            symmetry_type="none",
            enable_history_matrices=args.enable_history_matrices,
            seed=args.seed + total_samples + 999983,
            allow_nonfull_mask=args.allow_nonfull_mask,
        )):
            if args.max_val_batches > 0 and val_batch_idx >= args.max_val_batches:
                break
            if args.data_parallel:
                batch = jax.device_put_sharded(
                    shard_batch_for_data_parallel(
                        batch_np,
                        num_data_parallel_devices,
                        per_device_batch,
                    ),
                    data_parallel_devices,
                )
                val_metrics = val_metrics + first_replica(eval_step_dp(
                    params,
                    batch,
                    moving_sum,
                    moving_weight,
                ))
            else:
                batch = jax.device_put(batch_np)
                val_metrics = val_metrics + eval_step(params, batch, moving_sum, moving_weight)
            val_count += 1

        if val_count == 0:
            logging.warning("  VAL skipped: no full validation batches")
            return
        metrics_host = jax.device_get(val_metrics)
        log_metric_summary("  VAL", total_samples, metrics_host, val_count)

    def run_training_chunk(batch_list):
        nonlocal params, opt_state, ema_params, moving_sum, moving_weight
        nonlocal running_metrics, running_grad_norm, step, total_samples
        chunk_start = time.perf_counter()
        start_step = step
        start_samples = total_samples
        batch_count = len(batch_list)
        lr_wd = [lr_wd_at_step(step + i, args, samples_per_step) for i in range(batch_count)]
        lrs = np.asarray([x[0] for x in lr_wd], dtype=np.float32)
        wds = np.asarray([x[1] for x in lr_wd], dtype=np.float32)
        opt_steps = np.arange(step + 1, step + batch_count + 1, dtype=np.float32)
        exec_start = time.perf_counter()
        if args.optimizer == "muon" and args.muon_split_jit:
            if step == 0:
                logging.info(
                    "Muon uses split JIT execution: compiling loss/grad and optimizer update separately. "
                    "The first chunk can still take a while, but it avoids one oversized train-step HLO."
                )
            metrics = jnp.zeros((len(jax_losses.METRIC_KEYS),), dtype=jnp.float32)
            grad_norm = jnp.asarray(0.0, dtype=jnp.float32)
            for i, batch_np in enumerate(batch_list):
                batch_i = jax.device_put(batch_np)
                grads, moving_sum, moving_weight, metrics_i, grad_norm_i = loss_grad_step(
                    params,
                    batch_i,
                    moving_sum,
                    moving_weight,
                )
                params, opt_state = muon_update_step(
                    params,
                    opt_state,
                    grads,
                    jnp.asarray(opt_steps[i], dtype=jnp.float32),
                    jnp.asarray(lrs[i], dtype=jnp.float32),
                    jnp.asarray(wds[i], dtype=jnp.float32),
                )
                if args.ema_decay > 0:
                    ema_params = ema_update_step(ema_params, params)
                metrics = metrics + metrics_i
                grad_norm = grad_norm + grad_norm_i
        else:
            stacked_batch = stack_batch_list(batch_list)
            if args.data_parallel:
                batch = jax.device_put_sharded(
                    shard_stacked_batches_for_data_parallel(
                        stacked_batch,
                        num_data_parallel_devices,
                        per_device_batch,
                    ),
                    data_parallel_devices,
                )
                params, opt_state, ema_params, moving_sum, moving_weight, metrics, grad_norm = train_steps_dp(
                    params,
                    opt_state,
                    ema_params,
                    batch,
                    moving_sum,
                    moving_weight,
                    jnp.asarray(opt_steps),
                    jnp.asarray(lrs),
                    jnp.asarray(wds),
                )
                metrics = first_replica(metrics)
                grad_norm = first_replica(grad_norm)
            else:
                batch = jax.device_put(stacked_batch)
                params, opt_state, ema_params, moving_sum, moving_weight, metrics, grad_norm = train_steps(
                    params,
                    opt_state,
                    ema_params,
                    batch,
                    moving_sum,
                    moving_weight,
                    jnp.asarray(opt_steps),
                    jnp.asarray(lrs),
                    jnp.asarray(wds),
                )
        if args.log_step_time:
            metrics.block_until_ready()
            train_wait_elapsed = time.perf_counter() - exec_start
            total_elapsed = time.perf_counter() - chunk_start
            host_submit_elapsed = exec_start - chunk_start
            samples = args.batch_size * batch_count
            total_sps = samples / total_elapsed if total_elapsed > 0 else 0.0
            train_sps = samples / train_wait_elapsed if train_wait_elapsed > 0 else 0.0
            total_tflops = total_sps * train_flops_per_sample / 1e12
            train_tflops = train_sps * train_flops_per_sample / 1e12
            total_mfu = total_tflops / args.xla_peak_tflops * 100.0 if args.xla_peak_tflops > 0 else 0.0
            train_mfu = train_tflops / args.xla_peak_tflops * 100.0 if args.xla_peak_tflops > 0 else 0.0
            logging.info(
                "STEP_TIME steps=%d-%d samples=%d-%d chunk_steps=%d "
                "host_submit=%.4fs train_wait=%.4fs total=%.4fs "
                "per_step_total=%.4fs sps_total=%.1f sps_train=%.1f "
                "TFLOPS_total=%.2f TFLOPS_train=%.2f MFU_total=%.2f%% MFU_train=%.2f%%",
                start_step + 1,
                start_step + batch_count,
                start_samples + args.batch_size,
                start_samples + samples,
                batch_count,
                host_submit_elapsed,
                train_wait_elapsed,
                total_elapsed,
                total_elapsed / batch_count,
                total_sps,
                train_sps,
                total_tflops,
                train_tflops,
                total_mfu,
                train_mfu,
            )
        running_metrics = running_metrics + metrics
        running_grad_norm = running_grad_norm + grad_norm
        step += batch_count
        total_samples += args.batch_size * batch_count
        return float(lrs[-1]), float(wds[-1]), batch_count

    def after_training_chunk(lr, wd):
        nonlocal running_metrics, running_grad_norm
        nonlocal last_print_step, last_print_time, last_save_samples, last_val_samples

        if step - last_print_step >= args.print_every:
            metrics_host, grad_norm_host = jax.device_get((running_metrics, running_grad_norm))
            elapsed = time.perf_counter() - last_print_time
            batch_count = step - last_print_step
            weight_sum = max(float(metrics_host[-1]), 1e-10)
            sps = batch_count * args.batch_size / elapsed
            tflops = sps * train_flops_per_sample / 1e12
            mfu = tflops / args.xla_peak_tflops * 100.0 if args.xla_peak_tflops > 0 else 0.0
            by_key = dict(zip(jax_losses.METRIC_KEYS, metrics_host.tolist()))
            grad_norm_text = (
                f"{float(grad_norm_host) / batch_count:.3f}"
                if track_grad_norm else
                "off"
            )
            logging.info(
                "step=%d, samples=%d, time=%.1fs, lr=%.2e, wd=%.4f, "
                "loss=%.4f, p0loss=%.4f, vloss=%.4f, oloss=%.4f, skloss=%.4f, "
                "pacc1=%.4f, grad_norm=%s, sps=%.1f, TFLOPS=%.2f, MFU=%.2f%%",
                step,
                total_samples,
                elapsed,
                lr,
                wd,
                by_key["loss"] / batch_count,
                by_key["p0loss"] / weight_sum,
                by_key["vloss"] / weight_sum,
                by_key["oloss"] / weight_sum,
                by_key["skloss"] / weight_sum,
                by_key["pacc1"] / weight_sum,
                grad_norm_text,
                sps,
                tflops,
                mfu,
            )
            running_metrics = jnp.zeros_like(running_metrics)
            running_grad_norm = jnp.asarray(0.0, dtype=jnp.float32)
            last_print_step = step
            last_print_time = time.perf_counter()

        if total_samples - last_save_samples >= save_every:
            save_start = time.perf_counter()
            save_checkpoint(
                checkpoint_path,
                first_replica(params) if args.data_parallel else params,
                first_replica(opt_state) if args.data_parallel else opt_state,
                checkpoint_meta(),
                ema_params=(
                    first_replica(ema_params) if args.data_parallel else ema_params
                ) if args.ema_decay > 0 else None,
            )
            logging.info("Saved checkpoint at step %d, %d samples", step, total_samples)
            last_print_time += time.perf_counter() - save_start
            last_save_samples = total_samples

        if total_samples - last_val_samples >= val_every:
            val_start = time.perf_counter()
            run_validation()
            last_print_time += time.perf_counter() - val_start
            last_val_samples = total_samples

    while total_samples < args.max_training_samples:
        rand = np.random.default_rng(args.seed + step)
        shuffled = list(train_files)
        rand.shuffle(shuffled)
        pending_batches = []
        for batch_np in jax_data.read_npz_batches(
            shuffled,
            args.batch_size,
            args.pos_len,
            model_config,
            symmetry_type=args.symmetry_type,
            enable_history_matrices=args.enable_history_matrices,
            seed=args.seed + step,
            allow_nonfull_mask=args.allow_nonfull_mask,
        ):
            if total_samples >= args.max_training_samples:
                break
            pending_batches.append(batch_np)
            remaining_steps = (args.max_training_samples - total_samples) // args.batch_size
            if len(pending_batches) < min(args.steps_per_jit, remaining_steps):
                continue
            lr, wd, _ = run_training_chunk(pending_batches)
            pending_batches = []
            after_training_chunk(lr, wd)

        if pending_batches and total_samples < args.max_training_samples:
            lr, wd, _ = run_training_chunk(pending_batches)
            after_training_chunk(lr, wd)

        if not shuffled:
            break

    if not args.no_final_save and total_samples > last_save_samples:
        save_checkpoint(
            checkpoint_path,
            first_replica(params) if args.data_parallel else params,
            first_replica(opt_state) if args.data_parallel else opt_state,
            checkpoint_meta(),
            ema_params=(
                first_replica(ema_params) if args.data_parallel else ema_params
            ) if args.ema_decay > 0 else None,
        )
        logging.info("Saved checkpoint at step %d, %d samples", step, total_samples)
    elif args.no_final_save and total_samples > last_save_samples:
        logging.info("Final checkpoint skipped by --no-final-save")
    logging.info("Training complete: %d samples, %d steps", total_samples, step)


if __name__ == "__main__":
    main()
