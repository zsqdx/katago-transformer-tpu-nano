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


def init_adam_state(params):
    import jax.numpy as jnp

    zeros = _tree_map(lambda p: jnp.zeros_like(p), params)
    return {"m": zeros, "v": zeros}


def adamw_update(params, grads, state, step, lr, wd, beta1=0.9, beta2=0.95, eps=1e-8):
    import jax.numpy as jnp

    new_m = _tree_map(lambda m, g: beta1 * m + (1.0 - beta1) * g, state["m"], grads)
    new_v = _tree_map(lambda v, g: beta2 * v + (1.0 - beta2) * jnp.square(g), state["v"], grads)
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step

    def update_leaf(path, p, m, v):
        m_hat = m / bc1
        v_hat = v / bc2
        p_new = p - lr * (m_hat / (jnp.sqrt(v_hat) + eps))
        if _decay_mask_for_path(path):
            p_new = p_new - lr * wd * p
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
    return new_params, {"m": new_m, "v": new_v}


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


def save_checkpoint(path, params, opt_state, meta):
    import jax

    os.makedirs(os.path.dirname(path), exist_ok=True)
    host_state = {
        "params": jax.device_get(params),
        "opt_state": jax.device_get(opt_state),
        "meta": meta,
    }
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(host_state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


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
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--lr-schedule", type=str, default="cosine", choices=["cosine", "constant"])
    parser.add_argument("--max-training-samples", type=int, default=32768)
    parser.add_argument("--warmup-samples", type=int, default=4096)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--save-every-samples", type=int, default=0)
    parser.add_argument("--symmetry-type", type=str, default="xyt")
    parser.add_argument("--enable-history-matrices", action="store_true")
    parser.add_argument("--allow-nonfull-mask", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--score-mode", type=str, default="simple", choices=["simple"])
    parser.add_argument("--xla-peak-tflops", type=float, default=918.0)
    args = parser.parse_args()

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

    if args.model_kind not in configs.config_of_name:
        raise ValueError(f"Unknown model kind {args.model_kind}")
    model_config = configs.config_of_name[args.model_kind]
    logging.info("Model config: %s", json.dumps(model_config, indent=2))

    key = jax.random.PRNGKey(args.seed)
    params = jax_model.init_params(key, model_config, args.pos_len, init_std=args.init_std, score_mode=args.score_mode)
    params = jax.device_put(params)
    opt_state = jax.device_put(init_adam_state(params))
    rope_cache = tuple(jax.device_put(x) for x in jax_model.precompute_rope(
        model_config["hidden_size"] // model_config["num_heads"], args.pos_len
    ))
    score_offsets = jax.device_put(jax_model.score_belief_offsets(args.pos_len))
    moving_sum = jnp.asarray(0.0, dtype=jnp.float32)
    moving_weight = jnp.asarray(0.0, dtype=jnp.float32)
    td_value_loss_scales = (0.6, 0.6, 0.6)

    def loss_fn(params_, batch_, moving_sum_, moving_weight_):
        outputs = jax_model.forward(params_, batch_["binaryInputNCHW"], batch_["globalInputNC"],
                                    model_config, args.pos_len, rope_cache)
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
            is_training=True,
            td_value_loss_scales=td_value_loss_scales,
        )

    @jax.jit
    def train_step(params_, opt_state_, batch_, moving_sum_, moving_weight_, opt_step, lr, wd):
        def scalar_loss(p):
            loss, metrics, new_moving_sum, new_moving_weight = loss_fn(p, batch_, moving_sum_, moving_weight_)
            return loss, (metrics, new_moving_sum, new_moving_weight)

        (loss, (metrics, new_moving_sum, new_moving_weight)), grads = jax.value_and_grad(
            scalar_loss, has_aux=True
        )(params_)
        grad_norm = jnp.sqrt(_tree_sum_squares(grads))
        if args.grad_clip_norm > 0:
            scale = jnp.minimum(1.0, args.grad_clip_norm / (grad_norm + 1e-6))
            grads = _tree_map(lambda g: g * scale, grads)
        new_params, new_opt_state = adamw_update(params_, grads, opt_state_, opt_step, lr, wd)
        return new_params, new_opt_state, new_moving_sum, new_moving_weight, metrics, grad_norm

    train_files = jax_data.list_npz_files(args.datadir, "train")
    if not train_files:
        raise RuntimeError(f"No training files found in {os.path.join(args.datadir, 'train')}")
    logging.info("Training files: %d", len(train_files))

    samples_per_step = args.batch_size
    forward_flops = estimate_forward_flops(model_config, args.pos_len, args.score_mode)
    train_flops_per_sample = forward_flops * 3
    logging.info(
        "FLOPs/sample fwd=%.2fG train=%.2fG",
        forward_flops / 1e9,
        train_flops_per_sample / 1e9,
    )

    total_samples = 0
    step = 0
    running_metrics = jnp.zeros((len(jax_losses.METRIC_KEYS),), dtype=jnp.float32)
    running_grad_norm = jnp.asarray(0.0, dtype=jnp.float32)
    last_print_time = time.perf_counter()
    save_every = args.save_every_samples or args.max_training_samples
    last_save_samples = 0

    while total_samples < args.max_training_samples:
        rand = np.random.default_rng(args.seed + step)
        shuffled = list(train_files)
        rand.shuffle(shuffled)
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
            batch = jax.device_put(batch_np)
            lr, wd = lr_wd_at_step(step, args, samples_per_step)
            params, opt_state, moving_sum, moving_weight, metrics, grad_norm = train_step(
                params,
                opt_state,
                batch,
                moving_sum,
                moving_weight,
                jnp.asarray(step + 1, dtype=jnp.float32),
                jnp.asarray(lr, dtype=jnp.float32),
                jnp.asarray(wd, dtype=jnp.float32),
            )
            running_metrics = running_metrics + metrics
            running_grad_norm = running_grad_norm + grad_norm
            step += 1
            total_samples += args.batch_size

            if step % args.print_every == 0:
                metrics_host, grad_norm_host = jax.device_get((running_metrics, running_grad_norm))
                elapsed = time.perf_counter() - last_print_time
                batch_count = args.print_every
                weight_sum = max(float(metrics_host[-1]), 1e-10)
                sps = batch_count * args.batch_size / elapsed
                tflops = sps * train_flops_per_sample / 1e12
                mfu = tflops / args.xla_peak_tflops * 100.0 if args.xla_peak_tflops > 0 else 0.0
                by_key = dict(zip(jax_losses.METRIC_KEYS, metrics_host.tolist()))
                logging.info(
                    "step=%d, samples=%d, time=%.1fs, lr=%.2e, wd=%.4f, "
                    "loss=%.4f, p0loss=%.4f, vloss=%.4f, oloss=%.4f, skloss=%.4f, "
                    "pacc1=%.4f, grad_norm=%.3f, sps=%.1f, TFLOPS=%.2f, MFU=%.2f%%",
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
                    float(grad_norm_host) / batch_count,
                    sps,
                    tflops,
                    mfu,
                )
                running_metrics = jnp.zeros_like(running_metrics)
                running_grad_norm = jnp.asarray(0.0, dtype=jnp.float32)
                last_print_time = time.perf_counter()

            if total_samples - last_save_samples >= save_every:
                ckpt = os.path.join(args.traindir, "checkpoint_jax.pkl")
                save_checkpoint(ckpt, params, opt_state, {
                    "step": step,
                    "samples": total_samples,
                    "model_config": model_config,
                    "pos_len": args.pos_len,
                    "moving_sum": float(jax.device_get(moving_sum)),
                    "moving_weight": float(jax.device_get(moving_weight)),
                })
                logging.info("Saved checkpoint at step %d, %d samples", step, total_samples)
                last_save_samples = total_samples

        if not shuffled:
            break

    if total_samples > last_save_samples:
        ckpt = os.path.join(args.traindir, "checkpoint_jax.pkl")
        save_checkpoint(ckpt, params, opt_state, {
            "step": step,
            "samples": total_samples,
            "model_config": model_config,
            "pos_len": args.pos_len,
            "moving_sum": float(jax.device_get(moving_sum)),
            "moving_weight": float(jax.device_get(moving_weight)),
        })
        logging.info("Saved checkpoint at step %d, %d samples", step, total_samples)
    logging.info("Training complete: %d samples, %d steps", total_samples, step)


if __name__ == "__main__":
    main()
