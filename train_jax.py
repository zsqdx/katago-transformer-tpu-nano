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
    zeros = _tree_map(lambda p: jnp.zeros(p.shape, dtype=state_dtype), params)
    return {"m": zeros, "v": zeros}


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


def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def stack_batch_list(batch_list):
    return {
        key: np.stack([batch[key] for batch in batch_list], axis=0)
        for key in batch_list[0]
    }


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
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd", "none"])
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
    parser.add_argument("--log-grad-norm", action="store_true")
    parser.add_argument("--donate-train-buffers", action="store_true")
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

    rope_cache = tuple(jax.device_put(x) for x in jax_model.precompute_rope(
        model_config["hidden_size"] // model_config["num_heads"], args.pos_len
    ))
    score_offsets = jax.device_put(jax_model.score_belief_offsets(args.pos_len))
    checkpoint_path = os.path.join(args.traindir, "checkpoint_jax.pkl")

    moving_sum = jnp.asarray(0.0, dtype=jnp.float32)
    moving_weight = jnp.asarray(0.0, dtype=jnp.float32)
    total_samples = 0
    step = 0

    if not args.no_resume and os.path.exists(checkpoint_path):
        state = load_checkpoint(checkpoint_path)
        meta = state.get("meta", {})
        if meta.get("model_config") != model_config:
            raise ValueError(f"{checkpoint_path}: checkpoint model_config does not match {args.model_kind}")
        if int(meta.get("pos_len", args.pos_len)) != args.pos_len:
            raise ValueError(f"{checkpoint_path}: checkpoint pos_len does not match {args.pos_len}")
        state_params = state["params"]
        expected_stacked_blocks = args.stack_blocks or args.scan_blocks
        if isinstance(state_params.get("blocks"), dict) != expected_stacked_blocks:
            raise ValueError(f"{checkpoint_path}: checkpoint block layout does not match --stack-blocks/--scan-blocks")
        params = jax.device_put(_tree_map(lambda x: jnp.asarray(x, dtype=param_dtype), state_params))
        opt_state = jax.device_put(_tree_map(lambda x: jnp.asarray(x, dtype=opt_state_dtype), state["opt_state"]))
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
        params = jax.device_put(params)
        opt_state = jax.device_put(init_adam_state(params, dtype=opt_state_dtype))

    td_value_loss_scales = (0.6, 0.6, 0.6)

    def loss_fn(params_, batch_, moving_sum_, moving_weight_, is_training):
        outputs = jax_model.forward(params_, batch_["binaryInputNCHW"], batch_["globalInputNC"],
                                    model_config, args.pos_len, rope_cache,
                                    attention_impl=args.attention_impl,
                                    activation_dtype=activation_dtype,
                                    remat_blocks=args.remat_blocks,
                                    scan_blocks=args.scan_blocks)
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

    def train_one_step(params_, opt_state_, batch_, moving_sum_, moving_weight_, opt_step, lr, wd):
        def scalar_loss(p):
            loss, metrics, new_moving_sum, new_moving_weight = loss_fn(
                p, batch_, moving_sum_, moving_weight_, True
            )
            return loss, (metrics, new_moving_sum, new_moving_weight)

        (loss, (metrics, new_moving_sum, new_moving_weight)), grads = jax.value_and_grad(
            scalar_loss, has_aux=True
        )(params_)
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

    def train_steps_impl(params_, opt_state_, batches_, moving_sum_, moving_weight_, opt_steps, lrs, wds):
        def body(carry, xs):
            params_i, opt_state_i, moving_sum_i, moving_weight_i = carry
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
            )
            return (params_i, opt_state_i, moving_sum_i, moving_weight_i), (metrics_i, grad_norm_i)

        (params_, opt_state_, moving_sum_, moving_weight_), (metrics_seq, grad_norm_seq) = jax.lax.scan(
            body,
            (params_, opt_state_, moving_sum_, moving_weight_),
            (batches_, opt_steps, lrs, wds),
        )
        return (
            params_,
            opt_state_,
            moving_sum_,
            moving_weight_,
            jnp.sum(metrics_seq, axis=0),
            jnp.sum(grad_norm_seq),
        )

    train_steps = jax.jit(
        train_steps_impl,
        donate_argnums=(0,) if args.donate_train_buffers else (),
    )

    @jax.jit
    def eval_step(params_, batch_, moving_sum_, moving_weight_):
        _, metrics, _, _ = loss_fn(params_, batch_, moving_sum_, moving_weight_, False)
        return metrics

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

    running_metrics = jnp.zeros((len(jax_losses.METRIC_KEYS),), dtype=jnp.float32)
    running_grad_norm = jnp.asarray(0.0, dtype=jnp.float32)
    last_print_time = time.perf_counter()
    last_print_step = step
    save_every = args.save_every_samples or args.max_training_samples
    val_every = args.val_every_samples or args.max_training_samples
    last_save_samples = total_samples
    last_val_samples = total_samples

    def checkpoint_meta():
        return {
            "step": step,
            "samples": total_samples,
            "model_config": model_config,
            "pos_len": args.pos_len,
            "optimizer": args.optimizer,
            "moving_sum": float(jax.device_get(moving_sum)),
            "moving_weight": float(jax.device_get(moving_weight)),
            "fuse_projections": args.fuse_projections and not args.separate_projections,
            "attention_impl": args.attention_impl,
            "activation_dtype": args.activation_dtype,
            "param_dtype": args.param_dtype,
            "opt_state_dtype": args.opt_state_dtype,
            "opt_update_dtype": args.opt_update_dtype,
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
            batch = jax.device_put(batch_np)
            val_metrics = val_metrics + eval_step(params, batch, moving_sum, moving_weight)
            val_count += 1

        if val_count == 0:
            logging.warning("  VAL skipped: no full validation batches")
            return
        metrics_host = jax.device_get(val_metrics)
        log_metric_summary("  VAL", total_samples, metrics_host, val_count)

    def run_training_chunk(batch_list):
        nonlocal params, opt_state, moving_sum, moving_weight
        nonlocal running_metrics, running_grad_norm, step, total_samples
        batch_count = len(batch_list)
        batch = jax.device_put(stack_batch_list(batch_list))
        lr_wd = [lr_wd_at_step(step + i, args, samples_per_step) for i in range(batch_count)]
        lrs = np.asarray([x[0] for x in lr_wd], dtype=np.float32)
        wds = np.asarray([x[1] for x in lr_wd], dtype=np.float32)
        opt_steps = np.arange(step + 1, step + batch_count + 1, dtype=np.float32)
        params, opt_state, moving_sum, moving_weight, metrics, grad_norm = train_steps(
            params,
            opt_state,
            batch,
            moving_sum,
            moving_weight,
            jnp.asarray(opt_steps),
            jnp.asarray(lrs),
            jnp.asarray(wds),
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
            save_checkpoint(checkpoint_path, params, opt_state, checkpoint_meta())
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

    if total_samples > last_save_samples:
        save_checkpoint(checkpoint_path, params, opt_state, checkpoint_meta())
        logging.info("Saved checkpoint at step %d, %d samples", step, total_samples)
    logging.info("Training complete: %d samples, %d steps", total_samples, step)


if __name__ == "__main__":
    main()
