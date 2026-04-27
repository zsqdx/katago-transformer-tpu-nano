#!/usr/bin/python3
"""
Minimal Transformer training script for KataGo (nano version).
Self-contained — only depends on modules within nano/.
"""
import sys
import os
import argparse
import contextlib
import math
import time
import logging
import json
import glob
import signal
import threading
from datetime import timedelta
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
import torch.multiprocessing
from torch.nn.parallel import DistributedDataParallel
import atexit

import configs
import data as data_processing
from optimizers import MuonOptimizer, ShampooOptimizer
from zero import ZeROAdamW, ZeROMuon, ZeROShampoo, sync_zero_params, reduce_zero_grads, ZeROGradReducer
from losses import compute_loss, postprocess_and_loss_core, _METRIC_KEYS, estimate_forward_flops, get_gpu_peak_tflops


# ---------------------------------------------------------------------------
# Lightweight loss scaler for FP16 mixed-precision training.
# PyTorch's GradScaler couples unscale/step per-optimizer, which doesn't
# work cleanly with custom optimizers (Muon, Shampoo).  This class just
# manages a scalar scale factor with grow/backoff logic.
# ---------------------------------------------------------------------------
class SimpleGradScaler:
    def __init__(self, init_scale=2.**16, growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=2000):
        self._scale = init_scale
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._growth_tracker = 0

    def get_scale(self):
        return self._scale

    def scale(self, loss):
        return loss * self._scale

    def update(self, found_inf):
        if found_inf:
            self._scale *= self._backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker = 0

    def state_dict(self):
        return {"scale": self._scale, "growth_tracker": self._growth_tracker}

    def load_state_dict(self, state):
        self._scale = state["scale"]
        self._growth_tracker = state["growth_tracker"]


# ---------------------------------------------------------------------------
# Step profiler (CUDA sync + perf_counter per stage)
# ---------------------------------------------------------------------------
class StepProfiler:
    """Accumulates per-stage CUDA-synced timing across multiple steps."""

    def __init__(self, device):
        self._device = device
        self._accum = {}   # name -> total_seconds
        self._count = 0
        self._t = None

    def tick(self, name=None):
        torch.cuda.synchronize(self._device)
        now = time.perf_counter()
        if name is not None and self._t is not None:
            self._accum[name] = self._accum.get(name, 0.0) + (now - self._t)
        self._t = now

    def step_done(self):
        self._count += 1

    def report_and_reset(self):
        if self._count == 0:
            return None
        total = sum(self._accum.values())
        c = self._count
        parts = []
        for name, secs in self._accum.items():
            pct = secs / total * 100.0 if total > 0 else 0.0
            parts.append(f"{name}={secs/c*1000:.1f}ms({pct:.0f}%)")
        line = f"PROFILE step_ms={total/c*1000:.1f} | " + " ".join(parts)
        self._accum.clear()
        self._count = 0
        self._t = None
        return line


class _NullProfiler:
    """Zero-overhead drop-in when profiling is disabled."""
    def tick(self, name=None): pass
    def step_done(self): pass
    def report_and_reset(self): return None


def _import_xla():
    """Import PyTorch/XLA lazily so CUDA/CPU users do not need torch_xla."""
    import torch_xla
    import torch_xla.core.xla_model as xm
    return torch_xla, xm


def _get_xla_device(torch_xla_mod, xm):
    if hasattr(torch_xla_mod, "device"):
        return torch_xla_mod.device()
    return xm.xla_device()


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------
def multiprocessing_setup(rank: int, world_size: int, master_port: int = 23456):
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = str(master_port)
    logging.info(f"Running torch.distributed.init_process_group, rank={rank}, world_size={world_size}, "
                 f"master={os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=600))
    logging.info(f"Returned from init_process_group, rank={rank}")

def multiprocessing_cleanup():
    torch.distributed.destroy_process_group()


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average) of model parameters
# ---------------------------------------------------------------------------
class ModelEMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(p.data, 1.0 - self.decay)

    def state_dict(self):
        return {name: t.cpu() for name, t in self.shadow.items()}

    def load_state_dict(self, state, device):
        for name, t in state.items():
            if name in self.shadow:
                self.shadow[name].copy_(t.to(device))


# ---------------------------------------------------------------------------
# Training main loop
# ---------------------------------------------------------------------------
def main(rank, world_size, args, gpu_id):
    torch_xla_mod = None
    xm = None

    # Conditional model import
    if args.use_te:
        from model_te import (
            Model, detect_checkpoint_format,
            convert_checkpoint_model_to_te, convert_checkpoint_model_to_te_decomposed,
            convert_checkpoint_te_to_model, convert_checkpoint_te_decomposed_to_model,
        )
        model_extra_kwargs = {"use_fp8": args.use_fp8, "varlen": args.varlen,
                              "zero_centered_norm": args.zero_centered_norm, "hybrid": (args.use_te == 'hybrid'),
                              "learnable_rope": args.learnable_rope}
    else:
        from model import Model
        model_extra_kwargs = {"varlen": args.varlen, "gated_attn": args.gated_attn, "zero_centered_norm": args.zero_centered_norm, "learnable_rope": args.learnable_rope}

    # Parse td_value_loss_scales
    td_value_loss_scales = [float(x) for x in args.td_value_loss_scales.split(",")]
    assert len(td_value_loss_scales) == 3

    def apply_varlen_mode(varlen_value, source_name):
        args.varlen = varlen_value
        model_extra_kwargs["varlen"] = args.varlen

    # FP8 setup
    fp8_ctx_fn = contextlib.nullcontext
    if args.use_fp8:
        assert args.use_te, "--use-fp8 requires --use-te"
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import Float8CurrentScaling, Format
        fp8_recipe = Float8CurrentScaling(fp8_format=Format.HYBRID)
        fp8_ctx_fn = lambda: te.autocast(enabled=True, recipe=fp8_recipe)

    # Logging
    os.makedirs(args.traindir, exist_ok=True)
    logging.root.handlers = []
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(args.traindir, f"train{rank}.log"), mode="a"),
                logging.StreamHandler(),
            ],
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(args.traindir, f"train{rank}.log"), mode="a"),
            ],
        )
    logging.info(f"Args: {vars(args)}")

    if args.device == "xla" and world_size != 1:
        raise RuntimeError(
            "The current XLA path is single-device only. Use v6e-1 with one process "
            "for the initial smoke test."
        )

    # DDP init
    if world_size > 1:
        multiprocessing_setup(rank, world_size, args.master_port)
        atexit.register(multiprocessing_cleanup)

    # Random seed
    if args.seed is not None:
        seed = args.seed + rank  # different seed per rank for data diversity
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # Device selection
    if args.device == "xla":
        os.environ.setdefault("PJRT_DEVICE", "TPU")
        torch_xla_mod, xm = _import_xla()
        device = _get_xla_device(torch_xla_mod, xm)
    elif args.device in ("auto", "cuda") and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device("cuda", gpu_id)
    elif args.device == "cuda":
        raise RuntimeError("--device cuda was requested but CUDA is not available")
    elif args.device in ("auto", "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "mps":
        raise RuntimeError("--device mps was requested but MPS is not available")
    else:
        device = torch.device("cpu")
    logging.info(f"Device: {device}")
    if device.type == "cuda":
        logging.info(f"GPU: {torch.cuda.get_device_name()}")
    elif device.type == "xla":
        try:
            logging.info(f"XLA device: {xm.xla_device_hw(device)}")
        except Exception:
            logging.info("XLA device selected")

    if device.type == "xla" and args.prefetch_batches > 0:
        logging.warning("Disabling threaded prefetch on XLA for the initial TPU path")
        args.prefetch_batches = 0

    def sync_xla_if_needed():
        if torch_xla_mod is None:
            return
        if hasattr(torch_xla_mod, "sync"):
            torch_xla_mod.sync()
        else:
            xm.mark_step()

    # Step profiler
    profiler = StepProfiler(device) if args.profile else _NullProfiler()
    if args.profile:
        logging.info("PROFILE mode enabled - cuda.synchronize() between every stage (adds overhead)")

    # AMP setup
    grad_scaler = None
    if device.type == "cuda":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        amp_device = "cuda"
        if args.amp_dtype == "fp16":
            amp_dtype = torch.float16
            use_amp = True
            grad_scaler = SimpleGradScaler()
        elif args.amp_dtype == "none":
            amp_dtype = torch.bfloat16  # placeholder, not used
            use_amp = False
        else:  # bf16 (default)
            amp_dtype = torch.bfloat16
            use_amp = True
    elif device.type == "mps":
        amp_device = "mps"
        amp_dtype = torch.bfloat16
        use_amp = args.amp_dtype != "none"
    elif device.type == "xla":
        amp_device = "xla"
        amp_dtype = torch.bfloat16
        use_amp = args.amp_dtype != "none"
    else:
        amp_device = "cpu"
        amp_dtype = torch.bfloat16
        use_amp = False

    # Model config
    if args.num_layers is not None:
        model_config = configs.make_config(
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
        )
    else:
        model_config = configs.config_of_name[args.model_kind].copy()
    logging.info(f"Model config: {json.dumps(model_config, indent=2, default=str)}")

    pos_len = args.pos_len
    batch_size = args.batch_size

    # Load or create model
    checkpoint_path = os.path.join(args.traindir, "checkpoint.ckpt")
    if os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint: {checkpoint_path}")
        for _attempt in range(3):
            try:
                state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                break
            except RuntimeError as e:
                if _attempt < 2 and "corrupted" in str(e):
                    logging.warning(f"Checkpoint load attempt {_attempt+1} failed ({e}), retrying...")
                    import time as _time; _time.sleep(2)
                else:
                    raise
        ckpt_config = configs.migrate_config(state.get("config", model_config))
        if ckpt_config != model_config:
            logging.warning(f"Checkpoint config differs from command-line config, using checkpoint config")
            logging.warning(f"  checkpoint: {ckpt_config}")
            logging.warning(f"  command-line: {model_config}")
        model_config = ckpt_config
        # Restore varlen flag from checkpoint
        ckpt_varlen = state.get("varlen", False)
        if ckpt_varlen != args.varlen:
            logging.warning(f"Checkpoint varlen={ckpt_varlen} differs from command-line --varlen={args.varlen}, using checkpoint value")
        apply_varlen_mode(ckpt_varlen, "Checkpoint")
        # Verify gated_attn flag matches checkpoint
        ckpt_gated_attn = state.get("gated_attn", False)
        if ckpt_gated_attn != args.gated_attn:
            raise RuntimeError(
                f"Checkpoint gated_attn={ckpt_gated_attn} differs from --gated-attn={args.gated_attn}; "
                "rerun with a matching flag"
            )
        # Verify zero_centered_norm flag matches checkpoint
        ckpt_zero_centered_norm = state.get("zero_centered_norm", False)
        if ckpt_zero_centered_norm != args.zero_centered_norm:
            raise RuntimeError(
                f"Checkpoint zero_centered_norm={ckpt_zero_centered_norm} differs from "
                f"--zero-centered-norm={args.zero_centered_norm}; rerun with a matching flag"
            )
        model = Model(model_config, pos_len, score_mode=args.score_mode, **model_extra_kwargs)
        model_state = state["model"]
        if args.use_te:
            fmt = detect_checkpoint_format(model_state)
            if args.use_te == 'hybrid':
                if fmt == "pt":
                    logging.info("Converting model.py checkpoint to hybrid TE format")
                    model_state = convert_checkpoint_model_to_te_decomposed(model_state)
                elif fmt == "te":
                    logging.info("Converting full TE checkpoint to hybrid TE format")
                    model_state = convert_checkpoint_te_to_model(model_state)
                    model_state = convert_checkpoint_model_to_te_decomposed(model_state)
            else:  # full
                if fmt == "pt":
                    logging.info("Converting model.py checkpoint to TE format")
                    model_state = convert_checkpoint_model_to_te(model_state)
                elif fmt == "te_decomposed":
                    logging.info("Converting hybrid TE checkpoint to full TE format")
                    model_state = convert_checkpoint_te_decomposed_to_model(model_state)
                    model_state = convert_checkpoint_model_to_te(model_state)
            model.load_state_dict(model_state, strict=False)
        else:
            try:
                from model_te import detect_checkpoint_format as _detect, \
                    convert_checkpoint_te_to_model as _convert, \
                    convert_checkpoint_te_decomposed_to_model as _convert_decomp
                fmt = _detect(model_state)
                if fmt == "te":
                    logging.info("Converting TE checkpoint to model.py format")
                    model_state = _convert(model_state, zero_centered_norm=args.zero_centered_norm)
                elif fmt == "te_decomposed":
                    logging.info("Converting hybrid TE checkpoint to model.py format")
                    model_state = _convert_decomp(model_state, zero_centered_norm=args.zero_centered_norm)
            except ImportError:
                pass
            model.load_state_dict(model_state)
        model.moving_unowned_proportion_sum = state.get("moving_unowned_proportion_sum", 0.0)
        model.moving_unowned_proportion_weight = state.get("moving_unowned_proportion_weight", 0.0)
        global_step = state.get("global_step", 0)
        if "total_samples_trained" in state:
            total_samples_trained = state["total_samples_trained"]
        else:
            total_samples_trained = global_step * batch_size * world_size * args.grad_accum_steps
            logging.warning(
                f"Checkpoint missing total_samples_trained, estimated from current settings: {total_samples_trained}. "
                f"This may be inaccurate if batch_size/world_size/grad_accum_steps changed."
            )
        logging.info(f"Resumed from step {global_step}, {total_samples_trained} samples")
    elif args.initial_checkpoint is not None:
        logging.info(f"Loading initial checkpoint: {args.initial_checkpoint}")
        state = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
        model_config = configs.migrate_config(state.get("config", model_config))
        ckpt_has_varlen = "varlen" in state
        ckpt_varlen = state.get("varlen", False)
        if ckpt_has_varlen and ckpt_varlen != args.varlen:
            raise RuntimeError(
                f"Initial checkpoint varlen={ckpt_varlen} differs from --varlen={args.varlen}; "
                "rerun with a matching flag"
            )
        if ckpt_has_varlen:
            apply_varlen_mode(ckpt_varlen, "Initial checkpoint")
        # Verify gated_attn flag matches initial checkpoint
        ckpt_gated_attn = state.get("gated_attn", False)
        if ckpt_gated_attn != args.gated_attn:
            raise RuntimeError(
                f"Initial checkpoint gated_attn={ckpt_gated_attn} differs from --gated-attn={args.gated_attn}; "
                "rerun with a matching flag"
            )
        # Verify zero_centered_norm flag matches initial checkpoint
        ckpt_zero_centered_norm = state.get("zero_centered_norm", False)
        if ckpt_zero_centered_norm != args.zero_centered_norm:
            raise RuntimeError(
                f"Initial checkpoint zero_centered_norm={ckpt_zero_centered_norm} differs from "
                f"--zero-centered-norm={args.zero_centered_norm}; rerun with a matching flag"
            )
        model = Model(model_config, pos_len, score_mode=args.score_mode, **model_extra_kwargs)
        model_state = state["model"]
        if args.use_te:
            fmt = detect_checkpoint_format(model_state)
            if args.use_te == 'hybrid':
                if fmt == "pt":
                    logging.info("Converting model.py initial checkpoint to hybrid TE format")
                    model_state = convert_checkpoint_model_to_te_decomposed(model_state)
                elif fmt == "te":
                    logging.info("Converting full TE initial checkpoint to hybrid TE format")
                    model_state = convert_checkpoint_te_to_model(model_state)
                    model_state = convert_checkpoint_model_to_te_decomposed(model_state)
            else:  # full
                if fmt == "pt":
                    logging.info("Converting model.py initial checkpoint to TE format")
                    model_state = convert_checkpoint_model_to_te(model_state)
                elif fmt == "te_decomposed":
                    logging.info("Converting hybrid TE initial checkpoint to full TE format")
                    model_state = convert_checkpoint_te_decomposed_to_model(model_state)
                    model_state = convert_checkpoint_model_to_te(model_state)
            model.load_state_dict(model_state, strict=False)
        else:
            try:
                from model_te import detect_checkpoint_format as _detect, \
                    convert_checkpoint_te_to_model as _convert, \
                    convert_checkpoint_te_decomposed_to_model as _convert_decomp
                fmt = _detect(model_state)
                if fmt == "te":
                    logging.info("Converting TE initial checkpoint to model.py format")
                    model_state = _convert(model_state, zero_centered_norm=args.zero_centered_norm)
                elif fmt == "te_decomposed":
                    logging.info("Converting hybrid TE initial checkpoint to model.py format")
                    model_state = _convert_decomp(model_state, zero_centered_norm=args.zero_centered_norm)
            except ImportError:
                pass
            result = model.load_state_dict(model_state, strict=False)
            if result.unexpected_keys:
                logging.warning(f"Unexpected keys in initial checkpoint (ignored): {result.unexpected_keys}")
            if result.missing_keys:
                logging.warning(f"Missing keys in initial checkpoint: {result.missing_keys}")
        global_step = 0
        total_samples_trained = 0
    else:
        logging.info("Creating new model")
        model = Model(model_config, pos_len, score_mode=args.score_mode, **model_extra_kwargs)
        model.initialize(init_std=args.init_std)
        logging.info(f"Initialized weights: std={args.init_std}, output scaling /sqrt(2*{len(model.blocks)})")
        global_step = 0
        total_samples_trained = 0

    model.to(device)

    # EMA (shadow copy of params on same device, before compile/DDP)
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None:
        logging.info(f"EMA enabled: decay={args.ema_decay}, shadow params: {sum(t.numel() for t in ema.shadow.values()):,}")

    # torch.compile
    if not args.no_compile and device.type not in ("mps", "xla"):
        compiled_model = torch.compile(model, mode="default")
        compiled_loss_fn = torch.compile(postprocess_and_loss_core, mode="reduce-overhead")
    else:
        compiled_model = model
        compiled_loss_fn = postprocess_and_loss_core

    # DDP wrapper (skipped in zero dp-mode where we reduce gradients manually)
    dp_zero = (args.dp_mode == "zero") and world_size > 1
    if device.type == "xla" and world_size > 1:
        raise RuntimeError("XLA multi-device training is not wired up yet; start with v6e-1 single-device testing")
    if world_size > 1 and not dp_zero:
        ddp_model = DistributedDataParallel(compiled_model, device_ids=[device])
    else:
        ddp_model = compiled_model

    # Parameter stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # Optimizer: split params into Muon / Shampoo / Adam groups
    muon_params = {}
    shampoo_params = {}
    adam_params = {}
    no_decay_params = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or name.endswith("layer_norm_weight") or name.endswith("layer_norm_bias"):
            # Zero-centered norm weights need weight decay (pushes weight→0, gamma→1)
            if args.zero_centered_norm and "norm" in name and "weight" in name:
                adam_params[name] = p
            else:
                no_decay_params[name] = p
        elif "rope_freqs" in name:
            no_decay_params[name] = p
        elif args.optimizer == "muon" and "blocks." in name:
            muon_params[name] = p
        elif args.optimizer == "shampoo" and "blocks." in name:
            shampoo_params[name] = p
        else:
            adam_params[name] = p

    logging.info(f"Muon params: {sum(p.numel() for p in muon_params.values()):,}, "
                 f"Shampoo params: {sum(p.numel() for p in shampoo_params.values()):,}, "
                 f"Adam decay: {sum(p.numel() for p in adam_params.values()):,}, "
                 f"AdamW no-decay: {sum(p.numel() for p in no_decay_params.values()):,}")
    for name, p in muon_params.items():
        logging.info(f"  [Muon] {name}: {list(p.shape)}")
    for name, p in shampoo_params.items():
        logging.info(f"  [Shampoo] {name}: {list(p.shape)}")
    for name, p in adam_params.items():
        logging.info(f"  [Adam] {name}: {list(p.shape)}")
    for name, p in no_decay_params.items():
        logging.info(f"  [NoDecay] {name}: {list(p.shape)}")

    # FLOPs estimation
    forward_flops = estimate_forward_flops(model_config, pos_len, score_mode=args.score_mode)
    train_flops_per_sample = 3 * forward_flops
    gpu_peak_tflops = get_gpu_peak_tflops(device)
    logging.info(f"FLOPs/sample (fwd): {forward_flops/1e9:.2f}G, (train): {train_flops_per_sample/1e9:.2f}G")
    if gpu_peak_tflops > 0:
        logging.info(f"GPU BF16 peak: {gpu_peak_tflops:.1f} TFLOPS")

    num_heads = model_config["num_heads"]

    # LR/WD schedule
    grad_accum_steps = args.grad_accum_steps
    samples_per_step = batch_size * world_size * grad_accum_steps
    use_hzy_schedule = (args.lr_schedule == "hzy")

    if use_hzy_schedule:
        # HZY step-function LR/WD schedule (from lr_schedule.xlsx)
        _LR_WD_SCHEDULE = [
            (0,           1.131371e-4, 0.377124),
            (5_000_000,   3.200000e-4, 1.066667),
            (100_000_000, 2.262742e-4, 0.754247),
            (200_000_000, 1.600000e-4, 0.533333),
            (300_000_000, 1.131371e-4, 0.377124),
            (400_000_000, 8.000000e-5, 0.266667),
            (500_000_000, 5.656854e-5, 0.188562),
            (550_000_000, 4.000000e-5, 0.133333),
            (600_000_000, 2.828427e-5, 0.094281),
            (650_000_000, 2.000000e-5, 0.066667),
            (675_000_000, 1.414214e-5, 0.047140),
        ]

        def get_lr_wd(total_samples):
            lr, wd = _LR_WD_SCHEDULE[0][1], _LR_WD_SCHEDULE[0][2]
            for s, lr_s, wd_s in _LR_WD_SCHEDULE:
                if total_samples >= s:
                    lr, wd = lr_s, wd_s
                else:
                    break
            return lr, wd

        base_lr, base_wd = get_lr_wd(total_samples_trained)
        logging.info(f"HZY LR/WD schedule: {len(_LR_WD_SCHEDULE)} stages, "
                     f"current lr={base_lr:.2e}, wd={base_wd:.4f} at {total_samples_trained} samples")
    else:
        base_lr = args.lr
        base_wd = args.wd

    # Optimizers: ZeRO Stage 1 when multi-GPU, plain otherwise
    if world_size > 1:
        zero_adam = ZeROAdamW(
            adam_params, no_decay_params, lr=base_lr, betas=(0.9, 0.95),
            wd=base_wd, device=device, rank=rank, world_size=world_size,
        )
        inner_optimizer = zero_adam.optimizer
        muon_opt = ZeROMuon(
            muon_params, lr_multiplier=0.2,
            momentum=0.95, wd=base_wd,
            device=device, rank=rank, world_size=world_size, use_te=bool(args.use_te),
            num_heads=num_heads,
        ) if muon_params else None
        shampoo_opt = ZeROShampoo(
            shampoo_params, lr_multiplier=args.shampoo_lr_multiplier,
            momentum=0.9, wd=base_wd, beta2=0.95,
            device=device, rank=rank, world_size=world_size, use_te=bool(args.use_te),
            num_heads=num_heads,
        ) if shampoo_params else None
    else:
        zero_adam = None
        adam_param_groups = [
            {"params": list(no_decay_params.values()), "weight_decay": 0.0},
        ]
        if adam_params:
            adam_param_groups.append({"params": list(adam_params.values()), "weight_decay": base_wd})
        inner_optimizer = torch.optim.AdamW(adam_param_groups, lr=base_lr, betas=(0.9, 0.95), fused=(device.type == "cuda"))
        muon_opt = MuonOptimizer(
            muon_params, lr_multiplier=0.2,
            momentum=0.95, wd=base_wd, device=device, use_te=bool(args.use_te),
            num_heads=num_heads,
        ) if muon_params else None
        shampoo_opt = ShampooOptimizer(
            shampoo_params, lr_multiplier=args.shampoo_lr_multiplier,
            momentum=0.9, wd=base_wd, beta2=0.95, device=device,
            use_te=bool(args.use_te), num_heads=num_heads,
        ) if shampoo_params else None

    # Restore optimizer state
    is_zero = zero_adam is not None
    if os.path.exists(checkpoint_path):
        # Assert training mode consistency — no switching ZeRO/non-ZeRO or optimizer config on resume
        if "training_mode" in state:
            saved_mode = state["training_mode"]
            assert saved_mode["zero"] == is_zero, (
                f"Cannot switch between ZeRO and non-ZeRO mode on resume. "
                f"Checkpoint: zero={saved_mode['zero']}, current: zero={is_zero}"
            )
            assert saved_mode["has_muon"] == (muon_opt is not None), (
                f"Cannot change optimizer config (Muon) on resume. "
                f"Checkpoint: has_muon={saved_mode['has_muon']}, current: has_muon={muon_opt is not None}"
            )
            assert saved_mode["has_shampoo"] == (shampoo_opt is not None), (
                f"Cannot change optimizer config (Shampoo) on resume. "
                f"Checkpoint: has_shampoo={saved_mode['has_shampoo']}, current: has_shampoo={shampoo_opt is not None}"
            )
        else:
            # Legacy checkpoint (pre-ZeRO): assume non-ZeRO
            assert not is_zero, (
                "Cannot resume a legacy (non-ZeRO) checkpoint in ZeRO mode. "
                "Start fresh or re-run with single GPU."
            )

        if "optimizer" in state:
            if is_zero:
                zero_adam.load_state_distributed(state["optimizer"], device)
            else:
                inner_optimizer.load_state_dict(state["optimizer"])
            logging.info("Optimizer state loaded")
        if "muon_state" in state and muon_opt is not None:
            if is_zero:
                muon_opt.load_state_distributed(state["muon_state"], device)
            else:
                muon_opt.load_state_dict(state["muon_state"], device)
            logging.info("Muon state loaded")
        elif "muon_bufs" in state and muon_opt is not None:
            # Legacy Muon format (momentum-only)
            for k, v in state["muon_bufs"].items():
                if k in muon_opt.states:
                    muon_opt.states[k]["momentum"].copy_(v.to(device))
            logging.info("Muon momentum buffers loaded (legacy format)")
        if "shampoo_state" in state and shampoo_opt is not None:
            if is_zero:
                shampoo_opt.load_state_distributed(state["shampoo_state"], device)
            else:
                shampoo_opt.load_state_dict(state["shampoo_state"], device)
            logging.info("Shampoo state loaded")
        if "ema_shadow" in state and ema is not None:
            ema.load_state_dict(state["ema_shadow"], device)
            logging.info("EMA state loaded")
        elif ema is not None:
            logging.info("No EMA state in checkpoint, initialized from current params")
        if "grad_scaler" in state and grad_scaler is not None:
            grad_scaler.load_state_dict(state["grad_scaler"])
            logging.info(f"GradScaler state loaded (scale={grad_scaler.get_scale():.1f})")
    # ZeRO gradient reducer: overlap reduce with backward
    if dp_zero and args.overlap_reduce:
        grad_reducer = ZeROGradReducer(
            [zero_adam, muon_opt, shampoo_opt], model, rank=rank, world_size=world_size,
        )
    else:
        grad_reducer = None

    # Cosine LR schedule (when not using HZY)
    scheduler = None
    if not use_hzy_schedule:
        warmup_steps = args.warmup_samples // samples_per_step
        total_steps = args.max_training_samples // samples_per_step

        if global_step > 0:
            saved_sps = state.get("samples_per_step")
            if saved_sps is not None and saved_sps != samples_per_step:
                logging.warning(
                    f"samples_per_step changed: {saved_sps} -> {samples_per_step} "
                    f"(batch_size*world_size*grad_accum_steps). "
                    f"LR schedule may have a discontinuity."
                )

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        if global_step > 0:
            for pg in inner_optimizer.param_groups:
                pg["lr"] = args.lr
                pg["initial_lr"] = args.lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(inner_optimizer, lr_lambda, last_epoch=global_step - 1 if global_step > 0 else -1)

    # Data directories
    train_dir = os.path.join(args.datadir, "train")
    val_dir = os.path.join(args.datadir, "val")

    # TensorBoard (rank 0 only)
    tb_writer = None
    if rank == 0 and not args.no_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(args.traindir, "tb_logs")
            tb_writer = SummaryWriter(log_dir=tb_dir)
            logging.info(f"TensorBoard: {tb_dir}")
        except Exception as e:
            logging.warning(f"TensorBoard disabled: {type(e).__name__}: {e}")
    elif rank == 0:
        logging.info("TensorBoard disabled by --no-tensorboard")

    # Save checkpoint (all ranks must call when ZeRO is active — gather is collective)
    _async_save_thread = None

    def save_checkpoint(async_save=True):
        nonlocal _async_save_thread
        # Wait for previous async save to complete
        if _async_save_thread is not None:
            _async_save_thread.join()
            _async_save_thread = None

        # Gather optimizer states (collective operations — all ranks must participate)
        if zero_adam is not None:
            adam_state_gathered = zero_adam.gather_state_for_save()
        else:
            adam_state_gathered = None

        if muon_opt is not None and zero_adam is not None:
            muon_state_gathered = muon_opt.gather_state_for_save()
        else:
            muon_state_gathered = None

        if shampoo_opt is not None and zero_adam is not None:
            shampoo_state_gathered = shampoo_opt.gather_state_for_save()
        else:
            shampoo_state_gathered = None

        # Only rank 0 assembles state_dict and writes the file
        if rank == 0:
            sync_xla_if_needed()
            # Deep-copy model weights to CPU (training continues modifying GPU tensors)
            model_state_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
            state_dict = {
                "model": model_state_cpu,
                "config": model_config,
                "global_step": global_step,
                "total_samples_trained": total_samples_trained,
                "samples_per_step": samples_per_step,
                "moving_unowned_proportion_sum": model.moving_unowned_proportion_sum,
                "moving_unowned_proportion_weight": model.moving_unowned_proportion_weight,
                "varlen": args.varlen,
                "gated_attn": args.gated_attn,
                "head_bias": True,
                "norm_fp32": True,
                "zero_centered_norm": args.zero_centered_norm,
                "training_mode": {
                    "zero": zero_adam is not None,
                    "has_muon": muon_opt is not None,
                    "has_shampoo": shampoo_opt is not None,
                },
            }
            if zero_adam is not None:
                state_dict["optimizer"] = adam_state_gathered  # already on CPU from gather
            else:
                # fused AdamW keeps state on GPU — copy to CPU
                sd = inner_optimizer.state_dict()
                state_dict["optimizer"] = {
                    "state": {k: {sk: sv.cpu() if torch.is_tensor(sv) else sv
                                  for sk, sv in v.items()} for k, v in sd["state"].items()},
                    "param_groups": sd["param_groups"],
                }
            if muon_opt is not None:
                state_dict["muon_state"] = muon_state_gathered if muon_state_gathered is not None else muon_opt.state_dict()
            if shampoo_opt is not None:
                state_dict["shampoo_state"] = shampoo_state_gathered if shampoo_state_gathered is not None else shampoo_opt.state_dict()
            if ema is not None:
                state_dict["ema_shadow"] = ema.state_dict()
            if grad_scaler is not None:
                state_dict["grad_scaler"] = grad_scaler.state_dict()

            # Capture values for logging in background thread
            _step, _samples = global_step, total_samples_trained

            def _do_save():
                path = os.path.join(args.traindir, "checkpoint.ckpt")
                tmp_path = path + ".tmp"
                torch.save(state_dict, tmp_path)
                # fsync to ensure data is persisted before replacing (critical on DFS)
                with open(tmp_path, "ab") as f:
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, path)
                # Keep a numbered copy (use copy instead of hardlink for independence)
                numbered = os.path.join(args.traindir, f"checkpoint-s{_samples}.ckpt")
                if not os.path.exists(numbered):
                    import shutil
                    shutil.copy2(path, numbered)
                logging.info(f"Saved checkpoint at step {_step}, {_samples} samples")

            if async_save:
                _async_save_thread = threading.Thread(target=_do_save, daemon=True)
                _async_save_thread.start()
            else:
                _do_save()

    # Metrics accumulation
    _metric_keys = [
        "loss", "p0loss", "p1loss", "p0softloss", "p1softloss",
        "p0lopt", "p0sopt",
        "vloss", "tdvloss1", "tdvloss2", "tdvloss3", "tdsloss",
        "oloss", "sloss", "fploss", "skloss",
        "smloss", "sbcdfloss", "sbpdfloss", "sdregloss",
        "leadloss", "vtimeloss", "evstloss", "esstloss",
        "pacc1", "wsum",
    ]
    running = {k: 0.0 for k in _metric_keys}
    running["count"] = 0
    running["grad_norm"] = 0.0
    running["muon_update_rms"] = 0.0
    running["shampoo_precond_rms"] = 0.0

    def reset_running():
        for k in running:
            running[k] = 0.0

    _per_sample_keys = [k for k in _metric_keys if k not in ("loss", "wsum")]

    def print_metrics(elapsed):
        weight_sum = max(running["wsum"], 1e-10)
        batch_count = max(running["count"], 1)
        samples_per_sec = batch_count * batch_size * world_size * grad_accum_steps / elapsed
        achieved_tflops_per_gpu = samples_per_sec * train_flops_per_sample / (world_size * 1e12)
        mfu = achieved_tflops_per_gpu / gpu_peak_tflops * 100.0 if gpu_peak_tflops > 0 else 0.0
        logging.info(
            f"step={global_step}, samples={total_samples_trained}, "
            f"time={elapsed:.1f}s, "
            f"lr={base_lr:.2e}, wd={base_wd:.4f}, "
            f"loss={running['loss'] / batch_count:.4f}, "
            f"p0loss={running['p0loss'] / weight_sum:.4f}, "
            f"vloss={running['vloss'] / weight_sum:.4f}, "
            f"oloss={running['oloss'] / weight_sum:.4f}, "
            f"skloss={running['skloss'] / weight_sum:.4f}, "
            f"pacc1={running['pacc1'] / weight_sum:.4f}, "
            f"sps={samples_per_sec:.0f}, MFU={mfu:.1f}%"
        )
        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", running["loss"] / batch_count, total_samples_trained)
            for k in _per_sample_keys:
                tb_writer.add_scalar(f"train/{k}", running[k] / weight_sum, total_samples_trained)
            tb_writer.add_scalar("train/lr", base_lr, total_samples_trained)
            tb_writer.add_scalar("train/wd", base_wd, total_samples_trained)
            tb_writer.add_scalar("train/grad_norm", running["grad_norm"] / batch_count, total_samples_trained)
            if muon_opt is not None:
                tb_writer.add_scalar("train/muon_update_rms", running["muon_update_rms"] / batch_count, total_samples_trained)
            if shampoo_opt is not None:
                tb_writer.add_scalar("train/shampoo_precond_rms", running["shampoo_precond_rms"] / batch_count, total_samples_trained)
            tokens_per_sec = samples_per_sec * pos_len * pos_len
            tb_writer.add_scalar("perf/samples_per_sec", samples_per_sec, total_samples_trained)
            tb_writer.add_scalar("perf/tokens_per_sec", tokens_per_sec, total_samples_trained)
            tb_writer.add_scalar("perf/achieved_tflops_per_gpu", achieved_tflops_per_gpu, total_samples_trained)
            tb_writer.add_scalar("perf/mfu", mfu, total_samples_trained)
            if grad_scaler is not None:
                tb_writer.add_scalar("train/grad_scale", grad_scaler.get_scale(), total_samples_trained)
        profile_line = profiler.report_and_reset()
        if profile_line is not None:
            logging.info(f"  {profile_line}")

    # Start training
    effective_batch = batch_size * world_size * grad_accum_steps
    logging.info("=" * 60)
    logging.info(f"Starting training: {total_samples_trained}/{args.max_training_samples} samples done")
    logging.info(f"Effective batch size: {effective_batch} (micro={batch_size} x gpus={world_size} x accum={grad_accum_steps})")
    logging.info(f"AMP: dtype={args.amp_dtype}, GradScaler={'yes (scale=%.1f)' % grad_scaler.get_scale() if grad_scaler is not None else 'no'}")
    logging.info("=" * 60)

    last_save_samples = total_samples_trained
    last_val_samples = total_samples_trained
    reset_running()
    last_print_time = time.perf_counter()
    accum_step = 0
    micro_metrics_accum = {k: 0.0 for k in _metric_keys}
    accum_moving_sum = 0.0
    accum_moving_weight = 0.0
    _last_spatial = None
    _last_global = None

    while total_samples_trained < args.max_training_samples:
        model.train()

        # Discard partial gradient accumulation from previous file list
        if accum_step > 0:
            logging.info(f"Discarding partial gradient accumulation ({accum_step}/{grad_accum_steps} micro-steps)")
            accum_step = 0
            for k in _metric_keys:
                micro_metrics_accum[k] = 0.0
            accum_moving_sum = 0.0
            accum_moving_weight = 0.0

        # Find training files
        train_files = sorted(glob.glob(os.path.join(train_dir, "*.npz")))
        if not train_files:
            logging.warning(f"No training files found in {train_dir}, waiting...")
            time.sleep(10)
            continue

        np.random.shuffle(train_files)

        use_pin_memory = args.prefetch_batches > 0 and device.type == "cuda"
        train_gen = data_processing.read_npz_training_data(
            train_files,
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            pos_len=pos_len,
            device=device,
            symmetry_type=args.symmetry_type,
            include_meta=False,
            enable_history_matrices=args.enable_history_matrices,
            model_config=model_config,
            use_pin_memory=use_pin_memory,
            seed=args.seed + rank if args.seed is not None else None,
            varlen=args.varlen,
            allow_nonfull_mask=args.allow_nonfull_mask,
        )
        profiler.tick()
        for batch in data_processing.prefetch_generator(train_gen, args.prefetch_batches):
            profiler.tick("data")

            # Clear gradients at the start of each accumulation cycle
            if accum_step == 0:
                for p in model.parameters():
                    p.grad = None

            # DDP no_sync: skip all-reduce for intermediate micro-steps.
            # In zero dp-mode there is no DDP, so no_sync is unnecessary.
            is_last_micro = (accum_step + 1 == grad_accum_steps)
            if dp_zero:
                ctx = contextlib.nullcontext()
            else:
                ctx = contextlib.nullcontext() if (is_last_micro or world_size == 1) else ddp_model.no_sync()

            _last_spatial = batch["binaryInputNCHW"]
            _last_global = batch["globalInputNC"]

            with ctx:
                with torch.amp.autocast(amp_device, dtype=amp_dtype, enabled=use_amp):
                    with fp8_ctx_fn():
                        outputs = ddp_model(batch["binaryInputNCHW"], batch["globalInputNC"])
                profiler.tick("fwd")

                # Compiled postprocess + loss (seki moving average computed inside as tensor ops)
                moving_sum_t = torch.tensor(model.moving_unowned_proportion_sum, device=device)
                moving_weight_t = torch.tensor(model.moving_unowned_proportion_weight, device=device)
                train_mask = batch["binaryInputNCHW"][:, 0:1, :, :].contiguous() if args.varlen else None
                loss, metrics_stack, new_moving_sum, new_moving_weight = compiled_loss_fn(
                    outputs, model.value_head.score_belief_offset_vector,
                    batch["policyTargetsNCMove"],
                    batch["globalTargetsNC"], batch["scoreDistrN"], batch["valueTargetsNCHW"],
                    pos_len, moving_sum_t, moving_weight_t, True,
                    soft_policy_weight_scale=args.soft_policy_weight_scale,
                    value_loss_scale=args.value_loss_scale,
                    td_value_loss_scales=td_value_loss_scales,
                    seki_loss_scale=args.seki_loss_scale,
                    variance_time_loss_scale=args.variance_time_loss_scale,
                    disable_optimistic_policy=args.disable_optimistic_policy,
                    mask=train_mask,
                )
                profiler.tick("loss")

                # Scale loss for gradient averaging across accumulation steps
                if grad_reducer is not None and is_last_micro:
                    grad_reducer.enable()
                scaled_loss = loss / grad_accum_steps
                if grad_scaler is not None:
                    scaled_loss = grad_scaler.scale(scaled_loss)
                scaled_loss.backward()
            profiler.tick("bwd")

            # Accumulate micro-step metrics and seki moving average
            metrics = dict(zip(_METRIC_KEYS, metrics_stack.tolist()))
            for k in metrics:
                micro_metrics_accum[k] += metrics[k]
            accum_moving_sum += new_moving_sum.item()
            accum_moving_weight += new_moving_weight.item()

            accum_step += 1

            if accum_step == grad_accum_steps:
                accum_step = 0

                # Write back seki moving average (averaged across micro-steps and ranks)
                avg_moving_sum = accum_moving_sum / grad_accum_steps
                avg_moving_weight = accum_moving_weight / grad_accum_steps
                if world_size > 1:
                    mv_t = torch.tensor([avg_moving_sum, avg_moving_weight], device=device)
                    torch.distributed.all_reduce(mv_t, op=torch.distributed.ReduceOp.SUM)
                    mv_t /= world_size
                    avg_moving_sum, avg_moving_weight = mv_t.tolist()
                model.moving_unowned_proportion_sum = avg_moving_sum
                model.moving_unowned_proportion_weight = avg_moving_weight
                accum_moving_sum = 0.0
                accum_moving_weight = 0.0

                # In zero dp-mode, reduce gradients to owner ranks before clipping.
                if grad_reducer is not None:
                    grad_reducer.finalize()
                    profiler.tick("reduce")
                elif dp_zero:
                    reduce_zero_grads([zero_adam, muon_opt, shampoo_opt], rank=rank, world_size=world_size)
                    profiler.tick("reduce")

                # Unscale gradients (FP16 only) and check for inf/nan
                skip_step = False
                if grad_scaler is not None:
                    inv_scale = torch.tensor(1.0 / grad_scaler.get_scale(), dtype=torch.float32, device=device)
                    found_inf = torch.zeros(1, dtype=torch.float32, device=device)
                    if dp_zero:
                        params_to_check = [
                            p for opt in [zero_adam, muon_opt, shampoo_opt]
                            if opt is not None
                            for p in opt.partitions[rank].values()
                        ]
                    else:
                        params_to_check = list(model.parameters())
                    grads = [p.grad for p in params_to_check if p.grad is not None]
                    torch._amp_foreach_non_finite_check_and_unscale_(grads, found_inf, inv_scale)
                    if world_size > 1:
                        torch.distributed.all_reduce(found_inf, op=torch.distributed.ReduceOp.MAX)
                    skip_step = found_inf.item() > 0.0
                    profiler.tick("unscale")

                # Update LR/WD schedule
                if use_hzy_schedule:
                    base_lr, base_wd = get_lr_wd(total_samples_trained)
                    for pg in inner_optimizer.param_groups:
                        pg["lr"] = base_lr
                        if pg.get("weight_decay", 0) > 0:
                            pg["weight_decay"] = base_wd
                    for opt in [muon_opt, shampoo_opt]:
                        if opt is not None:
                            inner = getattr(opt, '_local_opt', opt)
                            if inner is not None:
                                inner.wd = base_wd

                if skip_step:
                    logging.warning(
                        f"Step {global_step}: inf/nan in gradients, skipping optimizer step "
                        f"(scale={grad_scaler.get_scale():.1f})"
                    )
                    for p in model.parameters():
                        p.grad = None
                    grad_norm = 0.0
                    grad_scaler.update(found_inf=True)
                else:
                    # Gradient clipping + optimizer step
                    if dp_zero:
                        # Distributed clip: each rank only has its partition's gradients.
                        owned_params = [
                            p for opt in [zero_adam, muon_opt, shampoo_opt]
                            if opt is not None
                            for p in opt.partitions[rank].values()
                            if p.grad is not None
                        ]
                        local_norm_sq = sum((p.grad.norm() ** 2 for p in owned_params),
                                            torch.tensor(0.0, device=device))
                        global_norm_sq = local_norm_sq.unsqueeze(0)
                        torch.distributed.all_reduce(global_norm_sq)
                        grad_norm = global_norm_sq.sqrt()
                        clip_coef = torch.clamp(1.0 / (grad_norm + 1e-6), max=1.0)
                        if clip_coef < 1.0:
                            for p in owned_params:
                                p.grad.mul_(clip_coef)
                        grad_norm = grad_norm.item()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    profiler.tick("clip")
                    if zero_adam is not None:
                        # Defer ZeRO param sync and do one coalesced sync after all optimizers step.
                        zero_adam.step(sync=False)
                    else:
                        inner_optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                        base_lr = scheduler.get_last_lr()[0]

                    # Muon / Shampoo update (use same LR as AdamW for this step)
                    if muon_opt is not None:
                        if zero_adam is not None:
                            muon_opt.step(base_lr, sync=False)
                        else:
                            muon_opt.step(base_lr)
                    if shampoo_opt is not None:
                        if zero_adam is not None:
                            shampoo_opt.step(base_lr, sync=False)
                        else:
                            shampoo_opt.step(base_lr)
                    profiler.tick("optim")
                    if zero_adam is not None:
                        # Sync Adam + Muon/Shampoo owned parameters in a single collective pass.
                        sync_zero_params([zero_adam, muon_opt, shampoo_opt], rank=rank, world_size=world_size)
                        profiler.tick("sync")
                    if ema is not None:
                        ema.update(model)
                    if grad_scaler is not None:
                        grad_scaler.update(found_inf=False)
                    sync_xla_if_needed()
                profiler.step_done()
                global_step += 1
                total_samples_trained += batch_size * world_size * grad_accum_steps

                # Average micro-step metrics and add to running totals
                for k in _metric_keys:
                    running[k] += micro_metrics_accum[k] / grad_accum_steps
                    micro_metrics_accum[k] = 0.0
                running["grad_norm"] += grad_norm if isinstance(grad_norm, float) else grad_norm.item()
                if muon_opt is not None:
                    running["muon_update_rms"] += muon_opt.last_update_rms
                if shampoo_opt is not None:
                    running["shampoo_precond_rms"] += shampoo_opt.last_precond_rms
                running["count"] += 1

                if global_step % args.print_every == 0:
                    # All-reduce running metrics across ranks
                    if world_size > 1:
                        keys = list(running.keys())
                        vals = torch.tensor([running[k] for k in keys], device=device)
                        torch.distributed.all_reduce(vals, op=torch.distributed.ReduceOp.SUM)
                        vals /= world_size
                        for i, k in enumerate(keys):
                            running[k] = vals[i].item()
                    if rank == 0:
                        time_now = time.perf_counter()
                        print_metrics(time_now - last_print_time)
                        last_print_time = time_now
                        if device.type == "cuda":
                            alloc_gb = torch.cuda.memory_allocated(device) / 1e9
                            resv_gb = torch.cuda.memory_reserved(device) / 1e9
                            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                            free_gb = free_bytes / 1e9
                            total_gb = total_bytes / 1e9
                            logging.info(
                                f"  GPU MEM: allocated={alloc_gb:.2f}GB reserved={resv_gb:.2f}GB "
                                f"free={free_gb:.2f}GB total={total_gb:.2f}GB "
                                f"(external={total_gb - free_gb - resv_gb:.2f}GB)"
                            )
                    reset_running()

                # Periodic save (all ranks call save_checkpoint for ZeRO gather)
                if total_samples_trained - last_save_samples >= args.save_every_samples:
                    t_save_start = time.perf_counter()
                    save_checkpoint()
                    last_print_time += time.perf_counter() - t_save_start
                    last_save_samples = total_samples_trained

                # Validation (all ranks participate, all_reduce to aggregate metrics)
                if total_samples_trained - last_val_samples >= args.val_every_samples:
                    t_val_start = time.perf_counter()
                    val_files = sorted(glob.glob(os.path.join(val_dir, "*.npz")))
                    if val_files:
                        model.eval()
                        val_metrics = {k: 0.0 for k in _metric_keys}
                        val_metrics["count"] = 0
                        with torch.no_grad():
                            val_gen = data_processing.read_npz_training_data(
                                val_files[:3],
                                batch_size=batch_size,
                                world_size=world_size,
                                rank=rank,
                                pos_len=pos_len,
                                device=device,
                                symmetry_type=None,
                                include_meta=False,
                                enable_history_matrices=args.enable_history_matrices,
                                model_config=model_config,
                                use_pin_memory=use_pin_memory,
                                varlen=args.varlen,
                                allow_nonfull_mask=args.allow_nonfull_mask,
                            )
                            for val_batch in data_processing.prefetch_generator(val_gen, args.prefetch_batches):
                                with torch.amp.autocast(amp_device, dtype=amp_dtype, enabled=use_amp):
                                    outputs = model(val_batch["binaryInputNCHW"], val_batch["globalInputNC"])
                                val_mask = val_batch["binaryInputNCHW"][:, 0:1, :, :].contiguous() if args.varlen else None
                                _, batch_metrics = compute_loss(
                                    model, outputs, val_batch, pos_len,
                                    is_training=False,
                                    soft_policy_weight_scale=args.soft_policy_weight_scale,
                                    value_loss_scale=args.value_loss_scale,
                                    td_value_loss_scales=td_value_loss_scales,
                                    seki_loss_scale=args.seki_loss_scale,
                                    variance_time_loss_scale=args.variance_time_loss_scale,
                                    disable_optimistic_policy=args.disable_optimistic_policy,
                                    mask=val_mask,
                                )
                                for k in batch_metrics:
                                    val_metrics[k] += batch_metrics[k]
                                val_metrics["count"] += 1

                        # Aggregate metrics across all ranks
                        if world_size > 1:
                            agg_keys = _metric_keys + ["count"]
                            agg_vals = torch.tensor([val_metrics[k] for k in agg_keys], device=device)
                            torch.distributed.all_reduce(agg_vals, op=torch.distributed.ReduceOp.SUM)
                            for i, k in enumerate(agg_keys):
                                val_metrics[k] = agg_vals[i].item()

                        if rank == 0:
                            weight_sum = max(val_metrics["wsum"], 1e-10)
                            batch_count = max(val_metrics["count"], 1)
                            logging.info(
                                f"  VAL [{total_samples_trained} samples]: loss={val_metrics['loss'] / batch_count:.4f}, "
                                f"p0loss={val_metrics['p0loss'] / weight_sum:.4f}, "
                                f"vloss={val_metrics['vloss'] / weight_sum:.4f}, "
                                f"oloss={val_metrics['oloss'] / weight_sum:.4f}, "
                                f"skloss={val_metrics['skloss'] / weight_sum:.4f}, "
                                f"pacc1={val_metrics['pacc1'] / weight_sum:.4f}"
                            )
                            if tb_writer is not None:
                                tb_writer.add_scalar("val/loss", val_metrics["loss"] / batch_count, total_samples_trained)
                                for k in _per_sample_keys:
                                    tb_writer.add_scalar(f"val/{k}", val_metrics[k] / weight_sum, total_samples_trained)
                        model.train()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    last_print_time += time.perf_counter() - t_val_start
                    last_val_samples = total_samples_trained

                profiler.tick()  # reset timer — exclude print/save/val overhead from next step

                if total_samples_trained >= args.max_training_samples:
                    break
            else:
                profiler.tick()  # reset timer for intermediate micro-step

    # Final save (all ranks call for ZeRO gather; only rank 0 writes; sync to ensure completion)
    save_checkpoint(async_save=False)
    logging.info(f"Training complete: {total_samples_trained} samples, {global_step} steps")
    if tb_writer is not None:
        tb_writer.close()


def _mp_main(spawn_rank, world_size, args, device_ids):
    main(spawn_rank, world_size, args, gpu_id=device_ids[spawn_rank])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal Transformer training for KataGo (nano)")
    parser.add_argument("--traindir", required=True, help="Training output directory")
    parser.add_argument("--datadir", required=True, help="Data directory with train/ and val/ subdirs")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu", "xla"],
                        help="Training device. Use xla for PyTorch/XLA TPU testing")
    parser.add_argument("--pos-len", type=int, default=19, help="Board size")
    parser.add_argument("--batch-size", type=int, default=256, help="Per-GPU micro batch size")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--model-kind", type=str, default="b12c192", help="Model config preset name")
    parser.add_argument("--num-layers", type=int, default=None, help="Number of transformer layers (overrides --model-kind)")
    parser.add_argument("--hidden-size", type=int, default=192, help="Hidden dimension")
    parser.add_argument("--num-heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate (cosine schedule)")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay (cosine schedule)")
    parser.add_argument("--lr-schedule", type=str, default="cosine", choices=["cosine", "hzy"],
                        help="LR schedule: cosine (warmup+cosine decay) or hzy (step-function from lr_schedule.xlsx)")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "muon", "shampoo"],
                        help="Optimizer: adam (pure AdamW), muon (Muon for blocks + AdamW), shampoo (Shampoo for blocks + AdamW)")
    parser.add_argument("--shampoo-lr-multiplier", type=float, default=2.0, help="Shampoo LR multiplier over base lr")
    parser.add_argument("--init-std", type=float, default=0.02, help="Init std for all weight initialization")
    parser.add_argument("--max-training-samples", type=int, default=100000000, help="Total training samples")
    parser.add_argument("--save-every-samples", type=int, default=1000000, help="Save checkpoint every N samples")
    parser.add_argument("--symmetry-type", type=str, default="xyt",
                        help="Data symmetry type (none/x/xy/x+y/t/xyt/all). "
                             "'all' applies all 8 symmetries per sample, requires batch-size divisible by 8")
    parser.add_argument("--print-every", type=int, default=100, help="Print every N optimizer steps")
    parser.add_argument("--val-every-samples", type=int, default=1000000, help="Run validation every N samples")
    parser.add_argument("--warmup-samples", type=int, default=2000000, help="LR warmup samples")
    parser.add_argument("--enable-history-matrices", action="store_true", help="Enable history matrices (for Go)")
    parser.add_argument("--initial-checkpoint", type=str, default=None, help="Initial checkpoint to load from")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--soft-policy-weight-scale", type=float, default=8.0, help="Soft policy loss coeff")
    parser.add_argument("--value-loss-scale", type=float, default=0.6, help="Value loss coeff")
    parser.add_argument("--td-value-loss-scales", type=str, default="0.6,0.6,0.6", help="TD value loss coeffs")
    parser.add_argument("--seki-loss-scale", type=float, default=1.0, help="Seki loss coeff")
    parser.add_argument("--variance-time-loss-scale", type=float, default=1.0, help="Variance time loss coeff")
    parser.add_argument("--disable-optimistic-policy", action="store_true", help="Disable optimistic policy")
    parser.add_argument("--dp-mode", type=str, default="ddp", choices=["ddp", "zero"],
                        help="Data parallel mode: ddp (standard DDP) or zero (manual gradient reduce, saves memory)")
    parser.add_argument("--overlap-reduce", action="store_true",
                        help="Overlap gradient reduce with backward (ZeRO mode only)")
    parser.add_argument("--multi-gpus", type=str, default=None, help="Comma-separated GPU device ids for DDP (e.g. 0,1,2,3)")
    parser.add_argument("--master-port", type=int, default=23456, help="Localhost port for DDP communication")
    parser.add_argument("--prefetch-batches", type=int, default=64, help="Prefetch queue depth (0=off)")
    parser.add_argument("--score-mode", type=str, default="simple", choices=["mixop", "mix", "simple"],
                        help="Score belief head mode: mixop=linear+offset/parity+MoS, mix=linear+MoS, simple=single linear")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--use-te", nargs='?', const='full', default=None,
                        choices=['full', 'hybrid'],
                        help="Use TransformerEngine: 'full' (default, fused TE block), "
                             "'hybrid' (TE linear + PyTorch SDPA)")
    parser.add_argument("--use-fp8", action="store_true", help="Enable FP8 training (requires --use-te and Hopper/Ada GPU)")
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16", "none"],
                        help="AMP dtype: bf16 (default), fp16 (with loss scaling), none (disable AMP)")
    parser.add_argument("--profile", action="store_true",
                        help="Enable per-stage CUDA-synced profiling (adds sync overhead)")
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="Disable TensorBoard logging")
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="EMA decay rate for model params (0=disabled, typical: 0.999 or 0.9999)")
    parser.add_argument("--varlen", action="store_true",
                        help="Enable variable-length board input with masking")
    parser.add_argument("--allow-nonfull-mask", action="store_true",
                        help="Allow legacy fixed-board data whose channel-0 mask is not all ones")
    parser.add_argument("--gated-attn", action="store_true",
                        help="Enable elementwise gated attention (sigmoid gate on attention output)")
    parser.add_argument("--zero-centered-norm", action="store_true",
                        help="Use zero-centered RMSNorm (weight=0 init, gamma=1+weight, WD pushes gamma toward 1)")
    parser.add_argument("--learnable-rope", action="store_true",
                        help="Enable learnable per-head RoPE frequencies (replaces fixed precomputed RoPE)")
    args = parser.parse_args()

    if args.device == "auto" and os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
        args.device = "xla"

    # Validation
    if args.grad_accum_steps < 1:
        parser.error("--grad-accum-steps must be >= 1")
    if args.print_every < 1:
        parser.error("--print-every must be >= 1")
    if args.symmetry_type == "all" and args.batch_size % 8 != 0:
        parser.error("--batch-size must be divisible by 8 when --symmetry-type is 'all'")
    if args.amp_dtype == "fp16" and not (args.device in ("auto", "cuda") and torch.cuda.is_available()):
        parser.error("--amp-dtype fp16 requires CUDA")
    if args.gated_attn and args.use_te:
        parser.error("--gated-attn and --use-te cannot be used together")
    if args.varlen and args.use_te == 'full':
        parser.error("--varlen requires --use-te hybrid (--use-te full does not support varlen)")
    if args.device == "xla":
        if args.use_te:
            parser.error("--device xla does not support --use-te/TransformerEngine")
        if args.use_fp8:
            parser.error("--device xla does not support --use-fp8")
        if args.optimizer != "adam":
            parser.error("--device xla currently supports --optimizer adam only")
        if args.dp_mode != "ddp":
            parser.error("--device xla currently supports single-device ddp mode only")
        if args.profile:
            parser.error("--profile is CUDA-only in this script")
        if args.multi_gpus is not None:
            parser.error("--multi-gpus is CUDA-only; use --device xla on v6e-1 for the initial TPU test")

    # Detect torchrun launch (torchrun sets RANK, LOCAL_RANK, WORLD_SIZE env vars)
    torchrun_rank = os.environ.get("RANK")
    torchrun_local_rank = os.environ.get("LOCAL_RANK")
    torchrun_world_size = os.environ.get("WORLD_SIZE")

    if torchrun_rank is not None and torchrun_local_rank is not None and torchrun_world_size is not None:
        # Launched via torchrun (supports multi-node multi-GPU)
        rank = int(torchrun_rank)
        local_rank = int(torchrun_local_rank)
        world_size = int(torchrun_world_size)
        main(rank, world_size, args, gpu_id=local_rank)
    else:
        # Legacy single-node launch via --multi-gpus or single GPU
        multi_gpu_device_ids = []
        if args.multi_gpus is not None:
            for piece in args.multi_gpus.split(","):
                piece = piece.strip()
                multi_gpu_device_ids.append(int(piece))
        else:
            multi_gpu_device_ids = [0]

        num_gpus_used = len(multi_gpu_device_ids)

        if num_gpus_used > 1:
            torch.multiprocessing.set_start_method("spawn")
            try:
                torch.multiprocessing.spawn(
                    _mp_main,
                    nprocs=num_gpus_used,
                    args=(num_gpus_used, args, multi_gpu_device_ids),
                )
            except KeyboardInterrupt:
                print("\nInterrupted. Killing all worker processes...", file=sys.stderr, flush=True)
                os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
        else:
            main(0, 1, args, gpu_id=multi_gpu_device_ids[0])
