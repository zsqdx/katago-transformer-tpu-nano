"""ZeRO Stage 1: optimizer state partitioning across GPUs."""

import logging
from collections import defaultdict

import torch
import torch.distributed as dist
from optimizers import MuonOptimizer, ShampooOptimizer, compute_split_info


# ---------------------------------------------------------------------------
# Cost estimation for load-balanced partitioning
# ---------------------------------------------------------------------------
def _make_muon_cost(use_te=False, num_heads=0):
    """Create a Muon cost function aware of TE/headwise splits."""
    def cost_fn(name, p):
        if p.ndim == 4:
            m, n = p.shape[0], p.shape[1] * p.shape[2] * p.shape[3]
        else:
            m, n = p.shape
        n_chunks, chunk_m, chunk_n, _ = compute_split_info(name, m, n, use_te, num_heads)
        lo, hi = min(chunk_m, chunk_n), max(chunk_m, chunk_n)
        # 5 NS iters, each ~3 matmuls of shape (lo, lo) @ (lo, hi)
        return n_chunks * 5 * lo * lo * (2 * hi + lo)
    return cost_fn


def _make_shampoo_cost(use_te=False, num_heads=0):
    """Create a Shampoo cost function aware of TE/headwise splits."""
    def cost_fn(name, p):
        if p.ndim >= 2:
            m, n = p.shape[0], p.shape[1:].numel()
        else:
            m, n = p.shape[0], 1
        n_chunks, chunk_m, chunk_n, _ = compute_split_info(name, m, n, use_te, num_heads)
        # 4 NS iters: each does ~8 matmuls for L (m^3) and R (n^3) plus sandwich (m^2*n + m*n^2)
        # Plus L/R EMA: grad @ grad.T (m^2*n) + grad.T @ grad (m*n^2)
        cm, cn = chunk_m, chunk_n
        return n_chunks * (
            4 * (8 * cm**3 + 8 * cn**3 + cm*cm*cn + cm*cn*cn) + (cm*cm*cn + cm*cn*cn)
        )
    return cost_fn


# ---------------------------------------------------------------------------
# LPT (Longest Processing Time) greedy partition
# ---------------------------------------------------------------------------
def _lpt_partition(named_params, cost_fn, world_size):
    """Partition named_params across ranks using LPT greedy algorithm.

    Returns: list of dicts, one per rank. Each dict is {name: param}.
    All ranks compute the same result deterministically.
    """
    items = [(cost_fn(name, p), name, p) for name, p in named_params.items()]
    # Sort by cost descending, break ties by name for determinism
    items.sort(key=lambda x: (-x[0], x[1]))

    partitions = [{} for _ in range(world_size)]
    loads = [0] * world_size

    for cost, name, p in items:
        # Assign to rank with smallest current load; break ties by rank index
        target = min(range(world_size), key=lambda r: (loads[r], r))
        partitions[target][name] = p
        loads[target] += cost

    return partitions


def _numel_partition(named_params, world_size):
    """Partition named_params by numel using LPT."""
    return _lpt_partition(named_params, lambda name, p: p.numel(), world_size)


# ---------------------------------------------------------------------------
# Coalesced broadcast: each owner rank broadcasts its params in one flat tensor
# ---------------------------------------------------------------------------
def _coalesced_broadcast(partitions, rank, world_size):
    """Broadcast updated parameters from each owner rank to all others.

    partitions: list of dicts [{name: param}, ...], one per rank.
    Each src_rank is fully processed (broadcast + unpack) before moving to the
    next, so at most one flat buffer is live at a time per (device, dtype).
    """
    for src_rank in range(world_size):
        params = list(partitions[src_rank].values())
        if not params:
            continue
        # Group by (device, dtype) to avoid implicit dtype promotion in torch.cat.
        buckets = {}
        for p in params:
            key = (p.device, p.dtype)
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(p)

        for (dev, dtype), bucket in buckets.items():
            if rank == src_rank:
                with torch.no_grad():
                    flat = torch.cat([p.detach().reshape(-1) for p in bucket], dim=0).contiguous()
            else:
                total_numel = sum(p.numel() for p in bucket)
                flat = torch.empty(total_numel, dtype=dtype, device=dev)
            dist.broadcast(flat, src=src_rank)
            if rank != src_rank:
                # Unpack back into param tensors.
                with torch.no_grad():
                    offset = 0
                    for p in bucket:
                        numel = p.numel()
                        p.copy_(flat[offset:offset + numel].reshape_as(p))
                        offset += numel


def _coalesced_reduce(partitions, rank, world_size):
    """Reduce gradients to each owner rank.

    For each owner rank, flattens the gradients of its parameters into one
    tensor, reduces (averages) across all ranks, and writes the result back.
    Non-owner ranks have their gradients freed (set to None).

    Each owner_rank is fully processed before moving to the next, so at most
    one flat copy is live at a time per (device, dtype), and non-owned
    gradients are freed progressively.

    partitions: list of dicts [{name: param}, ...], one per rank.
    """
    for owner_rank in range(world_size):
        params = list(partitions[owner_rank].values())
        if not params:
            continue
        # Group by (device, dtype).
        buckets = {}
        for p in params:
            if p.grad is None:
                continue
            key = (p.device, p.grad.dtype)
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(p)

        for (dev, dtype), bucket in buckets.items():
            flat = torch.cat([p.grad.detach().reshape(-1) for p in bucket], dim=0).contiguous()
            dist.reduce(flat, dst=owner_rank, op=dist.ReduceOp.AVG)
            if rank == owner_rank:
                # Unpack reduced gradients back into param .grad tensors.
                with torch.no_grad():
                    offset = 0
                    for p in bucket:
                        numel = p.numel()
                        p.grad = flat[offset:offset + numel].reshape_as(p)
                        offset += numel
            else:
                # Free non-owned gradients immediately.
                for p in bucket:
                    p.grad = None


def reduce_zero_grads(optimizers, rank, world_size):
    """Reduce gradients for all ZeRO optimizers in one pass.

    Merges partitions from multiple optimizers and calls _coalesced_reduce
    so that each rank only retains reduced gradients for its own parameters.
    """
    merged = [{} for _ in range(world_size)]
    for opt in optimizers:
        if opt is None:
            continue
        parts = opt.partitions
        if len(parts) != world_size:
            raise ValueError("Inconsistent world_size across ZeRO optimizers")
        for r in range(world_size):
            for name, p in parts[r].items():
                if name in merged[r]:
                    raise ValueError(f"Parameter '{name}' appears in multiple ZeRO optimizers on rank {r}")
                merged[r][name] = p
    _coalesced_reduce(merged, rank, world_size)


def sync_zero_params(optimizers, rank, world_size):
    """Sync parameters once for multiple ZeRO optimizer wrappers.

    Args:
        optimizers: iterable of ZeRO optimizer wrappers (None entries allowed).
        rank: local rank.
        world_size: total number of ranks.
    """
    merged = [{} for _ in range(world_size)]
    for opt in optimizers:
        if opt is None:
            continue
        parts = opt.partitions
        if len(parts) != world_size:
            raise ValueError("Inconsistent world_size across ZeRO optimizers")
        for r in range(world_size):
            for name, p in parts[r].items():
                if name in merged[r]:
                    raise ValueError(f"Parameter '{name}' appears in multiple ZeRO optimizers on rank {r}")
                merged[r][name] = p

    _coalesced_broadcast(merged, rank, world_size)


# ---------------------------------------------------------------------------
# ZeROAdamW
# ---------------------------------------------------------------------------
class ZeROAdamW:
    """ZeRO Stage 1 wrapper for AdamW (handles both decay and no-decay params)."""

    def __init__(self, adam_params, no_decay_params, lr, betas, wd, device, rank, world_size):
        """
        adam_params: dict {name: param} for weight-decay params
        no_decay_params: dict {name: param} for no-decay params (1D bias/norm)
        """
        self.rank = rank
        self.world_size = world_size

        # Merge all params for partitioning
        all_params = {}
        self._no_decay_names = set(no_decay_params.keys())
        for name, p in no_decay_params.items():
            all_params[name] = p
        for name, p in adam_params.items():
            all_params[name] = p

        self._all_params = all_params
        self.partitions = _numel_partition(all_params, world_size)

        # Build param groups for this rank's partition
        my_params = self.partitions[rank]
        my_decay = [p for name, p in my_params.items() if name not in self._no_decay_names]
        my_no_decay = [p for name, p in my_params.items() if name in self._no_decay_names]

        param_groups = []
        if my_no_decay:
            param_groups.append({"params": my_no_decay, "weight_decay": 0.0})
        if my_decay:
            param_groups.append({"params": my_decay, "weight_decay": wd})

        # Fallback: if this rank has no params (shouldn't happen with 30+ params),
        # create a dummy group so scheduler works
        if not param_groups:
            param_groups.append({"params": [], "weight_decay": 0.0})

        self._optimizer = torch.optim.AdamW(
            param_groups, lr=lr, betas=betas, fused=(device.type == "cuda"),
        )

        # Log partition info
        my_numel = sum(p.numel() for p in my_params.values())
        total_numel = sum(p.numel() for p in all_params.values())
        logging.info(
            f"ZeROAdamW rank {rank}: {len(my_params)}/{len(all_params)} params, "
            f"{my_numel:,}/{total_numel:,} elements "
            f"({100*my_numel/max(total_numel,1):.1f}%)"
        )

    @property
    def optimizer(self):
        """Expose internal optimizer for LR scheduler."""
        return self._optimizer

    def step(self, sync=True):
        """Run optimizer step on local partition.

        If sync=True (default), broadcast updated params immediately.
        """
        self._optimizer.step()
        if sync:
            _coalesced_broadcast(self.partitions, self.rank, self.world_size)

    def gather_state_for_save(self):
        """Collective operation: gather optimizer state to rank 0 as name-based dict.

        All ranks must call this. Returns the gathered state on rank 0, None on others.
        """
        # Build local name-based state
        my_params = self.partitions[self.rank]
        local_state = {}
        opt_state = self._optimizer.state

        for name, p in my_params.items():
            if p in opt_state:
                s = opt_state[p]
                local_state[name] = {k: v.cpu() if torch.is_tensor(v) else v for k, v in s.items()}

        # Gather to rank 0
        gathered = [None] * self.world_size if self.rank == 0 else None
        dist.gather_object(local_state, gathered, dst=0)

        if self.rank == 0:
            merged = {}
            for rank_state in gathered:
                merged.update(rank_state)
            return merged
        return None

    def load_state_distributed(self, saved_state, device):
        """Load name-based optimizer state into this rank's partition.

        saved_state: the full name-based state dict (same on all ranks).
        Handles the case where AdamW state is not yet initialized (lazy init).
        """
        opt_state = self._optimizer.state
        my_params = self.partitions[self.rank]

        for name, p in my_params.items():
            if name not in saved_state:
                continue
            saved_s = saved_state[name]
            if not isinstance(saved_s, dict):
                continue
            if p not in opt_state:
                # AdamW lazily initializes state on first step();
                # create the entry so we can populate it from the checkpoint.
                opt_state[p] = {}
            for k, v in saved_s.items():
                if torch.is_tensor(v):
                    if k in opt_state[p]:
                        opt_state[p][k].copy_(v.to(device))
                    else:
                        opt_state[p][k] = v.clone().to(device)
                else:
                    opt_state[p][k] = v


# ---------------------------------------------------------------------------
# ZeROMuon
# ---------------------------------------------------------------------------
class ZeROMuon:
    """ZeRO Stage 1 wrapper for MuonOptimizer."""

    def __init__(self, named_params, lr_multiplier, momentum, wd, device, rank, world_size, use_te=False, num_heads=0):
        self.rank = rank
        self.world_size = world_size
        self._all_params = named_params
        muon_cost = _make_muon_cost(use_te, num_heads)
        self.partitions = _lpt_partition(named_params, muon_cost, world_size)

        my_params = self.partitions[rank]
        self._local_numel = sum(p.numel() for p in my_params.values())
        self._local_opt = MuonOptimizer(
            my_params, lr_multiplier=lr_multiplier, momentum=momentum,
            wd=wd, device=device, use_te=use_te, num_heads=num_heads,
        ) if my_params else None

        self.last_update_rms = 0.0

        total_numel = sum(p.numel() for p in named_params.values())
        my_cost = sum(muon_cost(name, p) for name, p in my_params.items())
        total_cost = sum(muon_cost(name, p) for name, p in named_params.items())
        logging.info(
            f"ZeROMuon rank {rank}: {len(my_params)}/{len(named_params)} params, "
            f"{self._local_numel:,}/{total_numel:,} elements, "
            f"cost {my_cost/max(total_cost,1)*100:.1f}%"
        )

    def step(self, base_lr, sync=True):
        if self._local_opt is not None:
            self._local_opt.step(base_lr)
            self.last_update_rms = self._local_opt.last_update_rms
        else:
            self.last_update_rms = 0.0

        if sync:
            _coalesced_broadcast(self.partitions, self.rank, self.world_size)

        # Weighted all-reduce: global_rms = sqrt(sum(rms_i^2 * numel_i) / sum(numel_i))
        dev = next(iter(self._all_params.values())).device
        buf = torch.tensor(
            [self.last_update_rms ** 2 * self._local_numel, float(self._local_numel)],
            dtype=torch.float64, device=dev,
        )
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        self.last_update_rms = (buf[0] / max(buf[1].item(), 1)).sqrt().item()

    def state_dict(self):
        """Return name-based state dict (local partition only, for gather)."""
        if self._local_opt is not None:
            return self._local_opt.state_dict()
        return {}

    def gather_state_for_save(self):
        """Collective: gather Muon state to rank 0."""
        local_state = self.state_dict()
        gathered = [None] * self.world_size if self.rank == 0 else None
        dist.gather_object(local_state, gathered, dst=0)
        if self.rank == 0:
            merged = {}
            for rank_state in gathered:
                merged.update(rank_state)
            return merged
        return None

    def load_state_distributed(self, saved_state, device):
        """Load name-based state into local partition."""
        if self._local_opt is not None:
            self._local_opt.load_state_dict(saved_state, device)


# ---------------------------------------------------------------------------
# ZeROShampoo
# ---------------------------------------------------------------------------
class ZeROShampoo:
    """ZeRO Stage 1 wrapper for ShampooOptimizer."""

    def __init__(self, named_params, lr_multiplier, momentum, wd, beta2, device, rank, world_size,
                 use_te=False, num_heads=0):
        self.rank = rank
        self.world_size = world_size
        self._all_params = named_params
        shampoo_cost = _make_shampoo_cost(use_te, num_heads)
        self.partitions = _lpt_partition(named_params, shampoo_cost, world_size)

        my_params = self.partitions[rank]
        self._local_numel = sum(p.numel() for p in my_params.values())
        self._local_opt = ShampooOptimizer(
            my_params, lr_multiplier=lr_multiplier, momentum=momentum,
            wd=wd, beta2=beta2, device=device, use_te=use_te, num_heads=num_heads,
        ) if my_params else None

        self.last_precond_rms = 0.0

        total_numel = sum(p.numel() for p in named_params.values())
        my_cost = sum(shampoo_cost(name, p) for name, p in my_params.items())
        total_cost = sum(shampoo_cost(name, p) for name, p in named_params.items())
        logging.info(
            f"ZeROShampoo rank {rank}: {len(my_params)}/{len(named_params)} params, "
            f"{self._local_numel:,}/{total_numel:,} elements, "
            f"cost {my_cost/max(total_cost,1)*100:.1f}%"
        )

    def step(self, base_lr, sync=True):
        if self._local_opt is not None:
            self._local_opt.step(base_lr)
            self.last_precond_rms = self._local_opt.last_precond_rms
        else:
            self.last_precond_rms = 0.0

        if sync:
            _coalesced_broadcast(self.partitions, self.rank, self.world_size)

        # Weighted all-reduce: global_rms = sqrt(sum(rms_i^2 * numel_i) / sum(numel_i))
        dev = next(iter(self._all_params.values())).device
        buf = torch.tensor(
            [self.last_precond_rms ** 2 * self._local_numel, float(self._local_numel)],
            dtype=torch.float64, device=dev,
        )
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        self.last_precond_rms = (buf[0] / max(buf[1].item(), 1)).sqrt().item()

    def state_dict(self):
        if self._local_opt is not None:
            return self._local_opt.state_dict()
        return {}

    def gather_state_for_save(self):
        """Collective: gather Shampoo state to rank 0."""
        local_state = self.state_dict()
        gathered = [None] * self.world_size if self.rank == 0 else None
        dist.gather_object(local_state, gathered, dst=0)
        if self.rank == 0:
            merged = {}
            for rank_state in gathered:
                merged.update(rank_state)
            return merged
        return None

    def load_state_distributed(self, saved_state, device):
        if self._local_opt is not None:
            self._local_opt.load_state_dict(saved_state, device)


# ---------------------------------------------------------------------------
# ZeROGradReducer: overlap gradient reduce with backward computation
# ---------------------------------------------------------------------------
class ZeROGradReducer:
    """Overlap gradient reduction with backward pass using post-accumulate-grad hooks.

    Two-phase strategy:
    - warmup: record true gradient ready order, fall back to sync _coalesced_reduce.
    - steady state: use recorded order to build buckets, launch async reduce as
      buckets become ready during backward.
    """

    def __init__(self, optimizers, model, rank, world_size, bucket_size_mb=25, debug=False):
        self._rank = rank
        self._world_size = world_size
        self._bucket_size_bytes = bucket_size_mb * 1024 * 1024
        self._debug = debug
        self._debug_step_count = 0

        # Merge partitions (same validation as reduce_zero_grads)
        self._merged = [{} for _ in range(world_size)]
        for opt in optimizers:
            if opt is None:
                continue
            parts = opt.partitions
            if len(parts) != world_size:
                raise ValueError("Inconsistent world_size across ZeRO optimizers")
            for r in range(world_size):
                for name, p in parts[r].items():
                    if name in self._merged[r]:
                        raise ValueError(f"Parameter '{name}' appears in multiple ZeRO optimizers on rank {r}")
                    self._merged[r][name] = p

        # Build param_id -> owner_rank mapping, register hooks only on managed params
        self._param_to_owner = {}
        self._all_managed_ids = set()
        self._id_to_param = {}
        self._hooks = []
        for r in range(world_size):
            for name, p in self._merged[r].items():
                pid = id(p)
                self._param_to_owner[pid] = r
                self._all_managed_ids.add(pid)
                self._id_to_param[pid] = p
                handle = p.register_post_accumulate_grad_hook(self._grad_hook)
                self._hooks.append(handle)

        # State
        self._enabled = False
        self._warmup_done = False
        self._recording_order = []
        self._observed_dtypes = {}

    def _grad_hook(self, p):
        if not self._enabled:
            return
        pid = id(p)
        if not self._warmup_done:
            self._recording_order.append(pid)
            self._observed_dtypes[pid] = (p.device, p.grad.dtype)
            return
        # Steady state: check if this param completes a bucket
        bucket_idx = self._param_to_bucket_idx.get(pid)
        if bucket_idx is None:
            return
        self._pending[bucket_idx] -= 1
        if self._pending[bucket_idx] == 0:
            self._launch_bucket(bucket_idx)

    def _launch_bucket(self, bucket_idx):
        owner, params = self._buckets[bucket_idx]
        flat = torch.cat([p.grad.detach().reshape(-1) for p in params]).contiguous()
        # Free original grads immediately on all ranks
        for p in params:
            p.grad = None
        work = dist.reduce(flat, dst=owner, op=dist.ReduceOp.AVG, async_op=True)
        self._flat_buffers[bucket_idx] = flat
        self._works[bucket_idx] = work
        if self._debug:
            self._debug_launch_seq.append((bucket_idx, "async"))

    def enable(self):
        """Call before the last micro-step's backward pass."""
        self._enabled = True
        if self._warmup_done:
            self._pending = list(self._initial_pending)
            self._flat_buffers = [None] * len(self._buckets)
            self._works = [None] * len(self._buckets)
            self._finalize_params = {}
            if self._debug:
                self._debug_launch_seq = []
        else:
            self._recording_order = []
            self._observed_dtypes = {}

    def finalize(self):
        """Call after backward completes. Waits for async ops, unpacks grads to owners."""
        self._enabled = False

        if not self._warmup_done:
            # Warmup: sync reduce (reuse existing _coalesced_reduce)
            _coalesced_reduce(self._merged, self._rank, self._world_size)
            # Validate: all managed params observed?
            observed = set(self._recording_order)
            missing = self._all_managed_ids - observed
            if missing:
                logging.warning(
                    f"ZeROGradReducer: {len(missing)} params not observed in warmup, "
                    "staying in sync mode"
                )
                return  # _warmup_done stays False
            self._rebuild_buckets()
            self._warmup_done = True
            return

        # Steady state:
        # 1. Fallback: sync reduce for buckets not yet launched
        for bucket_idx in range(len(self._buckets)):
            if self._works[bucket_idx] is not None:
                continue  # Already launched async
            owner, params = self._buckets[bucket_idx]
            graded = [p for p in params if p.grad is not None]
            if not graded:
                continue
            flat = torch.cat([p.grad.detach().reshape(-1) for p in graded]).contiguous()
            for p in graded:
                p.grad = None
            dist.reduce(flat, dst=owner, op=dist.ReduceOp.AVG)  # sync
            self._flat_buffers[bucket_idx] = flat
            self._works[bucket_idx] = "sync"
            self._finalize_params[bucket_idx] = graded
            if self._debug:
                self._debug_launch_seq.append((bucket_idx, "sync"))

        # 2. Wait for all async ops in bucket order
        for bucket_idx in range(len(self._buckets)):
            work = self._works[bucket_idx]
            if work is not None and work != "sync":
                work.wait()

        # 3. Unpack: owner rank writes reduced grads back to p.grad
        for bucket_idx in range(len(self._buckets)):
            flat = self._flat_buffers[bucket_idx]
            if flat is None:
                continue
            owner, params = self._buckets[bucket_idx]
            actual_params = self._finalize_params.get(bucket_idx, params)
            if self._rank == owner:
                with torch.no_grad():
                    offset = 0
                    for p in actual_params:
                        numel = p.numel()
                        p.grad = flat[offset:offset + numel].reshape_as(p)
                        offset += numel

        # 4. Debug: check NCCL call sequence consistency
        if self._debug:
            self._debug_step_count += 1
            if self._debug_step_count <= 3:
                logging.info(
                    f"ZeROGradReducer rank {self._rank} step {self._debug_step_count} "
                    f"nccl seq: {self._debug_launch_seq}"
                )
            if hasattr(self, '_prev_launch_seq') and self._debug_launch_seq != self._prev_launch_seq:
                logging.warning(
                    f"ZeROGradReducer rank {self._rank}: NCCL call sequence changed "
                    f"at step {self._debug_step_count}! prev={self._prev_launch_seq} "
                    f"curr={self._debug_launch_seq}"
                )
            self._prev_launch_seq = list(self._debug_launch_seq)

        # 5. Cleanup
        self._flat_buffers = [None] * len(self._buckets)
        self._works = [None] * len(self._buckets)
        self._finalize_params = {}

    def _rebuild_buckets(self):
        """Build buckets from warmup-recorded gradient ready order."""
        # Deduplicate, preserving first-seen order
        seen = set()
        ordered_ids = []
        for pid in self._recording_order:
            if pid not in seen:
                seen.add(pid)
                ordered_ids.append(pid)

        # Group by (owner, device, grad_dtype), ordered by warmup first appearance
        groups = defaultdict(list)       # (owner, device, dtype) -> [param, ...]
        group_first_seen = {}            # (owner, device, dtype) -> first index
        for i, pid in enumerate(ordered_ids):
            p = self._id_to_param[pid]
            owner = self._param_to_owner[pid]
            dev, dtype = self._observed_dtypes[pid]
            key = (owner, dev, dtype)
            groups[key].append(p)
            if key not in group_first_seen:
                group_first_seen[key] = i

        # Sort groups by warmup first appearance (avoids comparing torch.device/dtype)
        sorted_keys = sorted(groups.keys(), key=lambda k: group_first_seen[k])

        # Split into buckets
        bucket_size_bytes = self._bucket_size_bytes
        self._buckets = []
        self._param_to_bucket_idx = {}
        for key in sorted_keys:
            params = groups[key]
            owner = key[0]
            grad_dtype = key[2]
            grad_elem_size = torch.empty((), dtype=grad_dtype).element_size()
            current = []
            current_bytes = 0
            for p in params:
                current.append(p)
                current_bytes += p.numel() * grad_elem_size
                if current_bytes >= bucket_size_bytes:
                    idx = len(self._buckets)
                    self._buckets.append((owner, current))
                    for bp in current:
                        self._param_to_bucket_idx[id(bp)] = idx
                    current = []
                    current_bytes = 0
            if current:
                idx = len(self._buckets)
                self._buckets.append((owner, current))
                for bp in current:
                    self._param_to_bucket_idx[id(bp)] = idx

        self._initial_pending = [len(params) for (_, params) in self._buckets]
        self._finalize_params = {}

        logging.info(
            f"ZeROGradReducer rank {self._rank}: warmup done, "
            f"{len(self._buckets)} buckets from {len(ordered_ids)} params"
        )

    def remove_hooks(self):
        """Remove all registered gradient hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
