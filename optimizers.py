"""Muon / Shampoo / Adam optimizers for KataGo nano training."""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Newton-Schulz coefficients for matrix orthogonalization (polar_express)
# ---------------------------------------------------------------------------
_POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

# Newton-Schulz coefficients for inverse 4th root (Shampoo preconditioner)
_NS_COEFFS_R4_SCALED = (
    (3.7745392156862745, -9.830711636812923, 7.211935063687831),
    (1.7744313725490195, -0.5323686439402083, 0.05420935725061334),
    (1.4744509803921568, -0.5384714581368423, 0.10138210476839715),
    (1.3786764705882353, -0.5094735805293277, 0.13074301029260285),
)


@torch.compile
def polar_express(G):
    """Newton-Schulz iteration for matrix orthogonalization, 5 steps."""
    assert G.ndim in (2, 3)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)

    for a, b, c in _POLAR_EXPRESS_COEFFS:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class MuonOptimizer:
    """Muon optimizer: momentum + Newton-Schulz orthogonalization."""

    def __init__(self, named_params, lr_multiplier, momentum, wd, device="cuda", use_te=False, num_heads=0):
        self.named_params = named_params
        self.lr_multiplier = lr_multiplier
        self.momentum = momentum
        self.wd = wd
        self._device = device
        self.use_te = use_te
        self.num_heads = num_heads
        self.last_update_rms = 0.0
        self.states = {name: self._init_state(p) for name, p in named_params.items()}
        self._split_info = {}
        for name, p in named_params.items():
            if p.ndim == 4:
                m, n = p.shape[0], p.shape[1] * p.shape[2] * p.shape[3]
            else:
                m, n = p.shape
            self._split_info[name] = compute_split_info(name, m, n, use_te, num_heads)

    @staticmethod
    def _init_state(p):
        return {"momentum": torch.zeros_like(p)}

    def step(self, base_lr):
        muon_lr = base_lr * self.lr_multiplier
        rms_sum = torch.tensor(0.0, device=self._device)
        rms_cnt = 0
        with torch.no_grad():
            for name, p in self.named_params.items():
                if p.grad is None:
                    continue
                assert p.grad.ndim in (2, 4), f"Muon only supports 2D/4D params, got ndim={p.grad.ndim}"
                state = self.states[name]
                grad = p.grad
                original_shape = grad.shape

                state["momentum"].mul_(self.momentum).add_(grad)
                update = state["momentum"]

                split_info = self._split_info[name]
                update = flatten_and_split(update, split_info)

                update = polar_express(update)
                update = update * max(update.size(-2), update.size(-1)) ** 0.5

                update = undo_split(update, original_shape, split_info[3])

                rms_sum += update.norm() * self.lr_multiplier / update.numel() ** 0.5
                rms_cnt += 1

                p.mul_(1 - base_lr * self.wd)
                p.add_(update.to(p.dtype), alpha=-muon_lr)

        self.last_update_rms = (rms_sum / rms_cnt).item() if rms_cnt > 0 else 0.0

    def state_dict(self):
        return {name: {k: v.cpu() for k, v in s.items()} for name, s in self.states.items()}

    def load_state_dict(self, saved, device):
        for name, tensors in saved.items():
            if name in self.states:
                for k, v in tensors.items():
                    self.states[name][k].copy_(v.to(device))


@torch.compile
def inv_quarter_sandwich(L, M, R):
    """Newton-Schulz iteration for L^{-1/4} @ M @ R^{-1/4}, 4 steps, fp32.
    Supports batched 3D input: L (B,m,m), M (B,m,n), R (B,n,n)."""
    assert L.ndim in (2, 3) and M.ndim == L.ndim and R.ndim == L.ndim
    eps = 1e-4
    M = M.float()

    m = L.size(-1)
    n = R.size(-1)
    I_L = torch.eye(m, device=L.device)
    I_R = torch.eye(n, device=L.device)

    tL = torch.sqrt((L * L.mT).sum(dim=(-2, -1), keepdim=True))
    tR = torch.sqrt((R * R.mT).sum(dim=(-2, -1), keepdim=True))
    L = L / tL + eps * I_L
    R = R / tR + eps * I_R

    for a, b, c in _NS_COEFFS_R4_SCALED:
        L2 = L @ L
        WL = a * I_L + b * L + c * L2

        R2 = R @ R
        WR = a * I_R + b * R + c * R2

        M = WL @ M @ WR

        WL4 = (WL @ WL) @ (WL @ WL)
        WR4 = (WR @ WR) @ (WR @ WR)
        L = L @ WL4
        R = R @ WR4

    M = M * (tL ** (-0.25)) * (tR ** (-0.25))
    return M


def compute_split_info(name, m, n, use_te=False, num_heads=0):
    """Compute how a parameter will be split for Muon/Shampoo processing.

    Returns: (n_chunks, chunk_m, chunk_n, headwise_permuted)
    """
    # TE fused param split (fc1_weight, fused QKV)
    if use_te and "fc1_weight" in name:
        return (2, m // 2, n, False)
    if use_te and "qkv" in name and m == 3 * n:
        if num_heads > 0:
            nc = 3 * num_heads
            return (nc, m // nc, n, False)
        return (3, m // 3, n, False)
    # Head-wise split for non-TE attention projections
    if num_heads > 0:
        if any(tag in name for tag in ("q_proj", "k_proj", "v_proj",
                                        "query_weight", "key_weight", "value_weight")):
            return (num_heads, m // num_heads, n, False)
    return (1, m, n, False)


def flatten_and_split(tensor, split_info):
    """4D->2D flatten + chunk/head split based on split_info."""
    n_chunks, chunk_m, chunk_n, headwise_permuted = split_info
    if tensor.ndim == 4:
        tensor = tensor.view(tensor.size(0), -1)
    if headwise_permuted:
        tensor = tensor.view(tensor.size(0), n_chunks, chunk_n).permute(1, 0, 2).contiguous()
    elif n_chunks > 1:
        tensor = tensor.view(n_chunks, chunk_m, chunk_n)
    return tensor


def undo_split(tensor, original_shape, headwise_permuted):
    """Undo flatten_and_split, restoring original shape."""
    if headwise_permuted:
        tensor = tensor.permute(1, 0, 2).contiguous()
    return tensor.view(original_shape)


class ShampooOptimizer:
    """Shampoo optimizer: L/R preconditioner EMA + matrix inverse root."""

    def __init__(self, named_params, lr_multiplier, momentum, wd, beta2=0.999, device="cuda",
                 use_te=False, num_heads=0):
        self.named_params = named_params
        self.lr_multiplier = lr_multiplier
        self.momentum = momentum
        self.wd = wd
        self.beta2 = beta2
        self.step_count = 0
        self._device = device
        self.use_te = use_te
        self.num_heads = num_heads
        self.last_precond_rms = 0.0
        # Pre-compute split info and init state per param
        self._split_info = {}
        self.states = {}
        for name, p in named_params.items():
            if p.ndim >= 2:
                m, n = p.shape[0], p.shape[1:].numel()
            else:
                m, n = p.shape[0], 1
            self._split_info[name] = compute_split_info(name, m, n, use_te, num_heads)
            n_chunks, chunk_m, chunk_n, _ = self._split_info[name]
            if n_chunks > 1:
                state = {
                    "momentum": torch.zeros_like(p),
                    "L": torch.zeros(n_chunks, chunk_m, chunk_m, dtype=torch.float32, device=device),
                    "R": torch.zeros(n_chunks, chunk_n, chunk_n, dtype=torch.float32, device=device),
                }
            else:
                state = {
                    "momentum": torch.zeros_like(p),
                    "L": torch.zeros(m, m, dtype=torch.float32, device=device),
                    "R": torch.zeros(n, n, dtype=torch.float32, device=device),
                }
            self.states[name] = state

    def step(self, base_lr):
        self.step_count += 1
        shampoo_lr = base_lr * self.lr_multiplier
        bias_corr1 = 1 - self.momentum ** self.step_count
        bias_corr2 = 1 - self.beta2 ** self.step_count
        rms_sum = torch.tensor(0.0, device=self._device)
        rms_cnt = 0
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            with torch.no_grad():
                for name, p in self.named_params.items():
                    if p.grad is None:
                        continue
                    assert p.grad.ndim in (2, 4), f"Shampoo only supports 2D/4D params, got ndim={p.grad.ndim}"
                    state = self.states[name]
                    grad = p.grad
                    original_shape = grad.shape
                    split_info = self._split_info[name]

                    state["momentum"].lerp_(grad, 1 - self.momentum)
                    grad_2d = flatten_and_split(grad, split_info)
                    momentum_2d_hat = flatten_and_split(state["momentum"], split_info) / bias_corr1

                    state["L"].lerp_(grad_2d @ grad_2d.mT, 1 - self.beta2)
                    state["R"].lerp_(grad_2d.mT @ grad_2d, 1 - self.beta2)

                    precond = inv_quarter_sandwich(
                        state["L"] / bias_corr2, momentum_2d_hat, state["R"] / bias_corr2,
                    )
                    precond = precond * max(precond.size(-2), precond.size(-1)) ** 0.5

                    rms_sum += precond.norm() * self.lr_multiplier / precond.numel() ** 0.5
                    rms_cnt += 1

                    precond = undo_split(precond, original_shape, split_info[3])

                    p.mul_(1 - base_lr * self.wd)
                    p.add_(precond.to(p.dtype), alpha=-shampoo_lr)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

        self.last_precond_rms = (rms_sum / rms_cnt).item() if rms_cnt > 0 else 0.0

    def state_dict(self):
        result = {name: {k: v.cpu() for k, v in s.items()} for name, s in self.states.items()}
        result["__step_count__"] = self.step_count
        return result

    def load_state_dict(self, saved, device):
        self.step_count = saved.get("__step_count__", 0)
        for name, tensors in saved.items():
            if name == "__step_count__":
                continue
            if name in self.states:
                for k, v in tensors.items():
                    self.states[name][k].copy_(v.to(device))


