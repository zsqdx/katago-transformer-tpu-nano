"""Transformer model architecture for KataGo nano training."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import get_num_bin_input_features, get_num_global_input_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXTRA_SCORE_DISTR_RADIUS = 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class SoftPlusWithGradientFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, grad_floor, square):
        ctx.save_for_backward(x)
        ctx.grad_floor = grad_floor
        if square:
            return torch.square(F.softplus(0.5 * x))
        else:
            return F.softplus(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_x = grad_output * (ctx.grad_floor + (1.0 - ctx.grad_floor) / (1.0 + torch.exp(-x)))
        return grad_x, None, None


def cross_entropy(pred_logits, target_probs, dim):
    return -torch.sum(target_probs * F.log_softmax(pred_logits, dim=dim), dim=dim)


# ---------------------------------------------------------------------------
# 2D RoPE
# ---------------------------------------------------------------------------


def precompute_freqs_cos_sin_2d(dim: int, pos_len: int, theta: float = 100.0):
    assert dim % 4 == 0
    dim_half = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, dim_half, 2).float() / dim_half))
    t = torch.arange(pos_len, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(t, t, indexing="ij")
    emb_h = grid_h.unsqueeze(-1) * freqs
    emb_w = grid_w.unsqueeze(-1) * freqs
    emb = torch.cat([emb_h, emb_w], dim=-1).flatten(0, 1)
    return emb.reshape(pos_len * pos_len, 1, 1, dim_half)


def apply_rotary_emb(xq, xk, cos, sin):
    """Apply rotary position embedding (computed in FP32 for numerical stability).
    cos, sin: (L, 1, 1, D) for standard RoPE or (1, L, H, D) for learnable RoPE.
    Both are reshaped to (1, L, ?, D) via view for broadcasting with (B, L, H, D).
    """
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    orig_dtype = xq.dtype
    with torch.amp.autocast(xq.device.type, enabled=False):
        xq, xk = xq.float(), xk.float()
        cos = cos.float().view(1, xq.shape[1], -1, xq.shape[-1])
        sin = sin.float().view(1, xq.shape[1], -1, xq.shape[-1])
        xq_out = xq * cos + rotate_half(xq) * sin
        xk_out = xk * cos + rotate_half(xk) * sin
    return xq_out.to(orig_dtype), xk_out.to(orig_dtype)


class RMSNormFP32(nn.Module):
    """RMSNorm that always runs in float32 (disables autocast)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=eps)

    def forward(self, x):
        with torch.amp.autocast(x.device.type, enabled=False):
            return self.norm(x.float()).to(x.dtype)


class ZeroCenteredRMSNormFP32(nn.Module):
    """Zero-Centered RMSNorm in FP32.

    Weight initialized to 0; forward uses (1 + weight) * rms_norm(x).
    Weight decay pushes weight toward 0, i.e. gamma toward 1.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        with torch.amp.autocast(x.device.type, enabled=False):
            x_f32 = x.float()
            mean_sq = (x_f32 * x_f32).mean(-1, keepdim=True)
            inv_rms = torch.rsqrt(mean_sq + self.eps)
            return ((1.0 + self.weight.float()) * (x_f32 * inv_rms)).to(x.dtype)


# ---------------------------------------------------------------------------
# Transformer Block (NLC format, RoPE + MHA + SwiGLU + RMSNorm)
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, c_main: int, num_heads: int, ffn_dim: int,
                 use_gated_attn: bool = False,
                 zero_centered_norm: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = c_main // num_heads
        self.ffn_dim = ffn_dim

        self.q_proj = nn.Linear(c_main, c_main, bias=False)
        self.k_proj = nn.Linear(c_main, c_main, bias=False)
        self.v_proj = nn.Linear(c_main, c_main, bias=False)
        self.out_proj = nn.Linear(c_main, c_main, bias=False)

        # SwiGLU FFN
        self.ffn_w1 = nn.Linear(c_main, ffn_dim, bias=False)
        self.ffn_wgate = nn.Linear(c_main, ffn_dim, bias=False)
        self.ffn_w2 = nn.Linear(ffn_dim, c_main, bias=False)

        NormClass = ZeroCenteredRMSNormFP32 if zero_centered_norm else RMSNormFP32
        self.norm1 = NormClass(c_main, eps=1e-6)
        self.norm2 = NormClass(c_main, eps=1e-6)

        self.use_gated_attn = use_gated_attn
        if use_gated_attn:
            self.attn_gate_proj = nn.Linear(c_main, c_main, bias=False)

    def forward(self, x, rope_cos, rope_sin, attn_mask=None):
        """
        x: (N, L, C)
        rope_cos, rope_sin: precomputed RoPE cos/sin, either
            (L, 1, 1, D) for standard or (1, L, H, D) for learnable
        attn_mask: optional (N, 1, 1, L) additive mask, 0 for valid, -inf for padding
        """
        B, L, C = x.shape

        # Prenorm: x = x + sublayer(norm(x))
        x_normed = self.norm1(x)

        q = self.q_proj(x_normed).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x_normed).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x_normed).view(B, L, self.num_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, rope_cos, rope_sin)

        # SDPA: (B, H, S, D)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, L, C)
        if self.use_gated_attn:
            attn_out = torch.sigmoid(self.attn_gate_proj(x_normed)) * attn_out
        x = x + self.out_proj(attn_out)

        # SwiGLU FFN:
        # keep the gate product in FP32 for numerical stability, but let W2
        # follow the ambient autocast / AMP precision.
        x_normed = self.norm2(x)
        w1_out = F.silu(self.ffn_w1(x_normed))
        wgate_out = self.ffn_wgate(x_normed)
        with torch.amp.autocast(x.device.type, enabled=False):
            ffn_hidden = (w1_out.float() * wgate_out.float()).to(x.dtype)
        x = x + self.ffn_w2(ffn_hidden)
        return x


# ---------------------------------------------------------------------------
# PolicyHead (NLC input)
# ---------------------------------------------------------------------------
class PolicyHead(nn.Module):
    """Per-position projection (board moves) + global pooling projection (pass)."""
    def __init__(self, c_in, pos_len):
        super().__init__()
        self.pos_len = pos_len
        self.num_policy_outputs = 6
        self.linear_board = nn.Linear(c_in, self.num_policy_outputs, bias=True)
        self.linear_pass = nn.Linear(c_in, self.num_policy_outputs, bias=True)

    def forward(self, x_nlc, mask=None):
        """
        x_nlc: (N, L, C)
        mask: optional (N, L) float, 1=valid, 0=padding
        """
        N, L, _ = x_nlc.shape
        board = self.linear_board(x_nlc).permute(0, 2, 1)  # (N, 6, L)
        if mask is not None:
            # Mask-aware global average pooling for pass logits
            mask_expanded = mask.unsqueeze(-1)  # (N, L, 1)
            pooled = (x_nlc * mask_expanded).sum(dim=1) / mask.sum(dim=1, keepdim=True)
            # Mask out invalid board positions with large negative number
            board = board - (1.0 - mask.unsqueeze(1)) * 5000.0  # (N, 6, L)
        else:
            pooled = x_nlc.mean(dim=1)
        pass_logits = self.linear_pass(pooled)  # (N, 6)
        return torch.cat([board, pass_logits.unsqueeze(-1)], dim=2)  # (N, 6, L+1)


# ---------------------------------------------------------------------------
# ValueHead (NLC input, per-position + mean-pool projection)
# ---------------------------------------------------------------------------
class ValueHead(nn.Module):
    def __init__(self, c_in, num_scorebeliefs, pos_len, score_mode="mixop"):
        super().__init__()
        self.pos_len = pos_len
        self.scorebelief_mid = pos_len * pos_len + EXTRA_SCORE_DISTR_RADIUS
        self.scorebelief_len = self.scorebelief_mid * 2
        self.num_scorebeliefs = num_scorebeliefs
        self.score_mode = score_mode

        # Per-position: ownership(1) + scoring(1) + futurepos(2) + seki(4)
        # Global (mean-pool): value(3) + misc(10) + moremisc(8)
        self.n_spatial = 1 + 1 + 2 + 4  # 8
        self.n_global = 3 + 10 + 8      # 21
        self.linear_sv = nn.Linear(c_in, self.n_spatial + self.n_global, bias=True)

        # Score belief head
        if score_mode == "simple":
            self.linear_s_simple = nn.Linear(c_in, self.scorebelief_len, bias=True)
        elif score_mode == "mix":
            self.linear_s_mix = nn.Linear(c_in, self.scorebelief_len * num_scorebeliefs + num_scorebeliefs, bias=True)
        elif score_mode == "mixop":
            self.linear_s_mix = nn.Linear(c_in, self.scorebelief_len * num_scorebeliefs + num_scorebeliefs, bias=True)
            self.linear_s2off = nn.Linear(1, num_scorebeliefs, bias=True)
            self.linear_s2par = nn.Linear(1, num_scorebeliefs, bias=True)

        self.register_buffer("score_belief_offset_vector", torch.tensor(
            [(float(i - self.scorebelief_mid) + 0.5) for i in range(self.scorebelief_len)],
            dtype=torch.float32,
        ), persistent=False)
        if score_mode == "mixop":
            self.register_buffer("score_belief_offset_bias_vector", torch.tensor(
                [0.05 * (float(i - self.scorebelief_mid) + 0.5) for i in range(self.scorebelief_len)],
                dtype=torch.float32,
            ), persistent=False)
            self.register_buffer("score_belief_parity_vector", torch.tensor(
                [0.5 - float((i - self.scorebelief_mid) % 2) for i in range(self.scorebelief_len)],
                dtype=torch.float32,
            ), persistent=False)

    def forward(self, x_nlc, score_parity, mask=None):
        """
        x_nlc: (N, L, C)
        score_parity: (N, 1)
        mask: optional (N, L) float, 1=valid, 0=padding
        """
        N, L, _ = x_nlc.shape
        H = W = self.pos_len

        spatial_global = self.linear_sv(x_nlc)
        spatial, global_feats = spatial_global.split([self.n_spatial, self.n_global], dim=-1)

        spatial = spatial.permute(0, 2, 1).view(N, self.n_spatial, H, W)
        if mask is not None:
            spatial = spatial * mask.view(N, 1, H, W)
        out_ownership, out_scoring, out_futurepos, out_seki = spatial.split([1, 1, 2, 4], dim=1)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # (N, L, 1)
            mask_sum = mask.sum(dim=1, keepdim=True)  # (N, 1)
            global_feats = (global_feats * mask_expanded).sum(dim=1) / mask_sum
        else:
            global_feats = global_feats.mean(dim=1)
        out_value, out_misc, out_moremisc = global_feats.split([3, 10, 8], dim=-1)

        # Score belief: mask-aware mean-pool then project
        if mask is not None:
            pooled_s = (x_nlc * mask_expanded).sum(dim=1) / mask_sum
        else:
            pooled_s = x_nlc.mean(dim=1)
        if self.score_mode == "simple":
            out_scorebelief_logprobs = F.log_softmax(self.linear_s_simple(pooled_s), dim=1)
        elif self.score_mode in ("mix", "mixop"):
            score_proj = self.linear_s_mix(pooled_s)
            belief_logits, mix_logits = score_proj.split(
                [self.scorebelief_len * self.num_scorebeliefs, self.num_scorebeliefs], dim=-1
            )
            belief_logits = belief_logits.view(N, self.scorebelief_len, self.num_scorebeliefs)
            if self.score_mode == "mixop":
                belief_logits = (
                    belief_logits
                    + self.linear_s2off(self.score_belief_offset_bias_vector.view(1, self.scorebelief_len, 1))
                    + self.linear_s2par(
                        (self.score_belief_parity_vector.view(1, self.scorebelief_len) * score_parity)
                        .view(N, self.scorebelief_len, 1)
                    )
                )
            mix_log_weights = F.log_softmax(mix_logits, dim=1)
            out_scorebelief_logprobs = F.log_softmax(belief_logits, dim=1)
            out_scorebelief_logprobs = torch.logsumexp(
                out_scorebelief_logprobs + mix_log_weights.view(-1, 1, self.num_scorebeliefs), dim=2
            )

        return (
            out_value, out_misc, out_moremisc,
            out_ownership, out_scoring, out_futurepos, out_seki,
            out_scorebelief_logprobs,
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, config: dict, pos_len: int, score_mode: str = "mixop", varlen: bool = False,
                 gated_attn: bool = False, zero_centered_norm: bool = False,
                 learnable_rope: bool = False):
        super().__init__()
        self.config = config
        self.pos_len = pos_len
        self.varlen = varlen
        self.gated_attn = gated_attn
        self.zero_centered_norm = zero_centered_norm
        self.c_trunk = config["hidden_size"]
        num_bin_features = get_num_bin_input_features(config)
        num_global_features = get_num_global_input_features(config)

        num_heads = config["num_heads"]
        ffn_dim = config["ffn_dim"]
        head_dim = self.c_trunk // num_heads

        # Stem
        self.conv_spatial = nn.Conv2d(num_bin_features, self.c_trunk,
                                      kernel_size=3, padding="same", bias=False)
        self.linear_global = nn.Linear(num_global_features, self.c_trunk, bias=False)

        # RoPE: fixed precomputed (default) or learnable per-head frequencies
        self.learnable_rope = learnable_rope
        if not learnable_rope:
            emb = precompute_freqs_cos_sin_2d(head_dim, pos_len)
            emb_expanded = torch.cat([emb, emb], dim=-1)
            self.register_buffer("rope_cos", emb_expanded.cos(), persistent=False)
            self.register_buffer("rope_sin", emb_expanded.sin(), persistent=False)
        else:
            self.rope_cos = None
            self.rope_sin = None
            # Precompute 2D grid coordinates: (L, 2) with [col, row] per position
            L = pos_len * pos_len
            idx = torch.arange(L)
            pos_xy = torch.stack([(idx % pos_len).float(),
                                  (idx // pos_len).float()], dim=-1)  # (L, 2)
            self.register_buffer("pos_xy", pos_xy, persistent=False)
            # Learnable per-head RoPE frequencies for all layers: (num_layers, H, P, 2)
            num_layers = config["num_layers"]
            P = head_dim // 2
            assert head_dim % 2 == 0, f"Head dim must be even for learnable RoPE, got {head_dim}"
            log_lo = math.log(1.0 / 50.0)
            log_hi = math.log(1.0)
            init_freqs = torch.exp(torch.empty(num_layers, num_heads, P, 2).uniform_(log_lo, log_hi))
            init_freqs = init_freqs * (torch.randint(0, 2, (num_layers, num_heads, P, 2)) * 2 - 1).float()
            self.all_rope_freqs = nn.Parameter(init_freqs)

        # Transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(config["num_layers"]):
            self.blocks.append(TransformerBlock(
                c_main=self.c_trunk,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                use_gated_attn=gated_attn,
                zero_centered_norm=zero_centered_norm,
            ))

        NormClass = ZeroCenteredRMSNormFP32 if zero_centered_norm else RMSNormFP32
        self.norm_final = NormClass(self.c_trunk, eps=1e-6)

        # Output heads
        num_scorebeliefs = config["num_scorebeliefs"]

        self.policy_head = PolicyHead(self.c_trunk, pos_len)
        self.value_head = ValueHead(self.c_trunk, num_scorebeliefs, pos_len, score_mode=score_mode)

        # Seki dynamic weight moving average state
        self.moving_unowned_proportion_sum = 0.0
        self.moving_unowned_proportion_weight = 0.0

    def initialize(self, init_std=0.02):
        """Weight initialization.

        All Linear/Conv layers use fixed init_std.
        Output layers (out_proj, ffn_w2) additionally scale by 1/sqrt(2*num_blocks).
        Learnable RoPE frequencies keep their geometric/random initialization.
        """
        num_blocks = len(self.blocks)

        for name, p in self.named_parameters():
            if "rope_freqs" in name:
                continue
            if p.dim() < 2:
                if "norm" not in name:
                    nn.init.zeros_(p)
            else:
                std = init_std
                if ".out_proj." in name or ".ffn_w2." in name:
                    std = std / math.sqrt(2.0 * num_blocks)
                nn.init.normal_(p, mean=0.0, std=std)

    def fuse_zero_centered_norm(self):
        """Fuse zero-centered norm: replace ZeroCenteredRMSNormFP32 with RMSNormFP32.

        Each module's weight is replaced by weight + 1, producing standard RMSNorm behavior.
        """
        for name, module in list(self.named_modules()):
            if isinstance(module, ZeroCenteredRMSNormFP32):
                new_norm = RMSNormFP32(module.weight.shape[0], eps=module.eps)
                new_norm.norm.weight.data.copy_(module.weight.data + 1.0)
                parts = name.split(".")
                parent = self
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], new_norm)
        self.zero_centered_norm = False

    def _forward_trunk_impl(self, input_spatial, input_global):
        x, attn_mask, mask_flat = self._forward_stem_impl(input_spatial, input_global)
        x = self._forward_blocks_impl(x, attn_mask=attn_mask)
        return x, mask_flat

    def _compute_all_learnable_rope(self):
        """Batch-compute cos/sin for all layers in one fused operation.
        Returns all_cos, all_sin each of shape (num_layers, L, H, D).
        """
        # pos_xy: (L, 2), all_rope_freqs: (num_layers, H, P, 2)
        # Explicit broadcast mul-add: (1,L,1,1)*(N,1,H,P) -> (N,L,H,P)
        sx = self.pos_xy[:, 0]                        # (L,)
        sy = self.pos_xy[:, 1]                        # (L,)
        omega_x = self.all_rope_freqs[:, :, :, 0]    # (N, H, P)
        omega_y = self.all_rope_freqs[:, :, :, 1]    # (N, H, P)
        all_angles = (sx[None, :, None, None] * omega_x[:, None, :, :]
                    + sy[None, :, None, None] * omega_y[:, None, :, :])
        all_angles = torch.cat([all_angles, all_angles], dim=-1)  # (num_layers, L, H, D)
        return all_angles.cos(), all_angles.sin()

    def _forward_blocks_impl(self, x, attn_mask=None):
        if self.learnable_rope:
            all_cos, all_sin = self._compute_all_learnable_rope()
            for i, block in enumerate(self.blocks):
                x = block(x, all_cos[i:i+1], all_sin[i:i+1], attn_mask=attn_mask)
        else:
            for block in self.blocks:
                x = block(x, self.rope_cos, self.rope_sin, attn_mask=attn_mask)
        return self.norm_final(x)

    def _forward_stem_impl(self, input_spatial, input_global):
        N = input_spatial.shape[0]
        H = W = self.pos_len
        L = H * W

        # Extract mask from channel 0 when varlen is enabled
        if self.varlen:
            mask = input_spatial[:, 0:1, :, :].contiguous()  # (N, 1, H, W)
            mask_flat = mask.view(N, L)  # (N, L)
        else:
            mask_flat = None

        # Stem: NCHW -> NLC
        x_global = self.linear_global(input_global)
        x_spatial = self.conv_spatial(input_spatial)
        stem_nchw = x_spatial + x_global.unsqueeze(-1).unsqueeze(-1)

        x = stem_nchw.view(N, self.c_trunk, L).permute(0, 2, 1)

        # Additive attention mask in x.dtype (fp16/bf16 under autocast)
        if self.varlen:
            attn_mask = torch.zeros(N, 1, 1, L, device=x.device, dtype=x.dtype)
            attn_mask.masked_fill_(mask_flat.view(N, 1, 1, L) == 0, float('-inf'))
        else:
            attn_mask = None

        return x, attn_mask, mask_flat

    def forward_stem_for_onnx_export(self, input_spatial, input_global):
        x, attn_mask, mask_flat = self._forward_stem_impl(input_spatial, input_global)
        if self.varlen:
            return x.float(), mask_flat.float()
        return x.float()

    def forward_blocks_for_onnx_export(self, input_stem, mask_flat=None):
        if self.varlen and mask_flat is not None:
            N, L = mask_flat.shape
            attn_mask = torch.zeros(N, 1, 1, L, device=mask_flat.device, dtype=input_stem.dtype)
            attn_mask.masked_fill_(mask_flat.view(N, 1, 1, L) == 0, float('-inf'))
            return self._forward_blocks_impl(input_stem, attn_mask=attn_mask).float()
        return self._forward_blocks_impl(input_stem).float()

    def forward_trunk_for_onnx_export(self, input_spatial, input_global):
        x, mask_flat = self._forward_trunk_impl(input_spatial, input_global)
        if self.varlen:
            return x.float(), mask_flat.float()
        return x.float()

    def forward(self, input_spatial, input_global):
        """
        input_spatial: (N, C_bin, H, W)
        input_global:  (N, C_global)
        """
        x, mask_flat = self._forward_trunk_impl(input_spatial, input_global)

        # Output heads in fp32.
        with torch.amp.autocast(x.device.type, enabled=False):
            x_fp32 = x.float()
            out_policy = self.policy_head(x_fp32, mask=mask_flat)
            (
                out_value, out_misc, out_moremisc,
                out_ownership, out_scoring, out_futurepos, out_seki,
                out_scorebelief,
            ) = self.value_head(x_fp32, input_global[:, -1:].float(), mask=mask_flat)

        return (
            out_policy, out_value, out_misc, out_moremisc,
            out_ownership, out_scoring, out_futurepos, out_seki,
            out_scorebelief,
        )

    def postprocess(self, outputs):
        (
            out_policy, out_value, out_misc, out_moremisc,
            out_ownership, out_scoring, out_futurepos, out_seki,
            out_scorebelief,
        ) = outputs

        td_score_multiplier = 20.0
        scoremean_multiplier = 20.0
        scorestdev_multiplier = 20.0
        lead_multiplier = 20.0
        variance_time_multiplier = 40.0
        shortterm_value_error_multiplier = 0.25
        shortterm_score_error_multiplier = 150.0

        policy_logits = out_policy
        value_logits = out_value
        td_value_logits = torch.stack(
            (out_misc[:, 4:7], out_misc[:, 7:10], out_moremisc[:, 2:5]), dim=1
        )
        pred_td_score = out_moremisc[:, 5:8] * td_score_multiplier
        ownership_pretanh = out_ownership
        pred_scoring = out_scoring
        futurepos_pretanh = out_futurepos
        seki_logits = out_seki
        pred_scoremean = out_misc[:, 0] * scoremean_multiplier
        pred_scorestdev = SoftPlusWithGradientFloor.apply(out_misc[:, 1], 0.05, False) * scorestdev_multiplier
        pred_lead = out_misc[:, 2] * lead_multiplier
        pred_variance_time = SoftPlusWithGradientFloor.apply(out_misc[:, 3], 0.05, False) * variance_time_multiplier

        pred_shortterm_value_error = SoftPlusWithGradientFloor.apply(out_moremisc[:, 0], 0.05, True) * shortterm_value_error_multiplier
        pred_shortterm_score_error = SoftPlusWithGradientFloor.apply(out_moremisc[:, 1], 0.05, True) * shortterm_score_error_multiplier

        scorebelief_logits = out_scorebelief

        return (
            policy_logits, value_logits, td_value_logits, pred_td_score,
            ownership_pretanh, pred_scoring, futurepos_pretanh, seki_logits,
            pred_scoremean, pred_scorestdev, pred_lead, pred_variance_time,
            pred_shortterm_value_error, pred_shortterm_score_error,
            scorebelief_logits,
        )
