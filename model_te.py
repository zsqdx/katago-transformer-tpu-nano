"""Transformer model architecture for KataGo nano training (TransformerEngine version).

This module provides the same Model API as model.py but uses NVIDIA TransformerEngine
(te.TransformerLayer) for fused kernels and optional FP8 training support.

All weights are directly compatible with model.py via checkpoint conversion utilities.

Usage:
    - Drop-in replacement for model.py's Model class
    - Requires: pip install transformer-engine[pytorch]
    - FP8 requires Hopper (H100/H200) or Ada (RTX 4090) GPU
    - Non-FP8 mode still benefits from TE's fused kernels

Torch compile note:
    - TE kernels currently call several PyCapsule ops that torch._dynamo cannot trace.
    - The TE trunk is isolated from torch.compile via @torch._dynamo.disable to avoid graph-break warnings.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te

from configs import get_num_bin_input_features, get_num_global_input_features
from model import (
    EXTRA_SCORE_DISTR_RADIUS,
    RMSNormFP32,
    ZeroCenteredRMSNormFP32,
    SoftPlusWithGradientFloor,
    apply_rotary_emb,
    cross_entropy,
    precompute_freqs_cos_sin_2d,
    PolicyHead,
    ValueHead,
)


# ---------------------------------------------------------------------------
# Transformer block: te.TransformerLayer (complete fused block)
# ---------------------------------------------------------------------------
class TransformerBlockTE(nn.Module):
    """Uses te.TransformerLayer for the entire block including residual connections.

    RoPE is handled internally by TE using rotate_half.
    """

    def __init__(self, c_main: int, num_heads: int, ffn_dim: int,
                 init_method=None, output_layer_init_method=None,
                 zero_centered_gamma: bool = False):
        super().__init__()
        self.layer = te.TransformerLayer(
            c_main, ffn_dim, num_heads,
            layernorm_epsilon=1e-6,
            hidden_dropout=0,
            attention_dropout=0,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            self_attn_mask_type="no_mask",
            normalization="RMSNorm",
            bias=False,
            activation="swiglu",
            attn_input_format="bshd",
            zero_centered_gamma=zero_centered_gamma,
        )

    def forward(self, x, rope, attn_mask=None):
        """
        x: (N, L, C)
        rope: (L, 1, 1, dim_half) raw RoPE embeddings for TE
        attn_mask: optional (N, 1, 1, L) bool, True=masked
        """
        if attn_mask is not None:
            return self.layer(x, rotary_pos_emb=rope,
                              attention_mask=attn_mask,
                              self_attn_mask_type="arbitrary")
        return self.layer(x, rotary_pos_emb=rope)


class TransformerBlockTEDecomposed(nn.Module):
    """Export-only TE block with manual RoPE outside TE custom fused kernels."""

    def __init__(self, c_main: int, num_heads: int, ffn_dim: int,
                 zero_centered_gamma: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = c_main // num_heads
        self.ln_qkv = te.LayerNormLinear(
            c_main,
            3 * c_main,
            eps=1e-6,
            bias=False,
            normalization="RMSNorm",
            zero_centered_gamma=zero_centered_gamma,
        )
        self.attention = te.DotProductAttention(
            num_attention_heads=num_heads,
            kv_channels=self.head_dim,
            attention_dropout=0.0,
            attn_mask_type="no_mask",
            qkv_format="bshd",
        )
        self.proj = te.Linear(c_main, c_main, bias=False)
        self.ln_mlp = te.LayerNormMLP(
            c_main,
            ffn_dim,
            eps=1e-6,
            bias=False,
            normalization="RMSNorm",
            activation="swiglu",
            zero_centered_gamma=zero_centered_gamma,
        )

    def forward(self, x, rope_cos, rope_sin, attn_mask=None):
        batch_size, seq_len, _ = x.shape
        residual = x
        qkv = self.ln_qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k = apply_rotary_emb(q, k, rope_cos, rope_sin)
        attn_kwargs = {}
        if attn_mask is not None:
            attn_kwargs["attention_mask"] = attn_mask
            attn_kwargs["attn_mask_type"] = "arbitrary"
        x = residual + self.proj(self.attention(q, k, v, **attn_kwargs))
        return x + self.ln_mlp(x)


class TransformerBlockTEHybrid(nn.Module):
    """TE block with fused linear kernels but PyTorch SDPA for attention.

    Uses te.LayerNormLinear for fused LayerNorm+QKV, te.LayerNormMLP for fused
    LayerNorm+SwiGLU FFN, but F.scaled_dot_product_attention for attention.
    Weight layout is identical to TransformerBlockTEDecomposed.
    """

    def __init__(self, c_main: int, num_heads: int, ffn_dim: int,
                 zero_centered_gamma: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = c_main // num_heads

        self.ln_qkv = te.LayerNormLinear(
            c_main,
            3 * c_main,
            eps=1e-6,
            bias=False,
            normalization="RMSNorm",
            zero_centered_gamma=zero_centered_gamma,
        )
        self.proj = te.Linear(c_main, c_main, bias=False)
        self.ln_mlp = te.LayerNormMLP(
            c_main,
            ffn_dim,
            eps=1e-6,
            bias=False,
            normalization="RMSNorm",
            activation="swiglu",
            zero_centered_gamma=zero_centered_gamma,
        )

    def forward(self, x, rope_cos, rope_sin, attn_mask=None):
        B, L, _ = x.shape

        # Prenorm
        residual = x
        qkv = self.ln_qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)           # (B, L, H, D)
        q, k = apply_rotary_emb(q, k, rope_cos, rope_sin)
        # SDPA expects (B, H, L, D)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, L, -1)
        x = residual + self.proj(attn_out)
        return x + self.ln_mlp(x)


def _replace_nn_linear_with_te(module):
    """Recursively replace nn.Linear with te.Linear in a module."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, te.Linear(
                child.in_features, child.out_features, bias=(child.bias is not None),
            ))
        else:
            _replace_nn_linear_with_te(child)


# ---------------------------------------------------------------------------
# Model (same API as model.py)
# ---------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, config: dict, pos_len: int, score_mode: str = "mixop",
                 use_fp8: bool = False, varlen: bool = False,
                 zero_centered_norm: bool = False, hybrid: bool = False,
                 learnable_rope: bool = False):
        super().__init__()
        self.config = config
        self.pos_len = pos_len
        self.varlen = varlen
        self.zero_centered_norm = zero_centered_norm
        self.hybrid = hybrid
        self.c_trunk = config["hidden_size"]
        num_bin_features = get_num_bin_input_features(config)
        num_global_features = get_num_global_input_features(config)

        num_heads = config["num_heads"]
        ffn_dim = config["ffn_dim"]
        head_dim = self.c_trunk // num_heads

        # Stem
        self.conv_spatial = nn.Conv2d(num_bin_features, self.c_trunk,
                                      kernel_size=3, padding="same", bias=False)
        # Non-FP8: use te.Linear for fused kernels; FP8: nn.Linear (dims not FP8-aligned)
        Linear = nn.Linear if use_fp8 else te.Linear
        self.linear_global = Linear(num_global_features, self.c_trunk, bias=False)

        # RoPE: fixed precomputed or learnable per-head frequencies
        self.learnable_rope = learnable_rope
        emb = precompute_freqs_cos_sin_2d(head_dim, pos_len)  # (L, 1, 1, dim_half)
        emb_full = torch.cat([emb, emb], dim=-1)              # (L, 1, 1, dim)
        if hybrid:
            if not learnable_rope:
                self.register_buffer("rope_cos", emb_full.cos(), persistent=False)
                self.register_buffer("rope_sin", emb_full.sin(), persistent=False)
            else:
                self.rope_cos = None
                self.rope_sin = None
                # Precompute 2D grid coordinates
                L = pos_len * pos_len
                idx = torch.arange(L)
                pos_xy = torch.stack([(idx % pos_len).float(),
                                      (idx // pos_len).float()], dim=-1)
                self.register_buffer("pos_xy", pos_xy, persistent=False)
                # Learnable per-head RoPE frequencies for all layers
                num_layers = config["num_layers"]
                P = head_dim // 2
                log_lo = math.log(1.0 / 50.0)
                log_hi = math.log(1.0)
                init_freqs = torch.exp(torch.empty(num_layers, num_heads, P, 2).uniform_(log_lo, log_hi))
                init_freqs = init_freqs * (torch.randint(0, 2, (num_layers, num_heads, P, 2)) * 2 - 1).float()
                self.all_rope_freqs = nn.Parameter(init_freqs)
        else:
            # Full TE mode: raw embeddings for TE's internal RoPE
            self.register_buffer("rope", emb_full, persistent=False)

        # Transformer blocks
        BlockClass = TransformerBlockTEHybrid if hybrid else TransformerBlockTE
        self.blocks = nn.ModuleList()
        block_kwargs = dict(
            c_main=self.c_trunk,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            zero_centered_gamma=zero_centered_norm,
        )
        for _ in range(config["num_layers"]):
            self.blocks.append(BlockClass(**block_kwargs))

        self.norm_final = te.RMSNorm(self.c_trunk, eps=1e-6,
                                      zero_centered_gamma=zero_centered_norm)

        # Output heads: non-FP8 uses te.Linear for fused kernels; FP8 keeps nn.Linear (dims not FP8-aligned)
        num_scorebeliefs = config["num_scorebeliefs"]
        self.policy_head = PolicyHead(self.c_trunk, pos_len)
        self.value_head = ValueHead(self.c_trunk, num_scorebeliefs, pos_len, score_mode=score_mode)
        if not use_fp8:
            _replace_nn_linear_with_te(self.policy_head)
            _replace_nn_linear_with_te(self.value_head)

        # Seki dynamic weight moving average state
        self.moving_unowned_proportion_sum = 0.0
        self.moving_unowned_proportion_weight = 0.0

    def _compute_all_learnable_rope(self):
        """Batch-compute cos/sin for all layers in one fused operation."""
        sx = self.pos_xy[:, 0]                        # (L,)
        sy = self.pos_xy[:, 1]                        # (L,)
        omega_x = self.all_rope_freqs[:, :, :, 0]    # (N, H, P)
        omega_y = self.all_rope_freqs[:, :, :, 1]    # (N, H, P)
        all_angles = (sx[None, :, None, None] * omega_x[:, None, :, :]
                    + sy[None, :, None, None] * omega_y[:, None, :, :])
        all_angles = torch.cat([all_angles, all_angles], dim=-1)
        return all_angles.cos(), all_angles.sin()

    def _run_trunk_impl(self, x, attn_mask=None):
        if self.hybrid:
            if self.learnable_rope:
                all_cos, all_sin = self._compute_all_learnable_rope()
                for i, block in enumerate(self.blocks):
                    x = block(x, all_cos[i:i+1], all_sin[i:i+1], attn_mask=attn_mask)
            else:
                for block in self.blocks:
                    x = block(x, self.rope_cos, self.rope_sin, attn_mask=attn_mask)
        else:
            for block in self.blocks:
                x = block(x, rope=self.rope, attn_mask=attn_mask)
        return self.norm_final(x)

    @torch._dynamo.disable
    def _run_trunk_no_compile(self, x, attn_mask=None):
        return self._run_trunk_impl(x, attn_mask=attn_mask)

    def initialize(self, init_std=0.02):
        """Weight initialization using TE-native init_method.

        All Linear/Conv layers use fixed init_std.
        Output layers additionally scale by 1/sqrt(2*num_blocks).
        """
        num_blocks = len(self.blocks)

        def init_fn(tensor):
            nn.init.normal_(tensor, mean=0.0, std=init_std)

        def output_init_fn(tensor):
            std = init_std / math.sqrt(2.0 * num_blocks)
            nn.init.normal_(tensor, mean=0.0, std=std)

        if self.hybrid:
            # Hybrid mode: iterate parameters directly (TransformerBlockTEHybrid
            # doesn't accept init_method, unlike TransformerBlockTE)
            for name, p in self.named_parameters():
                if not name.startswith("blocks."):
                    continue
                if "rope_freqs" in name:
                    continue
                if p.dim() < 2:
                    # 1D params: layer_norm / RMSNorm weights are left at default init
                    if "layer_norm" not in name and "norm" not in name:
                        nn.init.zeros_(p)
                else:
                    # Output layers: proj.weight (attn output), fc2_weight / ffn_w2.weight (FFN output)
                    if ".proj.weight" in name or "fc2_weight" in name or ".ffn_w2.weight" in name:
                        output_init_fn(p)
                    else:
                        init_fn(p)
        else:
            # Full TE mode: rebuild blocks with TE-native init methods
            num_heads = self.config["num_heads"]
            ffn_dim = self.config["ffn_dim"]
            self.blocks = nn.ModuleList([
                TransformerBlockTE(
                    c_main=self.c_trunk, num_heads=num_heads, ffn_dim=ffn_dim,
                    init_method=init_fn, output_layer_init_method=output_init_fn,
                    zero_centered_gamma=self.zero_centered_norm,
                )
                for _ in range(num_blocks)
            ])

        # Stem
        init_fn(self.conv_spatial.weight)
        init_fn(self.linear_global.weight)

        # Heads (nn.Linear from model.py)
        for m in (self.policy_head, self.value_head):
            for p in m.parameters():
                if p.dim() >= 2:
                    init_fn(p)
                else:
                    nn.init.zeros_(p)

    def forward_trunk_for_onnx_export(self, input_spatial, input_global):
        x, attn_mask, mask_flat = self._forward_stem_impl(input_spatial, input_global)
        x = self._forward_blocks_impl(x, attn_mask=attn_mask)
        if self.varlen:
            return x.float(), mask_flat.float()
        return x.float()

    def _forward_stem_impl(self, input_spatial, input_global):
        N = input_spatial.shape[0]
        L = self.pos_len * self.pos_len

        if self.varlen:
            mask_flat = input_spatial[:, 0, :, :].contiguous().view(N, L)  # (N, L) float
            if self.hybrid:
                # Hybrid: additive mask for SDPA (0=valid, -inf=masked)
                attn_mask = torch.zeros(N, 1, 1, L, device=input_spatial.device, dtype=input_spatial.dtype)
                attn_mask.masked_fill_(mask_flat.view(N, 1, 1, L) == 0, float('-inf'))
            else:
                # Full TE: boolean mask (True=masked)
                attn_mask = (mask_flat == 0).view(N, 1, 1, L)  # (N, 1, 1, L) bool
        else:
            attn_mask = None
            mask_flat = None

        x_global = self.linear_global(input_global)
        x_spatial = self.conv_spatial(input_spatial)
        x = x_spatial + x_global.unsqueeze(-1).unsqueeze(-1)
        x = x.view(N, self.c_trunk, L).permute(0, 2, 1)
        return x, attn_mask, mask_flat

    def forward_stem_for_onnx_export(self, input_spatial, input_global):
        x, attn_mask, mask_flat = self._forward_stem_impl(input_spatial, input_global)
        if self.varlen:
            return x.float(), mask_flat.float()
        return x.float()

    def _forward_blocks_impl(self, x, attn_mask=None):
        return self._run_trunk_impl(x, attn_mask=attn_mask)

    def forward_blocks_for_onnx_export(self, input_stem, mask_flat=None):
        if self.varlen and mask_flat is not None:
            N, L = mask_flat.shape
            attn_mask = (mask_flat == 0).view(N, 1, 1, L)
            return self._forward_blocks_impl(input_stem, attn_mask=attn_mask).float()
        return self._forward_blocks_impl(input_stem).float()

    def _forward_impl(self, input_spatial, input_global, for_onnx_export: bool):
        """
        input_spatial: (N, C_bin, H, W)
        input_global:  (N, C_global)
        """
        # Stem: NCHW -> NLC
        x, attn_mask, mask_flat = self._forward_stem_impl(input_spatial, input_global)

        # ONNX export needs the full TE graph visible to torch.export / torch.onnx.
        if for_onnx_export:
            x = self._forward_blocks_impl(x, attn_mask=attn_mask)
        elif self.hybrid:
            # Hybrid mode: let torch.compile trace through — TE ops cause graph breaks
            # but SDPA and surrounding tensor ops still get compiled.
            x = self._run_trunk_impl(x, attn_mask=attn_mask)
        else:
            # Full TE mode: isolate from torch.compile to avoid graph-break warnings.
            x = self._run_trunk_no_compile(x, attn_mask=attn_mask)

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

    def forward(self, input_spatial, input_global):
        return self._forward_impl(input_spatial, input_global, for_onnx_export=False)

    def forward_for_onnx_export(self, input_spatial, input_global):
        return self._forward_impl(input_spatial, input_global, for_onnx_export=True)

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


class ModelDecomposedExport(nn.Module):
    """Export-only TE model that keeps TE modules but applies RoPE via plain PyTorch ops."""

    def __init__(self, config: dict, pos_len: int, score_mode: str = "mixop",
                 use_fp8: bool = False, varlen: bool = False,
                 zero_centered_norm: bool = False):
        super().__init__()
        self.config = config
        self.pos_len = pos_len
        self.varlen = varlen
        self.zero_centered_norm = zero_centered_norm
        self.c_trunk = config["hidden_size"]
        num_bin_features = get_num_bin_input_features(config)
        num_global_features = get_num_global_input_features(config)

        num_heads = config["num_heads"]
        ffn_dim = config["ffn_dim"]
        head_dim = self.c_trunk // num_heads

        self.conv_spatial = nn.Conv2d(num_bin_features, self.c_trunk,
                                      kernel_size=3, padding="same", bias=False)
        Linear = nn.Linear if use_fp8 else te.Linear
        self.linear_global = Linear(num_global_features, self.c_trunk, bias=False)

        # Precompute RoPE embeddings (rotate_half)
        emb = precompute_freqs_cos_sin_2d(head_dim, pos_len)
        emb_expanded = torch.cat([emb, emb], dim=-1)
        self.register_buffer("rope_cos", emb_expanded.cos(), persistent=False)
        self.register_buffer("rope_sin", emb_expanded.sin(), persistent=False)

        self.blocks = nn.ModuleList([
            TransformerBlockTEDecomposed(
                c_main=self.c_trunk,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                zero_centered_gamma=zero_centered_norm,
            )
            for _ in range(config["num_layers"])
        ])
        self.norm_final = te.RMSNorm(self.c_trunk, eps=1e-6,
                                      zero_centered_gamma=zero_centered_norm)

        num_scorebeliefs = config["num_scorebeliefs"]
        self.policy_head = PolicyHead(self.c_trunk, pos_len)
        self.value_head = ValueHead(self.c_trunk, num_scorebeliefs, pos_len, score_mode=score_mode)
        if not use_fp8:
            _replace_nn_linear_with_te(self.policy_head)
            _replace_nn_linear_with_te(self.value_head)

        self.moving_unowned_proportion_sum = 0.0
        self.moving_unowned_proportion_weight = 0.0

    def _run_trunk_impl(self, x, attn_mask=None):
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin, attn_mask=attn_mask)
        return self.norm_final(x)

    def _forward_stem_impl(self, input_spatial, input_global):
        batch_size = input_spatial.shape[0]
        seq_len = self.pos_len * self.pos_len

        if self.varlen:
            mask_flat = input_spatial[:, 0, :, :].contiguous().view(batch_size, seq_len)
            attn_mask = (mask_flat == 0).view(batch_size, 1, 1, seq_len)
        else:
            attn_mask = None
            mask_flat = None

        x_global = self.linear_global(input_global)
        x_spatial = self.conv_spatial(input_spatial)
        x = x_spatial + x_global.unsqueeze(-1).unsqueeze(-1)
        x = x.view(batch_size, self.c_trunk, seq_len).permute(0, 2, 1)
        return x, attn_mask, mask_flat

    def forward_stem_for_onnx_export(self, input_spatial, input_global):
        x, attn_mask, mask_flat = self._forward_stem_impl(input_spatial, input_global)
        if self.varlen:
            return x.float(), mask_flat.float()
        return x.float()

    def _forward_blocks_impl(self, x, attn_mask=None):
        return self._run_trunk_impl(x, attn_mask=attn_mask)

    def forward_blocks_for_onnx_export(self, input_stem, mask_flat=None):
        if self.varlen and mask_flat is not None:
            N, L = mask_flat.shape
            attn_mask = (mask_flat == 0).view(N, 1, 1, L)
            return self._forward_blocks_impl(input_stem, attn_mask=attn_mask).float()
        return self._forward_blocks_impl(input_stem).float()

    def forward_trunk_for_onnx_export(self, input_spatial, input_global):
        x, attn_mask, mask_flat = self._forward_stem_impl(input_spatial, input_global)
        x = self._forward_blocks_impl(x, attn_mask=attn_mask)
        if self.varlen:
            return x.float(), mask_flat.float()
        return x.float()

    def forward(self, input_spatial, input_global):
        x, attn_mask, mask_flat = self._forward_stem_impl(input_spatial, input_global)
        x = self._forward_blocks_impl(x, attn_mask=attn_mask)

        out_policy = self.policy_head(x, mask=mask_flat)
        (
            out_value, out_misc, out_moremisc,
            out_ownership, out_scoring, out_futurepos, out_seki,
            out_scorebelief,
        ) = self.value_head(x, input_global[:, -1:], mask=mask_flat)

        return (
            out_policy.float(), out_value.float(), out_misc.float(), out_moremisc.float(),
            out_ownership.float(), out_scoring.float(), out_futurepos.float(), out_seki.float(),
            out_scorebelief.float(),
        )


# ---------------------------------------------------------------------------
# Checkpoint format detection and conversion
# ---------------------------------------------------------------------------
def detect_checkpoint_format(state_dict):
    """Detect checkpoint format: 'pt' (model.py), 'te' (full TE), or 'te_decomposed' (hybrid/decomposed TE)."""
    for key in state_dict:
        if ".layer.self_attention." in key:
            return "te"
        if ".ln_qkv." in key:
            return "te_decomposed"
    return "pt"


def convert_checkpoint_model_to_te(state_dict):
    """Convert model.py state_dict to TE (te.TransformerLayer) format."""
    new_sd = {}
    for key, value in state_dict.items():
        if ".ffn_w1.weight" in key:
            block_prefix = key.rsplit(".ffn_w1.weight", 1)[0]
            wgate_key = block_prefix + ".ffn_wgate.weight"
            new_sd[block_prefix + ".layer.layernorm_mlp.fc1_weight"] = torch.cat([value, state_dict[wgate_key]], dim=0)
        elif ".ffn_wgate.weight" in key:
            continue
        elif ".ffn_w2.weight" in key:
            new_sd[key.replace(".ffn_w2.weight", ".layer.layernorm_mlp.fc2_weight")] = value
        elif ".norm1.norm.weight" in key:
            new_sd[key.replace(".norm1.norm.weight", ".layer.self_attention.layernorm_qkv.layer_norm_weight")] = value
        elif ".norm2.norm.weight" in key:
            new_sd[key.replace(".norm2.norm.weight", ".layer.layernorm_mlp.layer_norm_weight")] = value
        elif key == "norm_final.norm.weight":
            new_sd["norm_final.weight"] = value
        elif ".norm_final.norm.weight" in key:
            new_sd[key.replace(".norm_final.norm.weight", ".norm_final.weight")] = value
        elif ".norm1.weight" in key:
            new_sd[key.replace(".norm1.weight", ".layer.self_attention.layernorm_qkv.layer_norm_weight")] = value
        elif ".norm1.bias" in key:
            new_sd[key.replace(".norm1.bias", ".layer.self_attention.layernorm_qkv.layer_norm_bias")] = value
        elif ".norm2.weight" in key:
            new_sd[key.replace(".norm2.weight", ".layer.layernorm_mlp.layer_norm_weight")] = value
        elif ".norm2.bias" in key:
            new_sd[key.replace(".norm2.bias", ".layer.layernorm_mlp.layer_norm_bias")] = value
        elif ".q_proj.weight" in key:
            new_sd[key.replace(".q_proj.weight", ".layer.self_attention.layernorm_qkv.query_weight")] = value
        elif ".k_proj.weight" in key:
            new_sd[key.replace(".k_proj.weight", ".layer.self_attention.layernorm_qkv.key_weight")] = value
        elif ".v_proj.weight" in key:
            new_sd[key.replace(".v_proj.weight", ".layer.self_attention.layernorm_qkv.value_weight")] = value
        elif ".out_proj.weight" in key:
            new_sd[key.replace(".out_proj.weight", ".layer.self_attention.proj.weight")] = value
        else:
            new_sd[key] = value
    return new_sd


def convert_checkpoint_model_to_te_decomposed(state_dict):
    """Convert model.py state_dict to the export-only decomposed TE format."""
    new_sd = {}
    for key, value in state_dict.items():
        if ".ffn_w1.weight" in key:
            block_prefix = key.rsplit(".ffn_w1.weight", 1)[0]
            wgate_key = block_prefix + ".ffn_wgate.weight"
            new_sd[block_prefix + ".ln_mlp.fc1_weight"] = torch.cat([value, state_dict[wgate_key]], dim=0)
        elif ".ffn_wgate.weight" in key:
            continue
        elif ".ffn_w2.weight" in key:
            new_sd[key.replace(".ffn_w2.weight", ".ln_mlp.fc2_weight")] = value
        elif ".norm1.norm.weight" in key:
            new_sd[key.replace(".norm1.norm.weight", ".ln_qkv.layer_norm_weight")] = value
        elif ".norm2.norm.weight" in key:
            new_sd[key.replace(".norm2.norm.weight", ".ln_mlp.layer_norm_weight")] = value
        elif key == "norm_final.norm.weight":
            new_sd["norm_final.weight"] = value
        elif ".norm_final.norm.weight" in key:
            new_sd[key.replace(".norm_final.norm.weight", ".norm_final.weight")] = value
        elif ".norm1.weight" in key:
            new_sd[key.replace(".norm1.weight", ".ln_qkv.layer_norm_weight")] = value
        elif ".norm2.weight" in key:
            new_sd[key.replace(".norm2.weight", ".ln_mlp.layer_norm_weight")] = value
        elif ".q_proj.weight" in key:
            block_prefix = key.rsplit(".q_proj.weight", 1)[0]
            qkv_weight = torch.cat([
                value,
                state_dict[block_prefix + ".k_proj.weight"],
                state_dict[block_prefix + ".v_proj.weight"],
            ], dim=0)
            new_sd[block_prefix + ".ln_qkv.weight"] = qkv_weight
        elif ".k_proj.weight" in key or ".v_proj.weight" in key:
            continue
        elif ".out_proj.weight" in key:
            new_sd[key.replace(".out_proj.weight", ".proj.weight")] = value
        else:
            new_sd[key] = value
    return new_sd


def convert_checkpoint_te_decomposed_to_model(state_dict, zero_centered_norm=False):
    """Convert hybrid/decomposed TE state_dict back to model.py format.

    Handles the fused weight layout used by TransformerBlockTEHybrid and
    TransformerBlockTEDecomposed (ln_qkv, ln_mlp, proj).
    """
    norm_suffix = ".weight" if zero_centered_norm else ".norm.weight"
    new_sd = {}
    for key, value in state_dict.items():
        if "_extra_state" in key:
            continue
        if ".ln_mlp.fc1_weight" in key:
            block_prefix = key.rsplit(".ln_mlp.fc1_weight", 1)[0]
            half = value.shape[0] // 2
            new_sd[block_prefix + ".ffn_w1.weight"] = value[:half]
            new_sd[block_prefix + ".ffn_wgate.weight"] = value[half:]
        elif ".ln_mlp.fc2_weight" in key:
            new_sd[key.replace(".ln_mlp.fc2_weight", ".ffn_w2.weight")] = value
        elif ".ln_qkv.layer_norm_weight" in key:
            new_sd[key.replace(".ln_qkv.layer_norm_weight", ".norm1" + norm_suffix)] = value
        elif ".ln_mlp.layer_norm_weight" in key:
            new_sd[key.replace(".ln_mlp.layer_norm_weight", ".norm2" + norm_suffix)] = value
        elif ".ln_qkv.weight" in key:
            block_prefix = key.rsplit(".ln_qkv.weight", 1)[0]
            c = value.shape[0] // 3
            new_sd[block_prefix + ".q_proj.weight"] = value[:c]
            new_sd[block_prefix + ".k_proj.weight"] = value[c:2*c]
            new_sd[block_prefix + ".v_proj.weight"] = value[2*c:]
        elif ".proj.weight" in key and ".ln_mlp." not in key:
            new_sd[key.replace(".proj.weight", ".out_proj.weight")] = value
        elif key == "norm_final.weight":
            new_sd["norm_final" + norm_suffix] = value
        elif key.endswith(".norm_final.weight"):
            new_sd[key.replace(".norm_final.weight", ".norm_final" + norm_suffix)] = value
        else:
            new_sd[key] = value
    return new_sd


def convert_checkpoint_te_to_model(state_dict, zero_centered_norm=False):
    """Convert TE (te.TransformerLayer) state_dict back to model.py format.

    Filters out TE-specific _extra_state keys.
    When zero_centered_norm=True, maps norm weights to ZeroCenteredRMSNormFP32 paths
    (e.g. .norm1.weight) instead of RMSNormFP32 paths (e.g. .norm1.norm.weight).
    """
    # ZeroCenteredRMSNormFP32 has .weight directly; RMSNormFP32 wraps nn.RMSNorm as .norm.weight
    norm_suffix = ".weight" if zero_centered_norm else ".norm.weight"
    new_sd = {}
    for key, value in state_dict.items():
        if "_extra_state" in key:
            continue
        if ".layer.layernorm_mlp.fc1_weight" in key:
            block_prefix = key.rsplit(".layer.layernorm_mlp.fc1_weight", 1)[0]
            half = value.shape[0] // 2
            new_sd[block_prefix + ".ffn_w1.weight"] = value[:half]
            new_sd[block_prefix + ".ffn_wgate.weight"] = value[half:]
        elif ".layer.layernorm_mlp.fc2_weight" in key:
            new_sd[key.replace(".layer.layernorm_mlp.fc2_weight", ".ffn_w2.weight")] = value
        elif ".layer.self_attention.layernorm_qkv.layer_norm_weight" in key:
            new_sd[key.replace(".layer.self_attention.layernorm_qkv.layer_norm_weight", ".norm1" + norm_suffix)] = value
        elif ".layer.self_attention.layernorm_qkv.layer_norm_bias" in key:
            # model.py uses nn.RMSNorm inside RMSNormFP32 and therefore has no bias parameter.
            continue
        elif ".layer.layernorm_mlp.layer_norm_weight" in key:
            new_sd[key.replace(".layer.layernorm_mlp.layer_norm_weight", ".norm2" + norm_suffix)] = value
        elif ".layer.layernorm_mlp.layer_norm_bias" in key:
            continue
        elif ".layer.self_attention.layernorm_qkv.query_weight" in key:
            new_sd[key.replace(".layer.self_attention.layernorm_qkv.query_weight", ".q_proj.weight")] = value
        elif ".layer.self_attention.layernorm_qkv.key_weight" in key:
            new_sd[key.replace(".layer.self_attention.layernorm_qkv.key_weight", ".k_proj.weight")] = value
        elif ".layer.self_attention.layernorm_qkv.value_weight" in key:
            new_sd[key.replace(".layer.self_attention.layernorm_qkv.value_weight", ".v_proj.weight")] = value
        elif ".layer.self_attention.proj.weight" in key:
            new_sd[key.replace(".layer.self_attention.proj.weight", ".out_proj.weight")] = value
        elif key == "norm_final.weight":
            new_sd["norm_final" + norm_suffix] = value
        elif key.endswith(".norm_final.weight"):
            new_sd[key.replace(".norm_final.weight", ".norm_final" + norm_suffix)] = value
        else:
            new_sd[key] = value
    return new_sd
