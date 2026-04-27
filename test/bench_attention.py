#!/usr/bin/env python3
"""Micro-benchmark for transformer stacks with different attention backends.

Primary goal: measure the *relative* cost of adding a varlen mask within
each stack implementation (no_mask → mask delta), NOT cross-stack absolute
comparisons (which also reflect norm/FFN differences).

Stacks tested:
  - TE TransformerLayer (fused block, TE attention, arbitrary mask)
  - TE Decomposed (TE norm/FFN, TE DotProductAttention, arbitrary mask)
  - TE TransformerLayer + post_scale_bias (additive bias instead of arbitrary mask)
  - TE Decomposed + post_scale_bias (additive bias instead of arbitrary mask)
  - Pure SDPA (RMSNormFP32/nn.Linear, F.scaled_dot_product_attention)
  - Hybrid (TE norm/FFN, SDPA attention)

Usage:
    cd nano && python test/bench_attention.py
    python test/bench_attention.py --layers 32 --batch-size 512
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te

from model import apply_rotary_emb, precompute_freqs_cos_sin_2d, RMSNormFP32


def build_varlen_masks(batch_size, pos_len, device):
    """Build varlen masks simulating mixed board sizes.

    Returns:
        te_mask:   (N, 1, 1, L) bool, True = masked  (for TE attention)
        sdpa_mask: (N, 1, 1, L) bfloat16, 0 = valid, -inf = masked  (for SDPA)
    """
    L = pos_len * pos_len
    board_sizes = [9, 13, 14, 15, 16, 17, 18, 19]
    mask_2d = torch.zeros(batch_size, pos_len, pos_len, device=device)
    for i in range(batch_size):
        bs = board_sizes[i % len(board_sizes)]
        mask_2d[i, :bs, :bs] = 1.0
    mask_flat = mask_2d.view(batch_size, L)
    # TE bool mask: True = masked
    te_mask = (mask_flat == 0).view(batch_size, 1, 1, L)
    # SDPA additive mask: 0 = valid, -inf = masked  (precomputed, no per-iter alloc)
    sdpa_mask = torch.zeros(batch_size, 1, 1, L, device=device, dtype=torch.bfloat16)
    sdpa_mask.masked_fill_(te_mask, float('-inf'))
    return te_mask, sdpa_mask


# ---------------------------------------------------------------------------
# Stack implementations
# ---------------------------------------------------------------------------

class TETransformerStack(nn.Module):
    """Stack of te.TransformerLayer (complete fused blocks)."""
    def __init__(self, hidden, heads, ffn_dim, num_layers, pos_len):
        super().__init__()
        self.blocks = nn.ModuleList([
            te.TransformerLayer(
                hidden, ffn_dim, heads,
                layernorm_epsilon=1e-6,
                hidden_dropout=0, attention_dropout=0,
                self_attn_mask_type="no_mask",
                normalization="RMSNorm",
                bias=False, activation="swiglu",
                attn_input_format="bshd",
            )
            for _ in range(num_layers)
        ])
        head_dim = hidden // heads
        emb = precompute_freqs_cos_sin_2d(head_dim, pos_len)
        emb_full = torch.cat([emb, emb], dim=-1)
        self.register_buffer("rope", emb_full, persistent=False)

    def forward(self, x, attn_mask=None):
        for block in self.blocks:
            if attn_mask is not None:
                x = block(x, rotary_pos_emb=self.rope,
                          attention_mask=attn_mask,
                          self_attn_mask_type="arbitrary")
            else:
                x = block(x, rotary_pos_emb=self.rope)
        return x


class TEDecomposedStack(nn.Module):
    """Stack of TE decomposed blocks: TE fused norm/FFN + te.DotProductAttention."""
    def __init__(self, hidden, heads, ffn_dim, num_layers, pos_len):
        super().__init__()
        self.num_heads = heads
        self.head_dim = hidden // heads
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(nn.ModuleDict({
                'ln_qkv': te.LayerNormLinear(hidden, 3 * hidden, eps=1e-6, bias=False, normalization="RMSNorm"),
                'attn': te.DotProductAttention(num_attention_heads=heads, kv_channels=self.head_dim,
                                                attention_dropout=0.0, attn_mask_type="no_mask", qkv_format="bshd"),
                'proj': te.Linear(hidden, hidden, bias=False),
                'ln_mlp': te.LayerNormMLP(hidden, ffn_dim, eps=1e-6, bias=False,
                                           normalization="RMSNorm", activation="swiglu"),
            }))
        head_dim = hidden // heads
        emb = precompute_freqs_cos_sin_2d(head_dim, pos_len)
        emb_full = torch.cat([emb, emb], dim=-1)
        self.register_buffer("rope_cos", emb_full.cos(), persistent=False)
        self.register_buffer("rope_sin", emb_full.sin(), persistent=False)

    def forward(self, x, attn_mask=None):
        B, L, _ = x.shape
        for block in self.blocks:
            residual = x
            qkv = block['ln_qkv'](x).view(B, L, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q, k = apply_rotary_emb(q, k, self.rope_cos, self.rope_sin)
            attn_kwargs = {}
            if attn_mask is not None:
                attn_kwargs["attention_mask"] = attn_mask
                attn_kwargs["attn_mask_type"] = "arbitrary"
            x = residual + block['proj'](block['attn'](q, k, v, **attn_kwargs))
            x = x + block['ln_mlp'](x)
        return x


class TETransformerBiasStack(nn.Module):
    """Stack of te.TransformerLayer using core_attention_bias (additive, post-scale)."""
    def __init__(self, hidden, heads, ffn_dim, num_layers, pos_len):
        super().__init__()
        self.blocks = nn.ModuleList([
            te.TransformerLayer(
                hidden, ffn_dim, heads,
                layernorm_epsilon=1e-6,
                hidden_dropout=0, attention_dropout=0,
                self_attn_mask_type="no_mask",
                normalization="RMSNorm",
                bias=False, activation="swiglu",
                attn_input_format="bshd",
            )
            for _ in range(num_layers)
        ])
        head_dim = hidden // heads
        emb = precompute_freqs_cos_sin_2d(head_dim, pos_len)
        emb_full = torch.cat([emb, emb], dim=-1)
        self.register_buffer("rope", emb_full, persistent=False)

    def forward(self, x, attn_mask=None):
        for block in self.blocks:
            if attn_mask is not None:
                x = block(x, rotary_pos_emb=self.rope,
                          core_attention_bias_type="post_scale_bias",
                          core_attention_bias=attn_mask)
            else:
                x = block(x, rotary_pos_emb=self.rope)
        return x


class TEDecomposedBiasStack(nn.Module):
    """Stack of TE decomposed blocks using core_attention_bias (additive, post-scale)."""
    def __init__(self, hidden, heads, ffn_dim, num_layers, pos_len):
        super().__init__()
        self.num_heads = heads
        self.head_dim = hidden // heads
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(nn.ModuleDict({
                'ln_qkv': te.LayerNormLinear(hidden, 3 * hidden, eps=1e-6, bias=False, normalization="RMSNorm"),
                'attn': te.DotProductAttention(num_attention_heads=heads, kv_channels=self.head_dim,
                                                attention_dropout=0.0, attn_mask_type="no_mask", qkv_format="bshd"),
                'proj': te.Linear(hidden, hidden, bias=False),
                'ln_mlp': te.LayerNormMLP(hidden, ffn_dim, eps=1e-6, bias=False,
                                           normalization="RMSNorm", activation="swiglu"),
            }))
        head_dim = hidden // heads
        emb = precompute_freqs_cos_sin_2d(head_dim, pos_len)
        emb_full = torch.cat([emb, emb], dim=-1)
        self.register_buffer("rope_cos", emb_full.cos(), persistent=False)
        self.register_buffer("rope_sin", emb_full.sin(), persistent=False)

    def forward(self, x, attn_mask=None):
        B, L, _ = x.shape
        for block in self.blocks:
            residual = x
            qkv = block['ln_qkv'](x).view(B, L, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q, k = apply_rotary_emb(q, k, self.rope_cos, self.rope_sin)
            attn_kwargs = {}
            if attn_mask is not None:
                attn_kwargs["core_attention_bias_type"] = "post_scale_bias"
                attn_kwargs["core_attention_bias"] = attn_mask
            x = residual + block['proj'](block['attn'](q, k, v, **attn_kwargs))
            x = x + block['ln_mlp'](x)
        return x


class SDPAStack(nn.Module):
    """Stack of pure PyTorch blocks using F.scaled_dot_product_attention."""
    def __init__(self, hidden, heads, ffn_dim, num_layers, pos_len):
        super().__init__()
        self.num_heads = heads
        self.head_dim = hidden // heads
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(nn.ModuleDict({
                'norm1': RMSNormFP32(hidden, eps=1e-6),
                'q_proj': nn.Linear(hidden, hidden, bias=False),
                'k_proj': nn.Linear(hidden, hidden, bias=False),
                'v_proj': nn.Linear(hidden, hidden, bias=False),
                'out_proj': nn.Linear(hidden, hidden, bias=False),
                'norm2': RMSNormFP32(hidden, eps=1e-6),
                'ffn_w1': nn.Linear(hidden, ffn_dim, bias=False),
                'ffn_wgate': nn.Linear(hidden, ffn_dim, bias=False),
                'ffn_w2': nn.Linear(ffn_dim, hidden, bias=False),
            }))
        head_dim = hidden // heads
        emb = precompute_freqs_cos_sin_2d(head_dim, pos_len)
        emb_full = torch.cat([emb, emb], dim=-1)
        self.register_buffer("rope_cos", emb_full.cos(), persistent=False)
        self.register_buffer("rope_sin", emb_full.sin(), persistent=False)

    def forward(self, x, attn_mask=None):
        B, L, C = x.shape
        for block in self.blocks:
            x_normed = block['norm1'](x)
            q = block['q_proj'](x_normed).view(B, L, self.num_heads, self.head_dim)
            k = block['k_proj'](x_normed).view(B, L, self.num_heads, self.head_dim)
            v = block['v_proj'](x_normed).view(B, L, self.num_heads, self.head_dim)
            q, k = apply_rotary_emb(q, k, self.rope_cos, self.rope_sin)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, L, C)
            x = x + block['out_proj'](attn_out)
            x_normed = block['norm2'](x)
            w1_out = F.silu(block['ffn_w1'](x_normed))
            wgate_out = block['ffn_wgate'](x_normed)
            x = x + block['ffn_w2'](w1_out * wgate_out)
        return x


class TEHybridStack(nn.Module):
    """TE fused norm/FFN + SDPA attention.

    Accepts SDPA additive mask (precomputed outside), no per-iter mask conversion.
    """
    def __init__(self, hidden, heads, ffn_dim, num_layers, pos_len):
        super().__init__()
        self.num_heads = heads
        self.head_dim = hidden // heads
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(nn.ModuleDict({
                'ln_qkv': te.LayerNormLinear(hidden, 3 * hidden, eps=1e-6, bias=False, normalization="RMSNorm"),
                'proj': te.Linear(hidden, hidden, bias=False),
                'ln_mlp': te.LayerNormMLP(hidden, ffn_dim, eps=1e-6, bias=False,
                                           normalization="RMSNorm", activation="swiglu"),
            }))
        head_dim = hidden // heads
        emb = precompute_freqs_cos_sin_2d(head_dim, pos_len)
        emb_full = torch.cat([emb, emb], dim=-1)
        self.register_buffer("rope_cos", emb_full.cos(), persistent=False)
        self.register_buffer("rope_sin", emb_full.sin(), persistent=False)

    def forward(self, x, attn_mask=None):
        """attn_mask: (N, 1, 1, L) additive mask for SDPA (precomputed, 0/-inf)."""
        B, L, C = x.shape
        for block in self.blocks:
            residual = x
            qkv = block['ln_qkv'](x).view(B, L, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q, k = apply_rotary_emb(q, k, self.rope_cos, self.rope_sin)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, L, C)
            x = residual + block['proj'](attn_out)
            x = x + block['ln_mlp'](x)
        return x


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def bench(name, model, x, mask, warmup=10, iters=50):
    """Run benchmark and return ms/iter."""
    torch.cuda.synchronize()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(warmup):
            _ = model(x, attn_mask=mask)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(iters):
            _ = model(x, attn_mask=mask)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms = elapsed / iters * 1000
    print(f"  {name:55s}  {ms:8.1f} ms/iter")
    return ms


def main():
    parser = argparse.ArgumentParser(description="Transformer stack micro-benchmark")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=1536)
    parser.add_argument("--pos-len", type=int, default=19)
    parser.add_argument("--layers", type=int, default=32,
                        help="Number of transformer layers (default: 32)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    B, H, D, FFN = args.batch_size, args.heads, args.hidden, args.ffn_dim
    PL = args.pos_len
    L = PL * PL
    NL = args.layers

    print("=" * 76)
    print("Transformer Stack Micro-benchmark")
    print(f"  batch={B}, hidden={D}, heads={H}, ffn={FFN}, seq_len={L}, layers={NL}")
    print(f"  GPU: {torch.cuda.get_device_name(device)}")
    print()
    print("NOTE: Cross-stack absolute times reflect norm/FFN implementation")
    print("differences (TE fused vs nn.*). The meaningful comparison is the")
    print("no_mask → mask delta WITHIN each stack.")
    print("=" * 76)

    # Precompute all inputs and masks on GPU
    x = torch.randn(B, L, D, device=device, dtype=torch.bfloat16)
    te_mask, sdpa_mask = build_varlen_masks(B, PL, device)

    # Collect results as (name, no_mask_ms, mask_ms) tuples for summary
    groups = []

    # --- 1. TE TransformerLayer ---
    print("\n[1] TE TransformerLayer (eager)")
    model = TETransformerStack(D, H, FFN, NL, PL).to(device).eval()
    t_no = bench("no_mask", model, x, None, args.warmup, args.iters)
    t_mk = bench("arbitrary mask", model, x, te_mask, args.warmup, args.iters)
    groups.append(("TE TransformerLayer", t_no, t_mk))

    # TE + torch.compile: TE's PyCapsule ops typically cause graph breaks,
    # so dynamo wraps around TE eager calls. This tests whether the compile
    # wrapper adds or removes overhead—NOT whether TE kernels get compiled.
    print("\n[2] TE TransformerLayer (wrapped by torch.compile, expect graph breaks)")
    compiled = torch.compile(model, mode="default")
    t_no = bench("no_mask (compile-wrapped)", compiled, x, None, args.warmup, args.iters)
    t_mk = bench("arbitrary mask (compile-wrapped)", compiled, x, te_mask, args.warmup, args.iters)
    groups.append(("TE TransformerLayer (compile-wrapped)", t_no, t_mk))
    del model, compiled

    # --- 2. TE Decomposed ---
    print("\n[3] TE Decomposed: te.DotProductAttention (eager)")
    model = TEDecomposedStack(D, H, FFN, NL, PL).to(device).eval()
    t_no = bench("no_mask", model, x, None, args.warmup, args.iters)
    t_mk = bench("arbitrary mask", model, x, te_mask, args.warmup, args.iters)
    groups.append(("TE Decomposed", t_no, t_mk))

    print("\n[4] TE Decomposed (wrapped by torch.compile, expect graph breaks)")
    compiled = torch.compile(model, mode="default")
    t_no = bench("no_mask (compile-wrapped)", compiled, x, None, args.warmup, args.iters)
    t_mk = bench("arbitrary mask (compile-wrapped)", compiled, x, te_mask, args.warmup, args.iters)
    groups.append(("TE Decomposed (compile-wrapped)", t_no, t_mk))
    del model, compiled

    # --- 3. TE TransformerLayer + post_scale_bias ---
    print("\n[5] TE TransformerLayer + post_scale_bias (eager)")
    model = TETransformerBiasStack(D, H, FFN, NL, PL).to(device).eval()
    t_no = bench("no_mask", model, x, None, args.warmup, args.iters)
    t_mk = bench("post_scale_bias", model, x, sdpa_mask, args.warmup, args.iters)
    groups.append(("TE TransformerLayer + bias", t_no, t_mk))

    print("\n[6] TE TransformerLayer + post_scale_bias (compile-wrapped)")
    compiled = torch.compile(model, mode="default")
    t_no = bench("no_mask (compile-wrapped)", compiled, x, None, args.warmup, args.iters)
    t_mk = bench("post_scale_bias (compile-wrapped)", compiled, x, sdpa_mask, args.warmup, args.iters)
    groups.append(("TE TransformerLayer + bias (compile)", t_no, t_mk))
    del model, compiled

    # --- 4. TE Decomposed + post_scale_bias ---
    print("\n[7] TE Decomposed + post_scale_bias (eager)")
    model = TEDecomposedBiasStack(D, H, FFN, NL, PL).to(device).eval()
    t_no = bench("no_mask", model, x, None, args.warmup, args.iters)
    t_mk = bench("post_scale_bias", model, x, sdpa_mask, args.warmup, args.iters)
    groups.append(("TE Decomposed + bias", t_no, t_mk))

    print("\n[8] TE Decomposed + post_scale_bias (compile-wrapped)")
    compiled = torch.compile(model, mode="default")
    t_no = bench("no_mask (compile-wrapped)", compiled, x, None, args.warmup, args.iters)
    t_mk = bench("post_scale_bias (compile-wrapped)", compiled, x, sdpa_mask, args.warmup, args.iters)
    groups.append(("TE Decomposed + bias (compile)", t_no, t_mk))
    del model, compiled

    # --- 5. Pure SDPA ---
    print("\n[9] Pure PyTorch SDPA (eager)")
    model = SDPAStack(D, H, FFN, NL, PL).to(device).eval()
    t_no = bench("no_mask", model, x, None, args.warmup, args.iters)
    t_mk = bench("additive mask", model, x, sdpa_mask, args.warmup, args.iters)
    groups.append(("Pure SDPA", t_no, t_mk))

    print("\n[10] Pure PyTorch SDPA + torch.compile")
    compiled = torch.compile(model, mode="default")
    t_no = bench("no_mask + compile", compiled, x, None, args.warmup, args.iters)
    t_mk = bench("additive mask + compile", compiled, x, sdpa_mask, args.warmup, args.iters)
    groups.append(("Pure SDPA + compile", t_no, t_mk))
    del model, compiled

    # --- 6. Hybrid: TE norm/FFN + SDPA attention ---
    # Uses precomputed SDPA additive mask — no per-iter mask conversion.
    print("\n[11] Hybrid: TE norm/FFN + SDPA attention (eager)")
    model = TEHybridStack(D, H, FFN, NL, PL).to(device).eval()
    t_no = bench("no_mask", model, x, None, args.warmup, args.iters)
    t_mk = bench("SDPA additive mask (precomputed)", model, x, sdpa_mask, args.warmup, args.iters)
    groups.append(("Hybrid TE+SDPA", t_no, t_mk))

    print("\n[12] Hybrid: TE norm/FFN + SDPA attention (wrapped by torch.compile)")
    compiled = torch.compile(model, mode="default")
    t_no = bench("no_mask (compile-wrapped)", compiled, x, None, args.warmup, args.iters)
    t_mk = bench("SDPA mask (compile-wrapped)", compiled, x, sdpa_mask, args.warmup, args.iters)
    groups.append(("Hybrid TE+SDPA (compile-wrapped)", t_no, t_mk))
    del model, compiled

    # --- Summary ---
    print("\n" + "=" * 76)
    print("Summary: no_mask → mask delta within each stack")
    print("=" * 76)
    print(f"  {'Stack':<42s}  {'no_mask':>8s}  {'mask':>8s}  {'delta':>8s}  {'ratio':>6s}")
    print(f"  {'-'*42}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")
    for name, t_no, t_mk in groups:
        delta = t_mk - t_no
        ratio = t_mk / t_no if t_no > 0 else float('inf')
        print(f"  {name:<42s}  {t_no:7.1f}   {t_mk:7.1f}   {delta:+7.1f}   {ratio:5.2f}x")
    print("=" * 76)


if __name__ == "__main__":
    main()
