"""Pure JAX KataGo nano transformer model.

The parameter shapes intentionally mirror the PyTorch module where convenient:
linear weights are stored as (out, in), and conv weights as (out, in, kh, kw).
"""

import math

import jax
import jax.numpy as jnp

import configs


EXTRA_SCORE_DISTR_RADIUS = 60
COMPUTE_DTYPE = jnp.bfloat16


def dtype_from_name(name):
    normalized = str(name).lower()
    if normalized in ("float32", "fp32"):
        return jnp.float32
    if normalized in ("bfloat16", "bf16"):
        return jnp.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _split(key, n):
    return jax.random.split(key, n)


def _normal(key, shape, std):
    return jax.random.normal(key, shape, dtype=jnp.float32) * jnp.asarray(std, dtype=jnp.float32)


def init_linear(key, in_dim, out_dim, std, bias=True):
    params = {"w": _normal(key, (out_dim, in_dim), std)}
    if bias:
        params["b"] = jnp.zeros((out_dim,), dtype=jnp.float32)
    return params


def init_conv(key, in_ch, out_ch, kernel, std):
    return {"w": _normal(key, (out_ch, in_ch, kernel, kernel), std)}


def stack_tree_sequence(sequence):
    first = sequence[0]
    if isinstance(first, dict):
        return {k: stack_tree_sequence([item[k] for item in sequence]) for k in first}
    if isinstance(first, tuple):
        return tuple(stack_tree_sequence([item[i] for item in sequence]) for i in range(len(first)))
    if isinstance(first, list):
        return [stack_tree_sequence([item[i] for item in sequence]) for i in range(len(first))]
    return jnp.stack(sequence, axis=0)


def tree_index(tree, index):
    if isinstance(tree, dict):
        return {k: tree_index(v, index) for k, v in tree.items()}
    if isinstance(tree, tuple):
        return tuple(tree_index(v, index) for v in tree)
    if isinstance(tree, list):
        return [tree_index(v, index) for v in tree]
    return tree[index]


def linear(params, x, out_dtype=jnp.float32):
    y = jnp.matmul(
        x.astype(COMPUTE_DTYPE),
        jnp.swapaxes(params["w"], -1, -2).astype(COMPUTE_DTYPE),
    )
    if "b" in params:
        y = y.astype(jnp.float32) + params["b"]
    return y.astype(out_dtype)


def conv2d_nchw(params, x, out_dtype=jnp.float32):
    return jax.lax.conv_general_dilated(
        x.astype(COMPUTE_DTYPE),
        params["w"].astype(COMPUTE_DTYPE),
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    ).astype(out_dtype)


def rms_norm(params, x, eps=1e-6):
    xf = x.astype(jnp.float32)
    inv_rms = jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (xf * inv_rms * params["weight"]).astype(x.dtype)


def silu(x):
    return x * jax.nn.sigmoid(x)


def precompute_rope(head_dim, pos_len, theta=100.0):
    if head_dim % 4 != 0:
        raise ValueError(f"head_dim must be divisible by 4 for fixed 2D RoPE, got {head_dim}")
    dim_half = head_dim // 2
    freqs = 1.0 / (theta ** (jnp.arange(0, dim_half, 2, dtype=jnp.float32) / dim_half))
    t = jnp.arange(pos_len, dtype=jnp.float32)
    grid_h, grid_w = jnp.meshgrid(t, t, indexing="ij")
    emb_h = grid_h[..., None] * freqs
    emb_w = grid_w[..., None] * freqs
    emb = jnp.concatenate([emb_h, emb_w], axis=-1).reshape(pos_len * pos_len, 1, 1, dim_half)
    emb = jnp.concatenate([emb, emb], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def rotate_half(x):
    a, b = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-b, a], axis=-1)


def apply_rope(q, k, cos, sin):
    cos = cos.reshape((1, q.shape[1], -1, q.shape[-1]))
    sin = sin.reshape((1, q.shape[1], -1, q.shape[-1]))
    qf = q.astype(jnp.float32)
    kf = k.astype(jnp.float32)
    return (
        (qf * cos + rotate_half(qf) * sin).astype(q.dtype),
        (kf * cos + rotate_half(kf) * sin).astype(k.dtype),
    )


def attention(q, k, v, mask=None, attention_impl="manual", out_dtype=jnp.float32):
    # q,k,v: B,L,H,D
    if attention_impl == "xla":
        if mask is not None:
            raise ValueError("attention_impl='xla' does not support the manual mask layout yet")
        return jax.nn.dot_product_attention(
            q.astype(COMPUTE_DTYPE),
            k.astype(COMPUTE_DTYPE),
            v.astype(COMPUTE_DTYPE),
            scale=1.0 / math.sqrt(q.shape[-1]),
            implementation="xla",
        ).astype(out_dtype)

    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))
    scale = 1.0 / math.sqrt(q.shape[-1])
    logits = jnp.einsum(
        "bhld,bhmd->bhlm",
        q.astype(COMPUTE_DTYPE),
        k.astype(COMPUTE_DTYPE),
    ).astype(jnp.float32) * scale
    if mask is not None:
        logits = logits + mask
    weights = jax.nn.softmax(logits, axis=-1).astype(COMPUTE_DTYPE)
    out = jnp.einsum("bhlm,bhmd->bhld", weights, v.astype(COMPUTE_DTYPE)).astype(out_dtype)
    return jnp.transpose(out, (0, 2, 1, 3))


def init_params(
    key,
    config,
    pos_len,
    init_std=0.02,
    score_mode="simple",
    fuse_projections=False,
    stack_blocks=False,
):
    if score_mode != "simple":
        raise ValueError("The first JAX TPU path currently supports score_mode='simple' only")

    c = config["hidden_size"]
    num_heads = config["num_heads"]
    head_dim = c // num_heads
    ffn_dim = config["ffn_dim"]
    num_layers = config["num_layers"]
    num_bin = configs.get_num_bin_input_features(config)
    num_global = configs.get_num_global_input_features(config)
    scorebelief_len = (pos_len * pos_len + EXTRA_SCORE_DISTR_RADIUS) * 2

    block_keys = 4 if fuse_projections else 7
    keys = iter(_split(key, 6 + num_layers * block_keys))
    params = {
        "conv_spatial": init_conv(next(keys), num_bin, c, 3, init_std),
        "linear_global": init_linear(next(keys), num_global, c, init_std, bias=False),
        "blocks": [],
    }
    out_std = init_std / math.sqrt(2.0 * num_layers)
    for _ in range(num_layers):
        block = {
            "norm1": {"weight": jnp.ones((c,), dtype=jnp.float32)},
            "norm2": {"weight": jnp.ones((c,), dtype=jnp.float32)},
            "out_proj": init_linear(next(keys), c, c, out_std, bias=False),
            "ffn_w2": init_linear(next(keys), ffn_dim, c, out_std, bias=False),
        }
        if fuse_projections:
            block.update({
                "qkv_proj": init_linear(next(keys), c, 3 * c, init_std, bias=False),
                "ffn_upgate": init_linear(next(keys), c, 2 * ffn_dim, init_std, bias=False),
            })
        else:
            block.update({
                "q_proj": init_linear(next(keys), c, c, init_std, bias=False),
                "k_proj": init_linear(next(keys), c, c, init_std, bias=False),
                "v_proj": init_linear(next(keys), c, c, init_std, bias=False),
                "ffn_w1": init_linear(next(keys), c, ffn_dim, init_std, bias=False),
                "ffn_wgate": init_linear(next(keys), c, ffn_dim, init_std, bias=False),
            })
        params["blocks"].append(block)
    if stack_blocks:
        params["blocks"] = stack_tree_sequence(params["blocks"])
    params["norm_final"] = {"weight": jnp.ones((c,), dtype=jnp.float32)}
    params["policy_head"] = {
        "linear_board": init_linear(next(keys), c, 6, init_std, bias=True),
        "linear_pass": init_linear(next(keys), c, 6, init_std, bias=True),
    }
    params["value_head"] = {
        "linear_sv": init_linear(next(keys), c, 29, init_std, bias=True),
        "linear_s_simple": init_linear(next(keys), c, scorebelief_len, init_std, bias=True),
    }
    return params


def transformer_block(
    params,
    x,
    rope_cos,
    rope_sin,
    num_heads,
    attention_impl="manual",
    activation_dtype=jnp.float32,
):
    bsz, seq_len, channels = x.shape
    head_dim = channels // num_heads

    with jax.named_scope("attn_norm"):
        x_norm = rms_norm(params["norm1"], x)
    with jax.named_scope("attn_qkv"):
        if "qkv_proj" in params:
            qkv = linear(params["qkv_proj"], x_norm, out_dtype=activation_dtype)
            q, k, v = jnp.split(qkv, 3, axis=-1)
        else:
            q = linear(params["q_proj"], x_norm, out_dtype=activation_dtype)
            k = linear(params["k_proj"], x_norm, out_dtype=activation_dtype)
            v = linear(params["v_proj"], x_norm, out_dtype=activation_dtype)
        q = q.reshape(bsz, seq_len, num_heads, head_dim)
        k = k.reshape(bsz, seq_len, num_heads, head_dim)
        v = v.reshape(bsz, seq_len, num_heads, head_dim)
    with jax.named_scope("attn_rope"):
        q, k = apply_rope(q, k, rope_cos, rope_sin)
    with jax.named_scope("attention"):
        attn_out = attention(
            q,
            k,
            v,
            attention_impl=attention_impl,
            out_dtype=activation_dtype,
        ).reshape(bsz, seq_len, channels)
    with jax.named_scope("attn_out_proj"):
        x = (x + linear(params["out_proj"], attn_out, out_dtype=activation_dtype)).astype(activation_dtype)

    with jax.named_scope("ffn_norm"):
        x_norm = rms_norm(params["norm2"], x)
    with jax.named_scope("ffn_upgate"):
        if "ffn_upgate" in params:
            upgate = linear(params["ffn_upgate"], x_norm, out_dtype=activation_dtype)
            w1, wg = jnp.split(upgate, 2, axis=-1)
            w1 = silu(w1)
        else:
            w1 = silu(linear(params["ffn_w1"], x_norm, out_dtype=activation_dtype))
            wg = linear(params["ffn_wgate"], x_norm, out_dtype=activation_dtype)
        hidden = (w1.astype(jnp.float32) * wg.astype(jnp.float32)).astype(activation_dtype)
    with jax.named_scope("ffn_down"):
        return (x + linear(params["ffn_w2"], hidden, out_dtype=activation_dtype)).astype(activation_dtype)


def forward_stem(
    params,
    binary_input,
    global_input,
    config,
    pos_len,
    activation_dtype=jnp.float32,
):
    c = config["hidden_size"]
    n = binary_input.shape[0]
    seq_len = pos_len * pos_len

    with jax.named_scope("stem"):
        x_spatial = conv2d_nchw(params["conv_spatial"], binary_input, out_dtype=activation_dtype)
        x_global = linear(params["linear_global"], global_input, out_dtype=activation_dtype)
        x = (x_spatial + x_global[:, :, None, None]).astype(activation_dtype)
        return jnp.transpose(x.reshape(n, c, seq_len), (0, 2, 1))


def forward_trunk(
    params,
    x,
    config,
    rope_cache,
    attention_impl="manual",
    activation_dtype=jnp.float32,
    remat_blocks=False,
    scan_blocks=False,
):
    num_heads = config["num_heads"]

    rope_cos, rope_sin = rope_cache

    def apply_block(block, x_in):
        return transformer_block(
            block,
            x_in,
            rope_cos,
            rope_sin,
            num_heads,
            attention_impl=attention_impl,
            activation_dtype=activation_dtype,
        )

    if remat_blocks:
        apply_block = jax.checkpoint(apply_block)

    with jax.named_scope("trunk"):
        if isinstance(params["blocks"], dict) and scan_blocks:
            def scan_body(x_carry, block):
                return apply_block(block, x_carry), None

            x, _ = jax.lax.scan(scan_body, x, params["blocks"])
        elif isinstance(params["blocks"], dict):
            num_layers = params["blocks"]["norm1"]["weight"].shape[0]
            for i in range(num_layers):
                x = apply_block(tree_index(params["blocks"], i), x)
        else:
            for block in params["blocks"]:
                x = apply_block(block, x)
        return rms_norm(params["norm_final"], x).astype(jnp.float32)


def forward_heads(params, x, pos_len):
    n = x.shape[0]
    board = jnp.transpose(linear(params["policy_head"]["linear_board"], x), (0, 2, 1))
    pooled = jnp.mean(x, axis=1)
    pass_logits = linear(params["policy_head"]["linear_pass"], pooled)
    out_policy = jnp.concatenate([board, pass_logits[:, :, None]], axis=2)

    spatial_global = linear(params["value_head"]["linear_sv"], x)
    spatial = spatial_global[:, :, :8]
    global_feats = jnp.mean(spatial_global[:, :, 8:], axis=1)
    spatial = jnp.transpose(spatial, (0, 2, 1)).reshape(n, 8, pos_len, pos_len)
    out_ownership = spatial[:, 0:1]
    out_scoring = spatial[:, 1:2]
    out_futurepos = spatial[:, 2:4]
    out_seki = spatial[:, 4:8]
    out_value = global_feats[:, 0:3]
    out_misc = global_feats[:, 3:13]
    out_moremisc = global_feats[:, 13:21]
    out_scorebelief = jax.nn.log_softmax(linear(params["value_head"]["linear_s_simple"], pooled), axis=1)

    return (
        out_policy, out_value, out_misc, out_moremisc,
        out_ownership, out_scoring, out_futurepos, out_seki,
        out_scorebelief,
    )


def forward(
    params,
    binary_input,
    global_input,
    config,
    pos_len,
    rope_cache,
    attention_impl="manual",
    activation_dtype=jnp.float32,
    remat_blocks=False,
    scan_blocks=False,
):
    x = forward_stem(params, binary_input, global_input, config, pos_len, activation_dtype)
    x = forward_trunk(
        params,
        x,
        config,
        rope_cache,
        attention_impl=attention_impl,
        activation_dtype=activation_dtype,
        remat_blocks=remat_blocks,
        scan_blocks=scan_blocks,
    )
    with jax.named_scope("heads"):
        return forward_heads(params, x, pos_len)


def score_belief_offsets(pos_len):
    mid = pos_len * pos_len + EXTRA_SCORE_DISTR_RADIUS
    return jnp.asarray([float(i - mid) + 0.5 for i in range(mid * 2)], dtype=jnp.float32)
