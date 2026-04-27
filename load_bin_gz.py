"""Load KataGo CNN model weights from bin.gz format.

Reverses the write logic in KataGo's export_model_pytorch.py.
Supports v12+ fixup NBT models with relu or mish activation.
"""

import gzip
import io
import struct

import numpy as np
import torch
import torch.nn as nn

from model_cnn import (
    CNNModel, CNNPolicyHead, CNNValueHead,
    NormBias, NormActConv, KataConvAndGPool,
    ResBlock, NestedBottleneckResBlock,
)


class BinGzReader:
    """Sequential reader for KataGo bin.gz files."""

    def __init__(self, filepath):
        with gzip.open(filepath, "rb") as f:
            self.buf = io.BytesIO(f.read())

    def readline(self):
        """Read one line of ascii text, stripping the newline."""
        line = b""
        while True:
            c = self.buf.read(1)
            if c == b"\n" or c == b"":
                break
            line += c
        return line.decode("ascii").strip()

    def read_int(self):
        return int(self.readline())

    def read_float(self):
        return float(self.readline())

    def read_weights(self, n):
        """Read @BIN@ marker followed by n float32 values."""
        marker = self.buf.read(5)
        assert marker == b"@BIN@", f"Expected @BIN@, got {marker!r}"
        data = self.buf.read(n * 4)
        assert len(data) == n * 4, f"Expected {n * 4} bytes, got {len(data)}"
        trail = self.buf.read(1)
        assert trail == b"\n", f"Expected newline after binary, got {trail!r}"
        return torch.from_numpy(np.frombuffer(data, dtype="<f4").copy())


# ---------------------------------------------------------------------------
# Reading primitives
# ---------------------------------------------------------------------------

def _read_conv(reader):
    """Read a conv weight block, return Conv2d with loaded weights."""
    _name = reader.readline()
    h = reader.read_int()
    w = reader.read_int()
    ic = reader.read_int()
    oc = reader.read_int()
    _dily = reader.read_int()
    _dilx = reader.read_int()
    weights = reader.read_weights(h * w * ic * oc)
    weights = weights.view(h, w, ic, oc).permute(3, 2, 0, 1).contiguous()
    conv = nn.Conv2d(ic, oc, kernel_size=(h, w), padding="same", bias=False)
    conv.weight.data.copy_(weights)
    return conv


def _read_matmul(reader):
    """Read a matmul weight block, return Linear (no bias)."""
    _name = reader.readline()
    ic = reader.read_int()
    oc = reader.read_int()
    weights = reader.read_weights(ic * oc)
    weights = weights.view(ic, oc).permute(1, 0).contiguous()
    linear = nn.Linear(ic, oc, bias=False)
    linear.weight.data.copy_(weights)
    return linear


def _read_matbias(reader, linear):
    """Read a matbias block and set bias on existing Linear module."""
    _name = reader.readline()
    oc = reader.read_int()
    bias = reader.read_weights(oc)
    assert linear.out_features == oc
    linear.bias = nn.Parameter(bias.clone())


def _read_norm(reader):
    """Read a bn/biasmask block, return NormBias module."""
    _name = reader.readline()
    c_in = reader.read_int()
    _epsilon = reader.read_float()
    has_gamma = reader.read_int()
    _has_beta = reader.read_int()

    _mean = reader.read_weights(c_in)   # ignored for fixup
    _var = reader.read_weights(c_in)    # ignored for fixup

    norm = NormBias(c_in, has_gamma=bool(has_gamma))
    if has_gamma:
        gamma = reader.read_weights(c_in)
        norm.gamma.data.copy_(gamma.view(1, c_in, 1, 1))

    beta = reader.read_weights(c_in)
    norm.beta.data.copy_(beta.view(1, c_in, 1, 1))
    return norm


def _read_activation(reader):
    """Read an activation entry, return the activation kind string."""
    _name = reader.readline()
    act_type = reader.readline()
    if act_type == "ACTIVATION_MISH":
        return "mish"
    elif act_type == "ACTIVATION_IDENTITY":
        return "identity"
    else:
        return "relu"


def _skip_matmul(reader):
    """Skip a matmul entry."""
    _name = reader.readline()
    ic = reader.read_int()
    oc = reader.read_int()
    reader.read_weights(ic * oc)


def _skip_matbias(reader):
    """Skip a matbias entry."""
    _name = reader.readline()
    oc = reader.read_int()
    reader.read_weights(oc)


def _skip_metadata_encoder(reader):
    """Skip the metadata encoder section."""
    _name = reader.readline()
    _c_input = reader.read_int()
    # linear1 + bias1 + act1 + linear2 + bias2 + act2 + linear3
    _skip_matmul(reader)
    _skip_matbias(reader)
    _read_activation(reader)
    _skip_matmul(reader)
    _skip_matbias(reader)
    _read_activation(reader)
    _skip_matmul(reader)


# ---------------------------------------------------------------------------
# Reading compound structures
# ---------------------------------------------------------------------------

def _read_normactconv(reader, has_gpool, act_kind="relu"):
    """Read a NormActConv, return module with loaded weights."""
    norm = _read_norm(reader)
    act_kind = _read_activation(reader)

    if has_gpool:
        conv1r = _read_conv(reader)
        conv1g = _read_conv(reader)
        normg = _read_norm(reader)
        _read_activation(reader)  # actg (same kind)
        linear_g = _read_matmul(reader)

        c_in = norm.c_in
        c_out = conv1r.out_channels
        c_gpool = conv1g.out_channels

        nac = NormActConv(c_in, c_out, c_gpool, 3, has_gamma=norm.has_gamma, act_kind=act_kind)
        nac.norm = norm
        nac.convpool.conv1r = conv1r
        nac.convpool.conv1g = conv1g
        nac.convpool.normg = normg
        nac.convpool.linear_g = linear_g
        return nac, act_kind
    else:
        conv = _read_conv(reader)
        ks = conv.kernel_size[0]

        nac = NormActConv(norm.c_in, conv.out_channels, None, ks, has_gamma=norm.has_gamma, act_kind=act_kind)
        nac.norm = norm
        nac.conv = conv
        return nac, act_kind


def _read_block(reader):
    """Read a block (recursive for nested bottleneck), return (module, act_kind)."""
    block_type = reader.readline()
    _name = reader.readline()

    if block_type == "nested_bottleneck_block":
        internal_length = reader.read_int()
        normactconvp, act_kind = _read_normactconv(reader, has_gpool=False)

        sub_blocks = []
        for _ in range(internal_length):
            sb, _ = _read_block(reader)
            sub_blocks.append(sb)

        normactconvq, _ = _read_normactconv(reader, has_gpool=False)

        c_main = normactconvp.norm.c_in
        c_mid = normactconvp.conv.out_channels

        # Determine c_gpool from first sub-block
        first_sub = sub_blocks[0]
        if first_sub.normactconv1.convpool is not None:
            c_gpool = first_sub.normactconv1.convpool.conv1g.out_channels
        else:
            c_gpool = None

        block = NestedBottleneckResBlock(c_main, c_mid, c_gpool, internal_length, act_kind=act_kind)
        block.normactconvp = normactconvp
        for i, sb in enumerate(sub_blocks):
            block.blockstack[i] = sb
        block.normactconvq = normactconvq
        return block, act_kind

    elif block_type == "ordinary_block":
        nac1, act_kind = _read_normactconv(reader, has_gpool=False)
        nac2, _ = _read_normactconv(reader, has_gpool=False)

        c_main = nac1.norm.c_in
        c_mid = nac1.conv.out_channels

        block = ResBlock(c_main, c_mid, None, act_kind=act_kind)
        block.normactconv1 = nac1
        block.normactconv2 = nac2
        return block, act_kind

    elif block_type == "gpool_block":
        nac1, act_kind = _read_normactconv(reader, has_gpool=True)
        nac2, _ = _read_normactconv(reader, has_gpool=False)

        c_main = nac1.norm.c_in
        c_gpool = nac1.convpool.conv1g.out_channels
        c_out1 = nac1.convpool.conv1r.out_channels
        c_mid = c_out1 + c_gpool

        block = ResBlock(c_main, c_mid, c_gpool, act_kind=act_kind)
        block.normactconv1 = nac1
        block.normactconv2 = nac2
        return block, act_kind

    else:
        raise ValueError(f"Unknown block type: {block_type}")


def _read_policy_head(reader, version, act_kind="relu"):
    """Read policy head, return CNNPolicyHead module."""
    _name = reader.readline()
    conv1p = _read_conv(reader)
    conv1g = _read_conv(reader)
    biasg = _read_norm(reader)
    act_kind = _read_activation(reader)  # actg
    linear_g = _read_matmul(reader)
    bias2 = _read_norm(reader)
    _read_activation(reader)  # act2

    c_in = conv1p.in_channels
    c_p1 = conv1p.out_channels
    c_g1 = conv1g.out_channels

    conv2p = _read_conv(reader)

    if version >= 15:
        # v15: linear_pass (full), bias_pass, act_pass, linear_pass2 (2ch)
        linear_pass = _read_matmul(reader)
        _read_matbias(reader, linear_pass)
        _read_activation(reader)  # act_pass
        linear_pass2 = _read_matmul(reader)
        v15_pass = True
    else:
        # v14: linear_pass directly outputs 2 channels
        linear_pass = _read_matmul(reader)
        linear_pass2 = None
        v15_pass = False

    head = CNNPolicyHead(c_in, c_p1, c_g1, v15_pass=v15_pass, act_kind=act_kind)
    head.conv1p = conv1p
    head.conv1g = conv1g
    head.biasg = biasg
    head.linear_g = linear_g
    head.bias2 = bias2
    head.conv2p = conv2p
    head.linear_pass = linear_pass
    if linear_pass2 is not None:
        head.linear_pass2 = linear_pass2
    return head


def _read_value_head(reader, act_kind="relu"):
    """Read value head, return CNNValueHead module."""
    _name = reader.readline()
    conv1 = _read_conv(reader)
    bias1 = _read_norm(reader)
    act_kind = _read_activation(reader)  # act1

    linear2 = _read_matmul(reader)
    _read_matbias(reader, linear2)
    _read_activation(reader)  # act2

    linear_valuehead = _read_matmul(reader)
    _read_matbias(reader, linear_valuehead)

    linear_miscvaluehead = _read_matmul(reader)
    _read_matbias(reader, linear_miscvaluehead)

    conv_ownership = _read_conv(reader)

    c_in = conv1.in_channels
    c_v1 = conv1.out_channels
    c_v2 = linear2.out_features

    head = CNNValueHead(c_in, c_v1, c_v2, act_kind=act_kind)
    head.conv1 = conv1
    head.bias1 = bias1
    head.linear2 = linear2
    head.linear_valuehead = linear_valuehead
    head.linear_miscvaluehead = linear_miscvaluehead
    head.conv_ownership = conv_ownership
    return head


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_bin_gz(filepath, pos_len=19):
    """Load a KataGo bin.gz model file, return a CNNModel ready for inference.

    Args:
        filepath: Path to .bin.gz model file.
        pos_len: Board size (default 19).

    Returns:
        CNNModel with all weights loaded.
    """
    reader = BinGzReader(filepath)

    # ---- Header ----
    model_name = reader.readline()
    version = reader.read_int()
    assert version >= 12, f"Only v12+ supported, got {version}"
    num_bin_features = reader.read_int()
    num_global_features = reader.read_int()

    # Multipliers (v13+)
    if version > 12:
        _td_score_mult = reader.read_float()
        _scoremean_mult = reader.read_float()
        _scorestdev_mult = reader.read_float()
        _lead_mult = reader.read_float()
        _vtime_mult = reader.read_float()
        _stverr_mult = reader.read_float()
        _stserr_mult = reader.read_float()

    # v15 fields
    meta_encoder_version = 0
    if version >= 15:
        meta_encoder_version = reader.read_int()
        for _ in range(7):
            reader.read_int()  # placeholders

    # ---- Trunk ----
    trunk_marker = reader.readline()
    assert trunk_marker == "trunk", f"Expected 'trunk', got '{trunk_marker}'"

    num_blocks = reader.read_int()
    c_trunk = reader.read_int()
    c_mid = reader.read_int()
    _c_regular = reader.read_int()
    _c_gpool = reader.read_int()
    _c_gpool2 = reader.read_int()

    # v15 trunk placeholders
    if version >= 15:
        for _ in range(6):
            reader.read_int()

    conv_spatial = _read_conv(reader)
    linear_global = _read_matmul(reader)

    if meta_encoder_version > 0:
        _skip_metadata_encoder(reader)

    act_kind = "relu"  # default, will be detected from first block
    blocks = nn.ModuleList()
    for _ in range(num_blocks):
        block, act_kind = _read_block(reader)
        blocks.append(block)

    norm_trunkfinal = _read_norm(reader)
    act_kind_trunk = _read_activation(reader)

    # ---- Policy Head ----
    policy_head = _read_policy_head(reader, version, act_kind=act_kind)

    # ---- Value Head ----
    value_head = _read_value_head(reader, act_kind=act_kind)

    # ---- Assemble Model ----
    model = CNNModel(pos_len, act_kind=act_kind)
    model.c_trunk = c_trunk
    model.conv_spatial = conv_spatial
    model.linear_global = linear_global
    model.blocks = blocks
    model.norm_trunkfinal = norm_trunkfinal
    model.policy_head = policy_head
    model.value_head = value_head
    model.model_name = model_name

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {model_name}")
    print(f"  Version: {version}")
    print(f"  Activation: {act_kind}")
    print(f"  Input features: {num_bin_features} spatial, {num_global_features} global")
    print(f"  Trunk: {num_blocks} blocks, {c_trunk} channels, {c_mid} mid channels")
    print(f"  Policy: c_p1={policy_head.conv1p.out_channels}, c_g1={policy_head.conv1g.out_channels}")
    print(f"  Value: c_v1={value_head.conv1.out_channels}, c_v2={value_head.linear2.out_features}")
    print(f"  Parameters: {num_params:,}")

    return model
