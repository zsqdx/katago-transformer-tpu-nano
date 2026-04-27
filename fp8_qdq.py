"""Manual FP8 quantization for TensorRT FP8 inference without Transformer Engine.

Provides FP8 QuantizeLinear/DequantizeLinear (Q/DQ) insertion at the PyTorch level
so the legacy ONNX exporter emits standard ONNX Q/DQ ops that TensorRT can fuse
into FP8 GEMM kernels.

Usage:
    from fp8_qdq import convert_model_to_fp8, calibrate_activation_scales, fix_fp8_onnx_types

    model = Model(config, pos_len)
    model.load_state_dict(state_dict)
    convert_model_to_fp8(model)
    calibrate_activation_scales(model, dummy_spatial, dummy_global)

    torch.onnx.export(model, ..., opset_version=25, dynamo=False)
    fix_fp8_onnx_types("model.onnx")
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# E4M3FN max representable value (4-bit exponent, 3-bit mantissa).
FP8_E4M3_MAX = 448.0

# ONNX TensorProto enum value for FLOAT8E4M3FN.
_ONNX_FLOAT8E4M3FN = 17


# ---------------------------------------------------------------------------
# Core Q/DQ autograd function with ONNX symbolic
# ---------------------------------------------------------------------------
class FP8QuantDequant(torch.autograd.Function):
    """Fake-quantize a tensor to FP8 E4M3FN and immediately dequantize.

    During PyTorch execution the forward pass is an identity (the model was
    already trained with TE Float8CurrentScaling, so no additional noise is
    needed).  The ONNX symbolic emits standard ``QuantizeLinear`` +
    ``DequantizeLinear`` nodes that TensorRT recognises and fuses into FP8
    GEMM kernels.
    """

    @staticmethod
    def forward(ctx, x, scale_inv):
        # Identity -- only the ONNX graph matters.
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

    @staticmethod
    def symbolic(g, x, scale_inv):
        # QuantizeLinear with output_dtype=FLOAT8E4M3FN (opset 21+).
        quantized = g.op(
            "QuantizeLinear", x, scale_inv,
            output_dtype_i=_ONNX_FLOAT8E4M3FN,
        )
        # DequantizeLinear back to higher-precision.
        dequantized = g.op("DequantizeLinear", quantized, scale_inv)
        return dequantized


def compute_dynamic_scale_inv(x):
    """Compute per-tensor dynamic FP8 scale (ONNX convention).

    Returns ``scale_inv = clamp(amax(|x|), min=eps) / FP8_MAX``.
    All ops (abs, amax, clamp, div) are standard PyTorch ops that the legacy
    ONNX exporter maps to Abs, ReduceMax, Clip, Div respectively.
    """
    amax = x.abs().reshape(-1).max()
    scale_inv = torch.clamp(amax, min=1e-12) / FP8_E4M3_MAX
    return scale_inv


# ---------------------------------------------------------------------------
# FP8Linear -- drop-in replacement for nn.Linear with Q/DQ
# ---------------------------------------------------------------------------
class FP8Linear(nn.Module):
    """``nn.Linear`` wrapper that inserts FP8 Q/DQ on both activation and weight.

    Both activation and weight scales are **static buffers** that become ONNX
    constant initialisers.  Call :func:`calibrate_activation_scales` before
    export to populate the activation scales from a representative forward pass.

    In the exported ONNX graph each ``FP8Linear`` produces::

        activation ─→ QL(static_scale) ─→ DQL ─→ ┐
                                                    MatMul ─→ output
        weight ─────→ QL(static_scale) ─→ DQL ─→ ┘

    where QL = QuantizeLinear, DQL = DequantizeLinear.  TRT fuses the
    ``DQL → MatMul ← DQL`` pattern into an FP8 GEMM kernel.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        assert linear.bias is None, (
            "FP8Linear only supports bias=False (all transformer block "
            "projections in this model are bias-free)"
        )
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        # Share the original weight parameter.
        self.weight = linear.weight

        # Static weight scale (buffer → ONNX constant initialiser).
        self.register_buffer(
            "weight_scale_inv",
            self._compute_scale_inv(linear.weight),
        )

        # Static activation scale (placeholder; call calibrate_activation_scales
        # before export to populate from a representative forward pass).
        self.register_buffer("act_scale_inv", torch.tensor(1.0))

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _compute_scale_inv(w):
        amax = w.detach().abs().max()
        return torch.clamp(amax, min=1e-12) / FP8_E4M3_MAX

    def refresh_weight_scale(self):
        """Recompute static weight scale from current weight values."""
        self.weight_scale_inv.copy_(self._compute_scale_inv(self.weight))

    # -- forward -------------------------------------------------------------
    def forward(self, x):
        # Static activation Q/DQ (scale is a buffer → ONNX constant).
        x_qdq = FP8QuantDequant.apply(x, self.act_scale_inv)

        # Static weight Q/DQ.
        w_qdq = FP8QuantDequant.apply(self.weight, self.weight_scale_inv)

        return F.linear(x_qdq, w_qdq)


# ---------------------------------------------------------------------------
# Model conversion
# ---------------------------------------------------------------------------
# Linear layers inside TransformerBlock that should be quantised.
_BLOCK_LINEAR_ATTRS = [
    "q_proj", "k_proj", "v_proj", "out_proj",
    "ffn_w1", "ffn_wgate", "ffn_w2",
]


def convert_model_to_fp8(model):
    """Replace ``nn.Linear`` layers in all ``TransformerBlock``s with ``FP8Linear``.

    Follows the same convention as Transformer Engine ``use_fp8=True``:
    only the 7 Linear layers per transformer block are quantised.  The stem
    (Conv2d, linear_global) and output heads (PolicyHead, ValueHead) are left
    in higher precision because their dimensions are not FP8-aligned.

    The model is modified **in-place** and returned for convenience.
    """
    for block in model.blocks:
        for attr in _BLOCK_LINEAR_ATTRS:
            linear = getattr(block, attr)
            setattr(block, attr, FP8Linear(linear))
    return model


def refresh_all_weight_scales(model):
    """Recompute all static weight scales in the model (call before export)."""
    for module in model.modules():
        if isinstance(module, FP8Linear):
            module.refresh_weight_scale()


def calibrate_activation_scales(model, *sample_inputs):
    """Run one forward pass to capture activation amax for each FP8Linear.

    The captured values are written into each ``FP8Linear.act_scale_inv``
    buffer so that the ONNX export embeds them as constant initialisers.
    """
    hooks = []
    for module in model.modules():
        if isinstance(module, FP8Linear):
            def _make_hook(mod):
                def _hook(_self, inputs, _output):
                    x = inputs[0]
                    amax = x.detach().abs().max()
                    mod.act_scale_inv.copy_(
                        torch.clamp(amax, min=1e-12) / FP8_E4M3_MAX
                    )
                return _hook
            hooks.append(module.register_forward_hook(_make_hook(module)))

    with torch.no_grad():
        model(*sample_inputs)

    for h in hooks:
        h.remove()


# ---------------------------------------------------------------------------
# ONNX post-processing
# ---------------------------------------------------------------------------
def fix_fp8_onnx_types(onnx_path):
    """Fix FP8 tensor types and attributes in the exported ONNX model.

    The legacy PyTorch ONNX exporter may not correctly:
    1. Set ``output_dtype`` attribute on ``QuantizeLinear`` nodes.
    2. Set ``elem_type`` of Q output tensors to FLOAT8E4M3FN in value_info.

    This function patches both issues in-place.
    """
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(onnx_path)
    modified = False

    # -- Pass 1: ensure QuantizeLinear nodes have output_dtype attribute ------
    ql_output_names = set()
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear":
            ql_output_names.update(node.output)
            has_attr = any(a.name == "output_dtype" for a in node.attribute)
            if not has_attr:
                node.attribute.append(
                    helper.make_attribute("output_dtype", TensorProto.FLOAT8E4M3FN)
                )
                modified = True

    # -- Pass 2: fix value_info elem_type ------------------------------------
    existing_vi_names = {vi.name for vi in model.graph.value_info}
    for vi in model.graph.value_info:
        if vi.name in ql_output_names:
            if vi.type.tensor_type.elem_type != TensorProto.FLOAT8E4M3FN:
                vi.type.tensor_type.elem_type = TensorProto.FLOAT8E4M3FN
                modified = True

    # Add missing value_info entries for QL outputs.
    for name in ql_output_names:
        if name not in existing_vi_names:
            vi = helper.make_tensor_value_info(name, TensorProto.FLOAT8E4M3FN, None)
            model.graph.value_info.append(vi)
            modified = True

    if modified:
        onnx.save(
            model, onnx_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(onnx_path) + ".data",
        )
        ql_count = sum(1 for n in model.graph.node if n.op_type == "QuantizeLinear")
        dql_count = sum(1 for n in model.graph.node if n.op_type == "DequantizeLinear")
        print(f"Fixed FP8 types in ONNX: {ql_count} QuantizeLinear, {dql_count} DequantizeLinear nodes")

    return modified


def verify_fp8_qdq_structure(onnx_path, num_blocks, linears_per_block=7):
    """Verify the ONNX graph contains the expected number of FP8 Q/DQ nodes."""
    import onnx
    from onnx import TensorProto

    model = onnx.load(onnx_path, load_external_data=False)
    ql_nodes = [n for n in model.graph.node if n.op_type == "QuantizeLinear"]
    dql_nodes = [n for n in model.graph.node if n.op_type == "DequantizeLinear"]

    expected = num_blocks * linears_per_block * 2  # 2 Q/DQ per FP8Linear (act + weight)
    print(f"FP8 Q/DQ structure check:")
    print(f"  QuantizeLinear nodes:   {len(ql_nodes)} (expected {expected})")
    print(f"  DequantizeLinear nodes: {len(dql_nodes)} (expected {expected})")

    # Check output_dtype attribute.
    missing_attr = 0
    for node in ql_nodes:
        has_attr = any(
            a.name == "output_dtype" and a.i == TensorProto.FLOAT8E4M3FN
            for a in node.attribute
        )
        if not has_attr:
            missing_attr += 1
    if missing_attr:
        print(f"  WARNING: {missing_attr} QuantizeLinear nodes missing output_dtype=FLOAT8E4M3FN")

    ok = len(ql_nodes) == expected and len(dql_nodes) == expected and missing_attr == 0
    if ok:
        print("  Structure check PASSED")
    else:
        print("  Structure check FAILED")
    return ok
