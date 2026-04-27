#!/usr/bin/env python3
"""Export KataGo nano transformer checkpoint to native CUDA inference format."""

import argparse
import gzip
import os
import struct

import torch

from configs import get_num_bin_input_features, get_num_global_input_features, migrate_config
from model import Model


MAGIC = b"KGTRN001"
POLICY_CHANNELS = (0, 5)


def _write_u32(out, value):
    out.write(struct.pack("<I", int(value)))


def _write_i32(out, value):
    out.write(struct.pack("<i", int(value)))


def _write_f32(out, value):
    out.write(struct.pack("<f", float(value)))


def _write_string(out, value):
    data = value.encode("utf-8")
    _write_u32(out, len(data))
    out.write(data)


def _write_tensor(out, name, tensor):
    tensor = tensor.detach().cpu().float().contiguous()
    _write_string(out, name)
    _write_u32(out, tensor.ndim)
    for dim in tensor.shape:
        _write_i32(out, dim)
    out.write(tensor.numpy().tobytes(order="C"))


def _resolve_model_state(state, use_ema):
    model_state = dict(state["model"])
    if not use_ema:
        return model_state
    ema_shadow = state.get("ema_shadow")
    if ema_shadow is None:
        raise RuntimeError("--use-ema 指定了，但 checkpoint 中没有 ema_shadow")
    for name, tensor in ema_shadow.items():
        if name in model_state:
            model_state[name] = tensor
    return model_state


def _looks_like_te_checkpoint(model_state):
    return any(".layer.self_attention." in key for key in model_state)


def _convert_checkpoint_te_to_model_standalone(state_dict, zero_centered_norm=False):
    """把 TransformerEngine checkpoint 映射回 model.py 的权重命名。"""
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


def _make_model(checkpoint_path, pos_len, score_mode, use_ema):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = migrate_config(state["config"])

    if config.get("version") != 15:
        raise RuntimeError("原生 CUDA 导出当前只支持 version=15")
    varlen = state.get("varlen", False)
    zero_centered_norm = state.get("zero_centered_norm", False)
    model = Model(config, pos_len=pos_len, score_mode=score_mode, varlen=varlen, zero_centered_norm=zero_centered_norm)
    model_state = _resolve_model_state(state, use_ema)
    if _looks_like_te_checkpoint(model_state):
        print("检测到 TransformerEngine checkpoint，先转换为 model.py 权重命名")
        model_state = _convert_checkpoint_te_to_model_standalone(model_state, zero_centered_norm=zero_centered_norm)
    result = model.load_state_dict(model_state, strict=False)
    if result.missing_keys:
        print(f"Missing keys (will use default init): {result.missing_keys}")
    if result.unexpected_keys:
        print(f"Unexpected keys (ignored): {result.unexpected_keys}")
    if zero_centered_norm:
        model.fuse_zero_centered_norm()
    model.eval()
    return model, config, varlen


def _transpose_linear(weight):
    return weight.detach().cpu().float().t().contiguous()


def _selected_policy_weight(weight):
    return _transpose_linear(weight[POLICY_CHANNELS, :])


def _full_policy_weight(weight):
    return _transpose_linear(weight)


def _selected_value_weights(model):
    linear_sv = model.value_head.linear_sv.weight.detach().cpu().float()
    spatial = linear_sv[: model.value_head.n_spatial, :]
    global_part = linear_sv[model.value_head.n_spatial :, :]

    value = global_part[0:3, :]
    misc = global_part[3:7, :]
    moremisc = global_part[13:15, :]
    score = torch.cat([misc, moremisc], dim=0)
    ownership = spatial[0:1, :]

    return (
        value.t().contiguous(),
        score.t().contiguous(),
        ownership.t().contiguous(),
    )


def _full_value_weights(model):
    linear_sv = model.value_head.linear_sv.weight.detach().cpu().float()
    spatial = linear_sv[: model.value_head.n_spatial, :]
    global_part = linear_sv[model.value_head.n_spatial :, :]

    value = global_part[0:3, :]
    misc = global_part[3:13, :]
    moremisc = global_part[13:21, :]
    ownership = spatial[0:1, :]
    scoring = spatial[1:2, :]
    futurepos = spatial[2:4, :]
    seki = spatial[4:8, :]

    return (
        value.t().contiguous(),
        misc.t().contiguous(),
        moremisc.t().contiguous(),
        ownership.t().contiguous(),
        scoring.t().contiguous(),
        futurepos.t().contiguous(),
        seki.t().contiguous(),
    )


def _get_bias(linear):
    """Get bias from a linear layer, or zeros if no bias."""
    if linear.bias is not None:
        return linear.bias.detach().cpu().float().contiguous()
    return torch.zeros(linear.out_features, dtype=torch.float32)


def _selected_value_biases(model):
    """Bias counterpart of _selected_value_weights."""
    bias = _get_bias(model.value_head.linear_sv)
    spatial = bias[: model.value_head.n_spatial]
    global_part = bias[model.value_head.n_spatial :]

    value = global_part[0:3].contiguous()
    misc = global_part[3:7]
    moremisc = global_part[13:15]
    score = torch.cat([misc, moremisc]).contiguous()
    ownership = spatial[0:1].contiguous()

    return value, score, ownership


def _full_value_biases(model):
    """Bias counterpart of _full_value_weights."""
    bias = _get_bias(model.value_head.linear_sv)
    spatial = bias[: model.value_head.n_spatial]
    global_part = bias[model.value_head.n_spatial :]

    value = global_part[0:3].contiguous()
    misc = global_part[3:13].contiguous()
    moremisc = global_part[13:21].contiguous()
    ownership = spatial[0:1].contiguous()
    scoring = spatial[1:2].contiguous()
    futurepos = spatial[2:4].contiguous()
    seki = spatial[4:8].contiguous()

    return value, misc, moremisc, ownership, scoring, futurepos, seki


def _score_mode_id(score_mode):
    return {"simple": 0, "mix": 1, "mixop": 2}[score_mode]


def _collect_tensors(model):
    tensors = []
    tensors.append(("stem.conv.weight", model.conv_spatial.weight.detach().cpu().float().contiguous()))
    tensors.append(("stem.global.weight", _transpose_linear(model.linear_global.weight)))

    rope_cos = model.rope_cos.squeeze(1).squeeze(1).detach().cpu().float().contiguous()
    rope_sin = model.rope_sin.squeeze(1).squeeze(1).detach().cpu().float().contiguous()
    tensors.append(("rope.cos", rope_cos))
    tensors.append(("rope.sin", rope_sin))

    for i, block in enumerate(model.blocks):
        prefix = f"blocks.{i}."
        tensors.extend([
            (prefix + "norm1.weight", block.norm1.norm.weight.detach().cpu().float().contiguous()),
            (prefix + "q.weight", _transpose_linear(block.q_proj.weight)),
            (prefix + "k.weight", _transpose_linear(block.k_proj.weight)),
            (prefix + "v.weight", _transpose_linear(block.v_proj.weight)),
            (prefix + "out.weight", _transpose_linear(block.out_proj.weight)),
            (prefix + "norm2.weight", block.norm2.norm.weight.detach().cpu().float().contiguous()),
            (prefix + "ffn_w1.weight", _transpose_linear(block.ffn_w1.weight)),
            (prefix + "ffn_wgate.weight", _transpose_linear(block.ffn_wgate.weight)),
            (prefix + "ffn_w2.weight", _transpose_linear(block.ffn_w2.weight)),
        ])

    tensors.append(("final_norm.weight", model.norm_final.norm.weight.detach().cpu().float().contiguous()))

    # Policy head
    tensors.append(("policy.board_full.weight", _full_policy_weight(model.policy_head.linear_board.weight)))
    tensors.append(("policy.pass_full.weight", _full_policy_weight(model.policy_head.linear_pass.weight)))
    tensors.append(("policy.board.weight", _selected_policy_weight(model.policy_head.linear_board.weight)))
    tensors.append(("policy.pass.weight", _selected_policy_weight(model.policy_head.linear_pass.weight)))
    tensors.append(("policy.board_full.bias", _get_bias(model.policy_head.linear_board)))
    tensors.append(("policy.pass_full.bias", _get_bias(model.policy_head.linear_pass)))
    tensors.append(("policy.board.bias", _get_bias(model.policy_head.linear_board)[list(POLICY_CHANNELS)].contiguous()))
    tensors.append(("policy.pass.bias", _get_bias(model.policy_head.linear_pass)[list(POLICY_CHANNELS)].contiguous()))

    # Value head (selected)
    value_w, score_w, ownership_w = _selected_value_weights(model)
    tensors.append(("value.value.weight", value_w))
    tensors.append(("value.score.weight", score_w))
    tensors.append(("value.ownership.weight", ownership_w))
    value_b, score_b, ownership_b = _selected_value_biases(model)
    tensors.append(("value.value.bias", value_b))
    tensors.append(("value.score.bias", score_b))
    tensors.append(("value.ownership.bias", ownership_b))

    # Value head (full)
    _unused_value_w, misc_w, moremisc_w, _unused_ownership_w, scoring_w, futurepos_w, seki_w = _full_value_weights(model)
    tensors.append(("value.misc.weight", misc_w))
    tensors.append(("value.moremisc.weight", moremisc_w))
    tensors.append(("value.scoring.weight", scoring_w))
    tensors.append(("value.futurepos.weight", futurepos_w))
    tensors.append(("value.seki.weight", seki_w))
    _unused_value_b, misc_b, moremisc_b, _unused_ownership_b, scoring_b, futurepos_b, seki_b = _full_value_biases(model)
    tensors.append(("value.misc.bias", misc_b))
    tensors.append(("value.moremisc.bias", moremisc_b))
    tensors.append(("value.scoring.bias", scoring_b))
    tensors.append(("value.futurepos.bias", futurepos_b))
    tensors.append(("value.seki.bias", seki_b))

    # Score belief head
    score_mode = model.value_head.score_mode
    if score_mode == "simple":
        tensors.append(("scorebelief.simple.weight", _transpose_linear(model.value_head.linear_s_simple.weight)))
        tensors.append(("scorebelief.simple.bias", _get_bias(model.value_head.linear_s_simple)))
    else:
        tensors.append(("scorebelief.mix.weight", _transpose_linear(model.value_head.linear_s_mix.weight)))
        tensors.append(("scorebelief.mix.bias", _get_bias(model.value_head.linear_s_mix)))
        if score_mode == "mixop":
            tensors.append(("scorebelief.s2off.weight", _transpose_linear(model.value_head.linear_s2off.weight)))
            tensors.append(("scorebelief.s2par.weight", _transpose_linear(model.value_head.linear_s2par.weight)))
            tensors.append(("scorebelief.s2off.bias", _get_bias(model.value_head.linear_s2off)))
            tensors.append(("scorebelief.s2par.bias", _get_bias(model.value_head.linear_s2par)))
    return tensors


def _open_output(path):
    if path.endswith(".gz"):
        return gzip.open(path, "wb")
    return open(path, "wb")


def export_checkpoint(checkpoint, output, pos_len, score_mode, use_ema):
    model, config, varlen = _make_model(checkpoint, pos_len, score_mode, use_ema)
    format_version = 3
    tensors = _collect_tensors(model)

    print(f"format_version={format_version}")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with _open_output(output) as out:
        out.write(MAGIC)
        _write_u32(out, format_version)
        _write_string(out, os.path.splitext(os.path.basename(output))[0])
        _write_i32(out, config["version"])
        _write_i32(out, pos_len)
        _write_i32(out, config["hidden_size"])
        _write_i32(out, config["num_layers"])
        _write_i32(out, config["num_heads"])
        _write_i32(out, config["ffn_dim"])
        _write_i32(out, get_num_bin_input_features(config))
        _write_i32(out, get_num_global_input_features(config))
        _write_i32(out, 3)  # stem conv kernel_size
        _write_i32(out, 0)  # APE flag (always disabled)
        _write_i32(out, _score_mode_id(model.value_head.score_mode))
        _write_i32(out, model.value_head.num_scorebeliefs)
        _write_i32(out, model.value_head.scorebelief_len)

        _write_f32(out, 20.0)
        _write_f32(out, 20.0)
        _write_f32(out, 20.0)
        _write_f32(out, 20.0)
        _write_f32(out, 40.0)
        _write_f32(out, 0.25)
        _write_f32(out, 150.0)
        _write_f32(out, 1.0)

        _write_u32(out, len(tensors))
        for name, tensor in tensors:
            _write_tensor(out, name, tensor)


def main():
    parser = argparse.ArgumentParser(description="导出 KataGo nano Transformer 到原生 CUDA 推理格式")
    parser.add_argument("--checkpoint", required=True, help="checkpoint.ckpt 路径")
    parser.add_argument("--output", required=True, help="输出模型路径，建议使用 .kgtr.gz")
    parser.add_argument("--pos-len", type=int, default=19, help="棋盘边长")
    parser.add_argument("--score-mode", default="mixop", choices=["simple", "mix", "mixop"])
    parser.add_argument("--use-ema", action="store_true", help="优先导出 EMA 权重")
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output, args.pos_len, args.score_mode, args.use_ema)


if __name__ == "__main__":
    main()
