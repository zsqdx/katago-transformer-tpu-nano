"""Loss functions and FLOPs estimation for KataGo nano training."""

import logging
import math

import torch
import torch.nn.functional as F

from model import EXTRA_SCORE_DISTR_RADIUS, SoftPlusWithGradientFloor, cross_entropy
from configs import get_num_bin_input_features, get_num_global_input_features


_METRIC_KEYS = [
    "loss", "p0loss", "p1loss", "p0softloss", "p1softloss",
    "p0lopt", "p0sopt", "vloss", "tdvloss1", "tdvloss2", "tdvloss3",
    "tdsloss", "oloss", "sloss", "fploss", "skloss", "smloss",
    "sbcdfloss", "sbpdfloss", "sdregloss", "leadloss", "vtimeloss",
    "evstloss", "esstloss", "pacc1", "wsum",
]


def postprocess_and_loss_core(
    outputs,
    score_belief_offset_vector,
    target_policy_ncmove, target_global, score_distr, target_value_nchw,
    pos_len,
    moving_unowned_proportion_sum, moving_unowned_proportion_weight,
    is_training,
    soft_policy_weight_scale=8.0,
    value_loss_scale=0.6,
    td_value_loss_scales=(0.6, 0.6, 0.6),
    seki_loss_scale=1.0,
    variance_time_loss_scale=1.0,
    disable_optimistic_policy=False,
    mask=None,
):
    """Postprocess model outputs and compute loss. torch.compile friendly.

    Returns (loss_sum, metrics_stack, new_moving_sum, new_moving_weight) where
    metrics_stack is a 1-D tensor of shape (26,) matching _METRIC_KEYS order.
    """
    # --- Postprocess (inlined from model.postprocess) ---
    (
        out_policy, out_value, out_misc, out_moremisc,
        out_ownership, out_scoring, out_futurepos, out_seki,
        out_scorebelief,
    ) = outputs

    policy_logits = out_policy
    value_logits = out_value
    td_value_logits = torch.stack(
        (out_misc[:, 4:7], out_misc[:, 7:10], out_moremisc[:, 2:5]), dim=1
    )
    pred_td_score = out_moremisc[:, 5:8] * 20.0
    ownership_pretanh = out_ownership
    pred_scoring = out_scoring
    futurepos_pretanh = out_futurepos
    seki_logits = out_seki
    pred_scoremean = out_misc[:, 0] * 20.0
    pred_scorestdev = SoftPlusWithGradientFloor.apply(out_misc[:, 1], 0.05, False) * 20.0
    pred_lead = out_misc[:, 2] * 20.0
    pred_variance_time = SoftPlusWithGradientFloor.apply(out_misc[:, 3], 0.05, False) * 40.0
    pred_shortterm_value_error = SoftPlusWithGradientFloor.apply(out_moremisc[:, 0], 0.05, True) * 0.25
    pred_shortterm_score_error = SoftPlusWithGradientFloor.apply(out_moremisc[:, 1], 0.05, True) * 150.0
    scorebelief_logits = out_scorebelief

    # --- Loss computation ---
    N = policy_logits.shape[0]
    pos_area = pos_len * pos_len

    # Mask setup for variable-length board inputs
    if mask is not None:
        # mask: (N, 1, H, W) float, 1=valid, 0=padding
        mask_hw = mask.view(N, pos_area)  # (N, L)
        mask_n1hw = mask.view(N, 1, pos_len, pos_len)  # (N, 1, H, W)
        pos_area_per_sample = mask_hw.sum(dim=1)  # (N,)
    else:
        mask_hw = None
        mask_n1hw = None
        pos_area_per_sample = None

    # Target distributions
    target_policy_player = target_policy_ncmove[:, 0, :]
    target_policy_player = target_policy_player / torch.sum(target_policy_player, dim=1, keepdim=True)
    target_policy_opponent = target_policy_ncmove[:, 1, :]
    target_policy_opponent = target_policy_opponent / torch.sum(target_policy_opponent, dim=1, keepdim=True)

    # Soft policy targets (0.25 power smoothing)
    # policymask: zero out padding positions so that (0+1e-7)^0.25 doesn't
    # leak probability mass onto board positions masked to -5000 logits.
    if mask_hw is not None:
        policymask = torch.cat([mask_hw, mask_hw.new_ones(N, 1)], dim=1)  # (N, L+1)
    else:
        policymask = 1.0
    target_policy_player_soft = (target_policy_player + 1e-7) * policymask
    target_policy_player_soft = torch.pow(target_policy_player_soft, 0.25)
    target_policy_player_soft = target_policy_player_soft / torch.sum(target_policy_player_soft, dim=1, keepdim=True)
    target_policy_opponent_soft = (target_policy_opponent + 1e-7) * policymask
    target_policy_opponent_soft = torch.pow(target_policy_opponent_soft, 0.25)
    target_policy_opponent_soft = target_policy_opponent_soft / torch.sum(target_policy_opponent_soft, dim=1, keepdim=True)

    target_weight_policy_player = target_global[:, 26]
    target_weight_policy_opponent = target_global[:, 28]
    target_value = target_global[:, 0:3]
    target_scoremean = target_global[:, 3]
    target_td_value = torch.stack(
        (target_global[:, 4:7], target_global[:, 8:11], target_global[:, 12:15]), dim=1
    )
    target_td_score = torch.cat(
        (target_global[:, 7:8], target_global[:, 11:12], target_global[:, 15:16]), dim=1
    )
    target_lead = target_global[:, 21]
    target_variance_time = target_global[:, 22]
    global_weight = target_global[:, 25]
    target_weight_ownership = target_global[:, 27]
    target_weight_lead = target_global[:, 29]
    target_weight_futurepos = target_global[:, 33]
    target_weight_scoring = target_global[:, 34]
    target_weight_value = 1.0 - target_global[:, 35]
    target_weight_td_value = 1.0 - target_global[:, 24]

    target_score_distribution = score_distr / 100.0
    target_ownership = target_value_nchw[:, 0, :, :]
    target_seki = target_value_nchw[:, 1, :, :]
    target_futurepos = target_value_nchw[:, 2:4, :, :]
    target_scoring = target_value_nchw[:, 4, :, :] / 120.0

    # --- Policy loss coefficients ---
    policy_opt_loss_scale = 0.93
    long_policy_opt_loss_scale = 0.1
    short_policy_opt_loss_scale = 0.2

    # --- Policy losses ---
    loss_policy_player = (global_weight * target_weight_policy_player * cross_entropy(
        policy_logits[:, 0, :], target_policy_player, dim=1
    )).sum()

    loss_policy_opponent = 0.15 * (global_weight * target_weight_policy_opponent * cross_entropy(
        policy_logits[:, 1, :], target_policy_opponent, dim=1
    )).sum()

    loss_policy_player_soft = (global_weight * target_weight_policy_player * cross_entropy(
        policy_logits[:, 2, :], target_policy_player_soft, dim=1
    )).sum()

    loss_policy_opponent_soft = 0.15 * (global_weight * target_weight_policy_opponent * cross_entropy(
        policy_logits[:, 3, :], target_policy_opponent_soft, dim=1
    )).sum()

    # --- Optimistic policy losses ---
    if disable_optimistic_policy:
        target_weight_longopt = target_weight_policy_player * 0.5
        loss_longoptimistic_policy = (global_weight * target_weight_longopt * cross_entropy(
            policy_logits[:, 4, :], target_policy_player, dim=1
        )).sum()
        target_weight_shortopt = target_weight_policy_player * 0.5
        loss_shortoptimistic_policy = (global_weight * target_weight_shortopt * cross_entropy(
            policy_logits[:, 5, :], target_policy_player, dim=1
        )).sum()
    else:
        # Long-term optimistic policy
        win_squared = torch.square(
            target_global[:, 0] + 0.5 * target_global[:, 2]
        )
        longterm_score_stdevs_excess = (
            target_global[:, 3] - pred_scoremean.detach()
        ) / torch.sqrt(torch.square(pred_scorestdev.detach()) + 0.25)
        target_weight_longopt = torch.clamp(
            win_squared + torch.sigmoid((longterm_score_stdevs_excess - 1.5) * 3.0),
            min=0.0, max=1.0,
        ) * target_weight_policy_player * target_weight_ownership
        loss_longoptimistic_policy = (global_weight * target_weight_longopt * cross_entropy(
            policy_logits[:, 4, :], target_policy_player, dim=1
        )).sum()

        # Short-term optimistic policy
        shortterm_value_actual = target_global[:, 12] - target_global[:, 13]
        shortterm_value_pred = torch.softmax(td_value_logits[:, 2, :].detach(), dim=1)
        shortterm_value_pred = shortterm_value_pred[:, 0] - shortterm_value_pred[:, 1]
        shortterm_value_stdevs_excess = (
            shortterm_value_actual - shortterm_value_pred
        ) / torch.sqrt(pred_shortterm_value_error.detach() + 0.0001)
        shortterm_score_stdevs_excess = (
            target_global[:, 15] - pred_td_score[:, 2].detach()
        ) / torch.sqrt(pred_shortterm_score_error.detach() + 0.25)
        target_weight_shortopt = torch.clamp(
            torch.sigmoid((shortterm_value_stdevs_excess - 1.5) * 3.0)
            + torch.sigmoid((shortterm_score_stdevs_excess - 1.5) * 3.0),
            min=0.0, max=1.0,
        ) * target_weight_policy_player * target_weight_ownership
        loss_shortoptimistic_policy = (global_weight * target_weight_shortopt * cross_entropy(
            policy_logits[:, 5, :], target_policy_player, dim=1
        )).sum()

    # --- Value loss ---
    loss_value = 1.20 * (global_weight * target_weight_value * cross_entropy(
        value_logits, target_value, dim=1
    )).sum()

    # TD value (3 independent terms)
    td_loss_raw = cross_entropy(td_value_logits, target_td_value, dim=2) - cross_entropy(
        torch.log(target_td_value + 1e-30), target_td_value, dim=2
    )
    td_loss_weighted = 1.20 * global_weight.unsqueeze(1) * target_weight_td_value.unsqueeze(1) * td_loss_raw
    loss_td_value1 = td_loss_weighted[:, 0].sum()
    loss_td_value2 = td_loss_weighted[:, 1].sum()
    loss_td_value3 = td_loss_weighted[:, 2].sum()

    loss_td_score = 0.0004 * (global_weight * target_weight_ownership * torch.sum(
        F.huber_loss(pred_td_score, target_td_score, reduction="none", delta=12.0), dim=1
    )).sum()

    # --- Spatial losses ---
    # Ownership
    pred_own_logits = torch.cat([ownership_pretanh, -ownership_pretanh], dim=1).view(N, 2, pos_area)
    target_own_probs = torch.stack([(1.0 + target_ownership) / 2.0, (1.0 - target_ownership) / 2.0], dim=1).view(N, 2, pos_area)
    own_ce = cross_entropy(pred_own_logits, target_own_probs, dim=1)  # (N, L)
    if mask_hw is not None:
        loss_ownership = 1.5 * (global_weight * target_weight_ownership * (
            torch.sum(own_ce * mask_hw, dim=1) / pos_area_per_sample
        )).sum()
    else:
        loss_ownership = 1.5 * (global_weight * target_weight_ownership * (
            torch.sum(own_ce, dim=1) / pos_area
        )).sum()

    # Scoring
    scoring_sq = torch.square(pred_scoring.squeeze(1) - target_scoring)  # (N, H, W)
    if mask_n1hw is not None:
        loss_scoring_raw = torch.sum(scoring_sq * mask_n1hw.squeeze(1), dim=(1, 2)) / pos_area_per_sample
    else:
        loss_scoring_raw = torch.sum(scoring_sq, dim=(1, 2)) / pos_area
    loss_scoring = (global_weight * target_weight_scoring * 4.0 * (torch.sqrt(loss_scoring_raw * 0.5 + 1.0) - 1.0)).sum()

    # Future position
    fp_loss = torch.square(torch.tanh(futurepos_pretanh) - target_futurepos)
    fp_weight = torch.tensor([1.0, 0.25], device=fp_loss.device).view(1, 2, 1, 1)
    if mask_n1hw is not None:
        loss_futurepos = 0.25 * (global_weight * target_weight_futurepos * (
            torch.sum(fp_loss * fp_weight * mask_n1hw, dim=(1, 2, 3)) / torch.sqrt(pos_area_per_sample)
        )).sum()
    else:
        loss_futurepos = 0.25 * (global_weight * target_weight_futurepos * (
            torch.sum(fp_loss * fp_weight, dim=(1, 2, 3)) / math.sqrt(pos_area)
        )).sum()

    # Seki (dynamic weight)
    owned_target = torch.square(target_ownership)
    unowned_target = 1.0 - owned_target

    # Seki dynamic weight: compute moving average and seki_weight_scale as tensors
    if mask_n1hw is not None:
        mask_hw_2d = mask_n1hw.squeeze(1)  # (N, H, W)
        unowned_proportion = torch.mean(
            torch.sum(unowned_target * mask_hw_2d, dim=(1, 2)) / (1.0 + pos_area_per_sample) * target_weight_ownership
        )
    else:
        unowned_proportion = torch.mean(
            torch.sum(unowned_target, dim=(1, 2)) / (1.0 + pos_area) * target_weight_ownership
        )

    if is_training:
        new_moving_sum = moving_unowned_proportion_sum * 0.998 + unowned_proportion
        new_moving_weight = moving_unowned_proportion_weight * 0.998 + 1.0
        moving_unowned_proportion = new_moving_sum / new_moving_weight
        seki_weight_scale = 8.0 * 0.005 / (0.005 + moving_unowned_proportion)
    else:
        new_moving_sum = moving_unowned_proportion_sum
        new_moving_weight = moving_unowned_proportion_weight
        seki_weight_scale = 7.0

    sign_pred = seki_logits[:, 0:3, :, :]
    sign_target = torch.stack([
        1.0 - torch.square(target_seki),
        F.relu(target_seki),
        F.relu(-target_seki),
    ], dim=1)
    seki_sign_ce = cross_entropy(sign_pred, sign_target, dim=1)  # (N, H, W)
    neutral_pred = torch.stack([seki_logits[:, 3, :, :], torch.zeros_like(target_ownership)], dim=1)
    neutral_target = torch.stack([unowned_target, owned_target], dim=1)
    seki_neutral_ce = cross_entropy(neutral_pred, neutral_target, dim=1)  # (N, H, W)
    if mask_n1hw is not None:
        loss_sign = torch.sum(seki_sign_ce * mask_hw_2d, dim=(1, 2))
        loss_neutral = torch.sum(seki_neutral_ce * mask_hw_2d, dim=(1, 2))
        loss_seki = (global_weight * seki_weight_scale * target_weight_ownership * (loss_sign + 0.5 * loss_neutral) / pos_area_per_sample).sum()
    else:
        loss_sign = torch.sum(seki_sign_ce, dim=(1, 2))
        loss_neutral = torch.sum(seki_neutral_ce, dim=(1, 2))
        loss_seki = (global_weight * seki_weight_scale * target_weight_ownership * (loss_sign + 0.5 * loss_neutral) / pos_area).sum()

    # --- Score belief loss ---
    loss_scoremean = 0.0015 * (global_weight * target_weight_ownership * F.huber_loss(
        pred_scoremean, target_scoremean, reduction="none", delta=12.0
    )).sum()

    pred_cdf = torch.cumsum(F.softmax(scorebelief_logits, dim=1), dim=1)
    target_cdf = torch.cumsum(target_score_distribution, dim=1)
    loss_sb_cdf = 0.020 * (global_weight * target_weight_ownership * torch.sum(
        torch.square(pred_cdf - target_cdf), dim=1
    )).sum()

    loss_sb_pdf = 0.020 * (global_weight * target_weight_ownership * cross_entropy(
        scorebelief_logits, target_score_distribution, dim=1
    )).sum()

    # Score stdev regularization
    score_belief_probs = F.softmax(scorebelief_logits, dim=1)
    score_belief_offsets = score_belief_offset_vector.view(1, -1)
    expected_score = torch.sum(score_belief_probs * score_belief_offsets, dim=1, keepdim=True)
    stdev_of_belief = torch.sqrt(0.001 + torch.sum(score_belief_probs * torch.square(score_belief_offsets - expected_score), dim=1))
    loss_scorestdev = 0.001 * (global_weight * F.huber_loss(pred_scorestdev, stdev_of_belief, reduction="none", delta=10.0)).sum()

    loss_lead = 0.0060 * (global_weight * target_weight_lead * F.huber_loss(
        pred_lead, target_lead, reduction="none", delta=8.0
    )).sum()

    loss_variance_time = 0.0003 * (global_weight * target_weight_ownership * F.huber_loss(
        pred_variance_time, target_variance_time + 1e-5, reduction="none", delta=50.0
    )).sum()

    # Short-term error losses
    td_val_pred_probs = torch.softmax(td_value_logits[:, 2, :], dim=1)
    predvalue = (td_val_pred_probs[:, 0] - td_val_pred_probs[:, 1]).detach()
    realvalue = target_td_value[:, 2, 0] - target_td_value[:, 2, 1]
    sqerror_v = torch.square(predvalue - realvalue) + 1e-8
    loss_st_value_error = 2.0 * (global_weight * target_weight_ownership * F.huber_loss(
        pred_shortterm_value_error, sqerror_v, reduction="none", delta=0.4
    )).sum()

    predscore = pred_td_score[:, 2].detach()
    realscore = target_td_score[:, 2]
    sqerror_s = torch.square(predscore - realscore) + 1e-4
    loss_st_score_error = 0.00002 * (global_weight * target_weight_ownership * F.huber_loss(
        pred_shortterm_score_error, sqerror_s, reduction="none", delta=100.0
    )).sum()

    # --- Total loss ---
    loss_sum = (
        loss_policy_player * policy_opt_loss_scale
        + loss_policy_opponent
        + loss_policy_player_soft * soft_policy_weight_scale
        + loss_policy_opponent_soft * soft_policy_weight_scale
        + loss_longoptimistic_policy * long_policy_opt_loss_scale
        + loss_shortoptimistic_policy * short_policy_opt_loss_scale
        + loss_value * value_loss_scale
        + loss_td_value1 * td_value_loss_scales[0]
        + loss_td_value2 * td_value_loss_scales[1]
        + loss_td_value3 * td_value_loss_scales[2]
        + loss_td_score
        + loss_ownership
        + loss_scoring * 0.25
        + loss_futurepos
        + loss_seki * seki_loss_scale
        + loss_scoremean
        + loss_sb_cdf
        + loss_sb_pdf
        + loss_scorestdev
        + loss_lead
        + loss_variance_time * variance_time_loss_scale
        + loss_st_value_error
        + loss_st_score_error
    ) / N

    # Accuracy
    policy_acc1 = (global_weight * target_weight_policy_player * (
        torch.argmax(policy_logits[:, 0, :], dim=1) == torch.argmax(target_policy_player, dim=1)
    ).float()).sum()

    metrics_stack = torch.stack([
        loss_sum, loss_policy_player, loss_policy_opponent,
        loss_policy_player_soft, loss_policy_opponent_soft,
        loss_longoptimistic_policy, loss_shortoptimistic_policy,
        loss_value, loss_td_value1, loss_td_value2, loss_td_value3,
        loss_td_score, loss_ownership, loss_scoring,
        loss_futurepos, loss_seki, loss_scoremean,
        loss_sb_cdf, loss_sb_pdf, loss_scorestdev,
        loss_lead, loss_variance_time,
        loss_st_value_error, loss_st_score_error,
        policy_acc1, global_weight.sum(),
    ])

    return loss_sum, metrics_stack, new_moving_sum, new_moving_weight


def compute_loss(
    model, outputs, batch, pos_len, is_training,
    soft_policy_weight_scale=8.0,
    value_loss_scale=0.6,
    td_value_loss_scales=(0.6, 0.6, 0.6),
    seki_loss_scale=1.0,
    variance_time_loss_scale=1.0,
    disable_optimistic_policy=False,
    mask=None,
):
    """Wrapper around postprocess_and_loss_core for backward compatibility.

    Accepts raw model outputs (9-tuple from forward()), handles seki moving
    average state, and returns (loss, metrics_dict).
    """
    dev = outputs[0].device
    moving_sum_t = torch.tensor(model.moving_unowned_proportion_sum, device=dev)
    moving_weight_t = torch.tensor(model.moving_unowned_proportion_weight, device=dev)

    loss_sum, metrics_stack, new_sum, new_weight = postprocess_and_loss_core(
        outputs, model.value_head.score_belief_offset_vector,
        batch["policyTargetsNCMove"],
        batch["globalTargetsNC"], batch["scoreDistrN"], batch["valueTargetsNCHW"],
        pos_len, moving_sum_t, moving_weight_t, is_training,
        soft_policy_weight_scale=soft_policy_weight_scale,
        value_loss_scale=value_loss_scale,
        td_value_loss_scales=td_value_loss_scales,
        seki_loss_scale=seki_loss_scale,
        variance_time_loss_scale=variance_time_loss_scale,
        disable_optimistic_policy=disable_optimistic_policy,
        mask=mask,
    )

    # Write back seki moving average state
    if is_training:
        model.moving_unowned_proportion_sum = new_sum.item()
        model.moving_unowned_proportion_weight = new_weight.item()

    metrics = dict(zip(_METRIC_KEYS, metrics_stack.tolist()))
    return loss_sum, metrics


def estimate_forward_flops(config, pos_len, score_mode="simple"):
    """Estimate forward-pass FLOPs for a single sample."""
    S = pos_len * pos_len
    D = config["hidden_size"]
    FF = config["ffn_dim"]
    num_blocks = config["num_layers"]

    # Per TransformerBlock (MHA: Q/K/V all have dimension D)
    attn_proj = 2 * S * D * D * 4  # Q, K, V, Out projections
    attn_scores = 2 * S * S * D
    attn_values = 2 * S * S * D
    ffn = 3 * 2 * S * D * FF  # SwiGLU: w1, wgate, w2
    block_flops = attn_proj + attn_scores + attn_values + ffn
    trunk_flops = block_flops * num_blocks

    # Input layer (stem conv)
    num_bin_features = get_num_bin_input_features(config)
    num_global_features = get_num_global_input_features(config)
    conv_flops = 2 * num_bin_features * D * 9 * S  # 3x3 conv
    global_flops = 2 * num_global_features * D

    # Output heads
    policy_flops = 2 * S * D * 6 + 2 * D * 6
    value_sv_flops = 2 * S * D * 29
    num_scorebeliefs = config["num_scorebeliefs"]
    scorebelief_len = (S + EXTRA_SCORE_DISTR_RADIUS) * 2
    if score_mode == "simple":
        score_flops = 2 * D * scorebelief_len
    else:
        score_mix_out = scorebelief_len * num_scorebeliefs + num_scorebeliefs
        score_flops = 2 * D * score_mix_out
        if score_mode == "mixop":
            # offset linear (D -> 1) and parity linear (D -> 1)
            score_flops += 2 * D * 2

    total = trunk_flops + conv_flops + global_flops + policy_flops + value_sv_flops + score_flops
    return total


def get_gpu_peak_tflops(device):
    """Return BF16 peak TFLOPS for MFU calculation."""
    if device.type != "cuda":
        return 0.0

    name = torch.cuda.get_device_name(device).lower()
    known_gpus = {
        "4090": 165.2,
        "4080 super": 97.5,
        "4080": 97.5,
        "4070 ti super": 93.2,
        "4070 ti": 40.1,
        "4070": 29.1,
        "3090 ti": 40.0,
        "3090": 35.6,
        "3080 ti": 34.1,
        "3080": 29.8,
        "a100 sxm": 312.0,
        "a100 pcie": 312.0,
        "a100": 312.0,
        "a6000": 38.7,
        "a10": 31.2,
        "h100 sxm": 989.5,
        "h100 pcie": 756.0,
        "h100": 756.0,
        "h200": 989.5,
        "l40s": 91.6,
        "l40": 90.5,
        "l4": 30.3,
    }
    for key, tflops in known_gpus.items():
        if key in name:
            return tflops

    props = torch.cuda.get_device_properties(device)
    clock_ghz = props.clock_rate / 1e6
    estimated = props.multi_processor_count * 128 * 2 * clock_ghz / 1e3
    logging.warning(
        f"Unknown GPU '{torch.cuda.get_device_name(device)}', "
        f"rough BF16 estimate: {estimated:.1f} TFLOPS (MFU may be inaccurate)"
    )
    return estimated
