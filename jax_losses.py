"""JAX loss function matching the PyTorch KataGo nano loss closely."""

import math

import jax
import jax.numpy as jnp


METRIC_KEYS = [
    "loss", "p0loss", "p1loss", "p0softloss", "p1softloss",
    "p0lopt", "p0sopt", "vloss", "tdvloss1", "tdvloss2", "tdvloss3",
    "tdsloss", "oloss", "sloss", "fploss", "skloss", "smloss",
    "sbcdfloss", "sbpdfloss", "sdregloss", "leadloss", "vtimeloss",
    "evstloss", "esstloss", "pacc1", "wsum",
]


def cross_entropy(logits, target_probs, axis):
    return -jnp.sum(target_probs * jax.nn.log_softmax(logits, axis=axis), axis=axis)


def huber_loss(pred, target, delta):
    err = pred - target
    abs_err = jnp.abs(err)
    quad = jnp.minimum(abs_err, delta)
    lin = abs_err - quad
    return 0.5 * quad * quad + delta * lin


def softplus_floor(x, square=False):
    # First JAX path uses the same forward values as PyTorch's custom autograd op.
    if square:
        return jnp.square(jax.nn.softplus(0.5 * x))
    return jax.nn.softplus(x)


def profile_loss_core(
    outputs,
    target_policy_ncmove,
    target_global,
    moving_unowned_proportion_sum,
    moving_unowned_proportion_weight,
    profile="policy_value",
    value_loss_scale=0.6,
):
    policy_logits, value_logits = outputs[0], outputs[1]
    n = policy_logits.shape[0]

    global_weight = target_global[:, 25]
    target_weight_policy_player = target_global[:, 26]
    target_weight_value = 1.0 - target_global[:, 35]

    target_policy_player = target_policy_ncmove[:, 0, :]
    target_policy_player = target_policy_player / jnp.sum(target_policy_player, axis=1, keepdims=True)
    target_value = target_global[:, 0:3]

    zero = jnp.asarray(0.0, dtype=jnp.float32)
    loss_policy_player = zero
    loss_value = zero
    policy_acc1 = zero

    if profile in ("policy_value", "policy_only"):
        loss_policy_player = (global_weight * target_weight_policy_player * cross_entropy(
            policy_logits[:, 0, :], target_policy_player, axis=1
        )).sum()
        policy_acc1 = (global_weight * target_weight_policy_player * (
            jnp.argmax(policy_logits[:, 0, :], axis=1) == jnp.argmax(target_policy_player, axis=1)
        ).astype(jnp.float32)).sum()

    if profile in ("policy_value", "value_only"):
        loss_value = 1.20 * (global_weight * target_weight_value * cross_entropy(
            value_logits, target_value, axis=1
        )).sum()

    if profile == "policy_value":
        loss_sum = (loss_policy_player * 0.93 + loss_value * value_loss_scale) / n
    elif profile == "policy_only":
        loss_sum = loss_policy_player * 0.93 / n
    elif profile == "value_only":
        loss_sum = loss_value * value_loss_scale / n
    else:
        raise ValueError(f"Unknown loss profile: {profile}")

    metrics = jnp.stack([
        loss_sum, loss_policy_player, zero,
        zero, zero,
        zero, zero,
        loss_value, zero, zero, zero,
        zero, zero, zero,
        zero, zero, zero,
        zero, zero, zero,
        zero, zero,
        zero, zero,
        policy_acc1, global_weight.sum(),
    ])
    return loss_sum, metrics, moving_unowned_proportion_sum, moving_unowned_proportion_weight


def postprocess_and_loss_core(
    outputs,
    score_belief_offset_vector,
    target_policy_ncmove,
    target_global,
    score_distr,
    target_value_nchw,
    pos_len,
    moving_unowned_proportion_sum,
    moving_unowned_proportion_weight,
    is_training=True,
    soft_policy_weight_scale=8.0,
    value_loss_scale=0.6,
    td_value_loss_scales=(0.6, 0.6, 0.6),
    seki_loss_scale=1.0,
    variance_time_loss_scale=1.0,
    disable_optimistic_policy=False,
):
    (
        policy_logits, value_logits, out_misc, out_moremisc,
        ownership_pretanh, pred_scoring, futurepos_pretanh, seki_logits,
        scorebelief_logits,
    ) = outputs
    td_value_logits = jnp.stack((out_misc[:, 4:7], out_misc[:, 7:10], out_moremisc[:, 2:5]), axis=1)
    pred_td_score = out_moremisc[:, 5:8] * 20.0
    pred_scoremean = out_misc[:, 0] * 20.0
    pred_scorestdev = softplus_floor(out_misc[:, 1], square=False) * 20.0
    pred_lead = out_misc[:, 2] * 20.0
    pred_variance_time = softplus_floor(out_misc[:, 3], square=False) * 40.0
    pred_shortterm_value_error = softplus_floor(out_moremisc[:, 0], square=True) * 0.25
    pred_shortterm_score_error = softplus_floor(out_moremisc[:, 1], square=True) * 150.0

    n = policy_logits.shape[0]
    pos_area = pos_len * pos_len

    target_policy_player = target_policy_ncmove[:, 0, :]
    target_policy_player = target_policy_player / jnp.sum(target_policy_player, axis=1, keepdims=True)
    target_policy_opponent = target_policy_ncmove[:, 1, :]
    target_policy_opponent = target_policy_opponent / jnp.sum(target_policy_opponent, axis=1, keepdims=True)

    target_policy_player_soft = jnp.power(target_policy_player + 1e-7, 0.25)
    target_policy_player_soft = target_policy_player_soft / jnp.sum(target_policy_player_soft, axis=1, keepdims=True)
    target_policy_opponent_soft = jnp.power(target_policy_opponent + 1e-7, 0.25)
    target_policy_opponent_soft = target_policy_opponent_soft / jnp.sum(target_policy_opponent_soft, axis=1, keepdims=True)

    target_weight_policy_player = target_global[:, 26]
    target_weight_policy_opponent = target_global[:, 28]
    target_value = target_global[:, 0:3]
    target_scoremean = target_global[:, 3]
    target_td_value = jnp.stack((target_global[:, 4:7], target_global[:, 8:11], target_global[:, 12:15]), axis=1)
    target_td_score = jnp.concatenate((target_global[:, 7:8], target_global[:, 11:12], target_global[:, 15:16]), axis=1)
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

    loss_policy_player = (global_weight * target_weight_policy_player * cross_entropy(
        policy_logits[:, 0, :], target_policy_player, axis=1
    )).sum()
    loss_policy_opponent = 0.15 * (global_weight * target_weight_policy_opponent * cross_entropy(
        policy_logits[:, 1, :], target_policy_opponent, axis=1
    )).sum()
    loss_policy_player_soft = (global_weight * target_weight_policy_player * cross_entropy(
        policy_logits[:, 2, :], target_policy_player_soft, axis=1
    )).sum()
    loss_policy_opponent_soft = 0.15 * (global_weight * target_weight_policy_opponent * cross_entropy(
        policy_logits[:, 3, :], target_policy_opponent_soft, axis=1
    )).sum()

    if disable_optimistic_policy:
        target_weight_longopt = target_weight_policy_player * 0.5
        target_weight_shortopt = target_weight_policy_player * 0.5
    else:
        win_squared = jnp.square(target_global[:, 0] + 0.5 * target_global[:, 2])
        longterm_score_stdevs_excess = (
            target_global[:, 3] - jax.lax.stop_gradient(pred_scoremean)
        ) / jnp.sqrt(jnp.square(jax.lax.stop_gradient(pred_scorestdev)) + 0.25)
        target_weight_longopt = jnp.clip(
            win_squared + jax.nn.sigmoid((longterm_score_stdevs_excess - 1.5) * 3.0),
            0.0,
            1.0,
        ) * target_weight_policy_player * target_weight_ownership

        shortterm_value_actual = target_global[:, 12] - target_global[:, 13]
        shortterm_value_pred = jax.nn.softmax(jax.lax.stop_gradient(td_value_logits[:, 2, :]), axis=1)
        shortterm_value_pred = shortterm_value_pred[:, 0] - shortterm_value_pred[:, 1]
        shortterm_value_stdevs_excess = (
            shortterm_value_actual - shortterm_value_pred
        ) / jnp.sqrt(jax.lax.stop_gradient(pred_shortterm_value_error) + 0.0001)
        shortterm_score_stdevs_excess = (
            target_global[:, 15] - jax.lax.stop_gradient(pred_td_score[:, 2])
        ) / jnp.sqrt(jax.lax.stop_gradient(pred_shortterm_score_error) + 0.25)
        target_weight_shortopt = jnp.clip(
            jax.nn.sigmoid((shortterm_value_stdevs_excess - 1.5) * 3.0)
            + jax.nn.sigmoid((shortterm_score_stdevs_excess - 1.5) * 3.0),
            0.0,
            1.0,
        ) * target_weight_policy_player * target_weight_ownership

    loss_longoptimistic_policy = (global_weight * target_weight_longopt * cross_entropy(
        policy_logits[:, 4, :], target_policy_player, axis=1
    )).sum()
    loss_shortoptimistic_policy = (global_weight * target_weight_shortopt * cross_entropy(
        policy_logits[:, 5, :], target_policy_player, axis=1
    )).sum()

    loss_value = 1.20 * (global_weight * target_weight_value * cross_entropy(
        value_logits, target_value, axis=1
    )).sum()
    td_loss_raw = cross_entropy(td_value_logits, target_td_value, axis=2) - cross_entropy(
        jnp.log(target_td_value + 1e-30), target_td_value, axis=2
    )
    td_loss_weighted = 1.20 * global_weight[:, None] * target_weight_td_value[:, None] * td_loss_raw
    loss_td_value1 = td_loss_weighted[:, 0].sum()
    loss_td_value2 = td_loss_weighted[:, 1].sum()
    loss_td_value3 = td_loss_weighted[:, 2].sum()
    loss_td_score = 0.0004 * (global_weight * target_weight_ownership * jnp.sum(
        huber_loss(pred_td_score, target_td_score, delta=12.0), axis=1
    )).sum()

    pred_own_logits = jnp.concatenate([ownership_pretanh, -ownership_pretanh], axis=1).reshape(n, 2, pos_area)
    target_own_probs = jnp.stack(
        [(1.0 + target_ownership) / 2.0, (1.0 - target_ownership) / 2.0], axis=1
    ).reshape(n, 2, pos_area)
    own_ce = cross_entropy(pred_own_logits, target_own_probs, axis=1)
    loss_ownership = 1.5 * (global_weight * target_weight_ownership * (jnp.sum(own_ce, axis=1) / pos_area)).sum()

    scoring_sq = jnp.square(jnp.squeeze(pred_scoring, axis=1) - target_scoring)
    loss_scoring_raw = jnp.sum(scoring_sq, axis=(1, 2)) / pos_area
    loss_scoring = (
        global_weight * target_weight_scoring * 4.0 * (jnp.sqrt(loss_scoring_raw * 0.5 + 1.0) - 1.0)
    ).sum()

    fp_weight = jnp.asarray([1.0, 0.25], dtype=jnp.float32).reshape(1, 2, 1, 1)
    fp_loss = jnp.square(jnp.tanh(futurepos_pretanh) - target_futurepos)
    loss_futurepos = 0.25 * (global_weight * target_weight_futurepos * (
        jnp.sum(fp_loss * fp_weight, axis=(1, 2, 3)) / math.sqrt(pos_area)
    )).sum()

    owned_target = jnp.square(target_ownership)
    unowned_target = 1.0 - owned_target
    unowned_proportion = jnp.mean(
        jnp.sum(unowned_target, axis=(1, 2)) / (1.0 + pos_area) * target_weight_ownership
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
    sign_target = jnp.stack([
        1.0 - jnp.square(target_seki),
        jax.nn.relu(target_seki),
        jax.nn.relu(-target_seki),
    ], axis=1)
    seki_sign_ce = cross_entropy(sign_pred, sign_target, axis=1)
    neutral_pred = jnp.stack([seki_logits[:, 3, :, :], jnp.zeros_like(target_ownership)], axis=1)
    neutral_target = jnp.stack([unowned_target, owned_target], axis=1)
    seki_neutral_ce = cross_entropy(neutral_pred, neutral_target, axis=1)
    loss_seki = (
        global_weight
        * seki_weight_scale
        * target_weight_ownership
        * (jnp.sum(seki_sign_ce, axis=(1, 2)) + 0.5 * jnp.sum(seki_neutral_ce, axis=(1, 2)))
        / pos_area
    ).sum()

    loss_scoremean = 0.0015 * (global_weight * target_weight_ownership * huber_loss(
        pred_scoremean, target_scoremean, delta=12.0
    )).sum()
    score_belief_probs = jnp.exp(scorebelief_logits)
    pred_cdf = jnp.cumsum(score_belief_probs, axis=1)
    target_cdf = jnp.cumsum(target_score_distribution, axis=1)
    loss_sb_cdf = 0.020 * (global_weight * target_weight_ownership * jnp.sum(
        jnp.square(pred_cdf - target_cdf), axis=1
    )).sum()
    loss_sb_pdf = 0.020 * (global_weight * target_weight_ownership * (
        -jnp.sum(target_score_distribution * scorebelief_logits, axis=1)
    )).sum()
    score_belief_offsets = score_belief_offset_vector.reshape(1, -1)
    expected_score = jnp.sum(score_belief_probs * score_belief_offsets, axis=1, keepdims=True)
    stdev_of_belief = jnp.sqrt(
        0.001 + jnp.sum(score_belief_probs * jnp.square(score_belief_offsets - expected_score), axis=1)
    )
    loss_scorestdev = 0.001 * (global_weight * huber_loss(
        pred_scorestdev, stdev_of_belief, delta=10.0
    )).sum()
    loss_lead = 0.0060 * (global_weight * target_weight_lead * huber_loss(
        pred_lead, target_lead, delta=8.0
    )).sum()
    loss_variance_time = 0.0003 * (global_weight * target_weight_ownership * huber_loss(
        pred_variance_time, target_variance_time + 1e-5, delta=50.0
    )).sum()

    td_val_pred_probs = jax.nn.softmax(td_value_logits[:, 2, :], axis=1)
    predvalue = jax.lax.stop_gradient(td_val_pred_probs[:, 0] - td_val_pred_probs[:, 1])
    realvalue = target_td_value[:, 2, 0] - target_td_value[:, 2, 1]
    sqerror_v = jnp.square(predvalue - realvalue) + 1e-8
    loss_st_value_error = 2.0 * (global_weight * target_weight_ownership * huber_loss(
        pred_shortterm_value_error, sqerror_v, delta=0.4
    )).sum()
    predscore = jax.lax.stop_gradient(pred_td_score[:, 2])
    realscore = target_td_score[:, 2]
    sqerror_s = jnp.square(predscore - realscore) + 1e-4
    loss_st_score_error = 0.00002 * (global_weight * target_weight_ownership * huber_loss(
        pred_shortterm_score_error, sqerror_s, delta=100.0
    )).sum()

    loss_sum = (
        loss_policy_player * 0.93
        + loss_policy_opponent
        + loss_policy_player_soft * soft_policy_weight_scale
        + loss_policy_opponent_soft * soft_policy_weight_scale
        + loss_longoptimistic_policy * 0.1
        + loss_shortoptimistic_policy * 0.2
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
    ) / n

    policy_acc1 = (global_weight * target_weight_policy_player * (
        jnp.argmax(policy_logits[:, 0, :], axis=1) == jnp.argmax(target_policy_player, axis=1)
    ).astype(jnp.float32)).sum()

    metrics = jnp.stack([
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
    return loss_sum, metrics, new_moving_sum, new_moving_weight
