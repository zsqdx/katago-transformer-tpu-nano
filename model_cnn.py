"""Simplified KataGo CNN (NBT) inference model for evaluation.

Only supports fixup norm + nested bottleneck (bottlenest2) architecture.
Supports relu and mish activations. Loaded from bin.gz via load_bin_gz.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import EXTRA_SCORE_DISTR_RADIUS


def _act(x, act_kind):
    """Apply activation function."""
    if act_kind == "mish":
        return F.mish(x)
    else:
        return F.relu(x)


class NormBias(nn.Module):
    """Unified FixupNorm / BiasMask for inference.

    When loaded from bin.gz, gamma already includes all scale factors
    (e.g., (gamma+1)*scale for gamma_weight_decay_center_1 models).
    Forward: (x * gamma + beta) * mask if gamma else (x + beta) * mask.
    """

    def __init__(self, c_in, has_gamma):
        super().__init__()
        self.c_in = c_in
        self.has_gamma = has_gamma
        if has_gamma:
            self.gamma = nn.Parameter(torch.ones(1, c_in, 1, 1))
        else:
            self.gamma = None
        self.beta = nn.Parameter(torch.zeros(1, c_in, 1, 1))

    def forward(self, x, mask):
        if self.gamma is not None:
            return (x * self.gamma + self.beta) * mask
        else:
            return (x + self.beta) * mask


class KataGPool(nn.Module):
    """Global pooling: mean, mean * sqrt_offset/10, max."""

    def forward(self, x, mask, mask_sum_hw):
        offset = torch.sqrt(mask_sum_hw) - 14.0
        mean = torch.sum(x, dim=(2, 3), keepdim=True, dtype=torch.float32) / mask_sum_hw
        layer_max, _ = torch.max(
            (x + (mask - 1.0)).view(x.shape[0], x.shape[1], -1).to(torch.float32),
            dim=2,
        )
        layer_max = layer_max.view(x.shape[0], x.shape[1], 1, 1)
        return torch.cat([mean, mean * (offset / 10.0), layer_max], dim=1)


class KataValueHeadGPool(nn.Module):
    """Value head global pooling: mean, mean * offset/10, mean * (offset^2/100 - 0.1)."""

    def forward(self, x, mask, mask_sum_hw):
        offset = torch.sqrt(mask_sum_hw) - 14.0
        mean = torch.sum(x, dim=(2, 3), keepdim=True, dtype=torch.float32) / mask_sum_hw
        return torch.cat([
            mean,
            mean * (offset / 10.0),
            mean * (offset * offset / 100.0 - 0.1),
        ], dim=1)


class KataConvAndGPool(nn.Module):
    """Conv3x3 + [Conv3x3 -> norm -> act -> gpool -> linear] -> add."""

    def __init__(self, c_in, c_out, c_gpool, act_kind="relu"):
        super().__init__()
        self.act_kind = act_kind
        self.conv1r = nn.Conv2d(c_in, c_out, 3, padding="same", bias=False)
        self.conv1g = nn.Conv2d(c_in, c_gpool, 3, padding="same", bias=False)
        self.normg = NormBias(c_gpool, has_gamma=False)
        self.gpool = KataGPool()
        self.linear_g = nn.Linear(3 * c_gpool, c_out, bias=False)

    def forward(self, x, mask, mask_sum_hw):
        outr = self.conv1r(x)
        outg = self.conv1g(x)
        outg = self.normg(outg, mask)
        outg = _act(outg, self.act_kind)
        outg = self.gpool(outg, mask, mask_sum_hw).squeeze(-1).squeeze(-1)
        outg = self.linear_g(outg).unsqueeze(-1).unsqueeze(-1)
        return outr + outg


class NormActConv(nn.Module):
    """Norm -> Act -> Conv (or ConvAndGPool)."""

    def __init__(self, c_in, c_out, c_gpool, kernel_size, has_gamma, act_kind="relu"):
        super().__init__()
        self.act_kind = act_kind
        self.norm = NormBias(c_in, has_gamma)
        if c_gpool is not None:
            self.convpool = KataConvAndGPool(c_in, c_out, c_gpool, act_kind=act_kind)
            self.conv = None
        else:
            self.conv = nn.Conv2d(c_in, c_out, kernel_size, padding="same", bias=False)
            self.convpool = None

    def forward(self, x, mask, mask_sum_hw):
        out = self.norm(x, mask)
        out = _act(out, self.act_kind)
        if self.convpool is not None:
            out = self.convpool(out, mask, mask_sum_hw)
        else:
            out = self.conv(out)
        return out


class ResBlock(nn.Module):
    """Pre-activation residual block. First normactconv may have gpool."""

    def __init__(self, c_main, c_mid, c_gpool, act_kind="relu"):
        super().__init__()
        c_out1 = c_mid - (c_gpool if c_gpool is not None else 0)
        self.normactconv1 = NormActConv(c_main, c_out1, c_gpool, 3, has_gamma=False, act_kind=act_kind)
        self.normactconv2 = NormActConv(c_out1, c_main, None, 3, has_gamma=True, act_kind=act_kind)

    def forward(self, x, mask, mask_sum_hw):
        out = self.normactconv1(x, mask, mask_sum_hw)
        out = self.normactconv2(out, mask, mask_sum_hw)
        return x + out


class NestedBottleneckResBlock(nn.Module):
    """x + [1x1 down -> N ResBlocks -> 1x1 up]. Only first ResBlock has gpool."""

    def __init__(self, c_main, c_mid, c_gpool, internal_length=2, act_kind="relu"):
        super().__init__()
        self.internal_length = internal_length
        self.normactconvp = NormActConv(c_main, c_mid, None, 1, has_gamma=False, act_kind=act_kind)
        self.blockstack = nn.ModuleList()
        for i in range(internal_length):
            self.blockstack.append(ResBlock(
                c_main=c_mid,
                c_mid=c_mid,
                c_gpool=(c_gpool if i == 0 else None),
                act_kind=act_kind,
            ))
        self.normactconvq = NormActConv(c_mid, c_main, None, 1, has_gamma=True, act_kind=act_kind)

    def forward(self, x, mask, mask_sum_hw):
        out = self.normactconvp(x, mask, mask_sum_hw)
        for block in self.blockstack:
            out = block(out, mask, mask_sum_hw)
        out = self.normactconvq(out, mask, mask_sum_hw)
        return x + out


class CNNPolicyHead(nn.Module):
    """Policy head outputting (N, 6, L+1). bin.gz only has 2 channels (player + short-opt)."""

    def __init__(self, c_in, c_p1, c_g1, v15_pass=True, act_kind="relu"):
        super().__init__()
        self.act_kind = act_kind
        self.conv1p = nn.Conv2d(c_in, c_p1, 1, bias=False)
        self.conv1g = nn.Conv2d(c_in, c_g1, 1, bias=False)
        self.biasg = NormBias(c_g1, has_gamma=False)
        self.gpool = KataGPool()
        self.linear_g = nn.Linear(3 * c_g1, c_p1, bias=False)
        if v15_pass:
            # v15: linear -> bias -> act -> linear2
            self.linear_pass = nn.Linear(3 * c_g1, c_p1, bias=True)
            self.linear_pass2 = nn.Linear(c_p1, 2, bias=False)
        else:
            # v14: linear directly outputs 2 channels
            self.linear_pass = nn.Linear(3 * c_g1, 2, bias=False)
            self.linear_pass2 = None
        self.bias2 = NormBias(c_p1, has_gamma=False)
        self.conv2p = nn.Conv2d(c_p1, 2, 1, bias=False)

    def forward(self, x, mask, mask_sum_hw):
        N = x.shape[0]
        L = x.shape[2] * x.shape[3]

        outp = self.conv1p(x)
        outg = self.conv1g(x)
        outg = self.biasg(outg, mask)
        outg = _act(outg, self.act_kind)
        outg = self.gpool(outg, mask, mask_sum_hw).squeeze(-1).squeeze(-1)

        # Pass prediction
        if self.linear_pass2 is not None:
            outpass = _act(self.linear_pass(outg), self.act_kind)
            outpass = self.linear_pass2(outpass)  # (N, 2)
        else:
            outpass = self.linear_pass(outg)  # (N, 2)

        # Board prediction
        outg = self.linear_g(outg).unsqueeze(-1).unsqueeze(-1)
        outp = outp + outg
        outp = self.bias2(outp, mask)
        outp = _act(outp, self.act_kind)
        outp = self.conv2p(outp)  # (N, 2, H, W)
        outp = outp - (1.0 - mask) * 5000.0

        # Combine board + pass -> (N, 2, L+1)
        policy_2ch = torch.cat([outp.view(N, 2, L), outpass.unsqueeze(-1)], dim=2)

        # Expand to 6 channels: ch0=player, ch5=short-opt, rest=0
        policy_6ch = torch.zeros(N, 6, L + 1, device=x.device, dtype=x.dtype)
        policy_6ch[:, 0, :] = policy_2ch[:, 0, :]
        policy_6ch[:, 5, :] = policy_2ch[:, 1, :]
        return policy_6ch


class CNNValueHead(nn.Module):
    """Value head. bin.gz has: value(3), misc(6), ownership(1)."""

    def __init__(self, c_in, c_v1, c_v2, act_kind="relu"):
        super().__init__()
        self.act_kind = act_kind
        self.conv1 = nn.Conv2d(c_in, c_v1, 1, bias=False)
        self.bias1 = NormBias(c_v1, has_gamma=False)
        self.gpool = KataValueHeadGPool()
        self.linear2 = nn.Linear(3 * c_v1, c_v2, bias=True)
        self.linear_valuehead = nn.Linear(c_v2, 3, bias=True)
        self.linear_miscvaluehead = nn.Linear(c_v2, 6, bias=True)
        self.conv_ownership = nn.Conv2d(c_v1, 1, 1, bias=False)

    def forward(self, x, mask, mask_sum_hw):
        outv1 = self.conv1(x)
        outv1 = self.bias1(outv1, mask)
        outv1 = _act(outv1, self.act_kind)

        outpooled = self.gpool(outv1, mask, mask_sum_hw).squeeze(-1).squeeze(-1)
        outv2 = _act(self.linear2(outpooled), self.act_kind)

        out_value = self.linear_valuehead(outv2)       # (N, 3)
        out_misc_6 = self.linear_miscvaluehead(outv2)  # (N, 6)
        out_ownership = self.conv_ownership(outv1) * mask  # (N, 1, H, W)
        return out_value, out_misc_6, out_ownership


class CNNModel(nn.Module):
    """KataGo CNN model for inference. Modules are set by load_bin_gz."""

    def __init__(self, pos_len, act_kind="relu"):
        super().__init__()
        self.pos_len = pos_len
        self.act_kind = act_kind
        # Modules set by loader: conv_spatial, linear_global, blocks,
        # norm_trunkfinal, policy_head, value_head
        self.moving_unowned_proportion_sum = 0.0
        self.moving_unowned_proportion_weight = 0.0

    def forward(self, input_spatial, input_global):
        N = input_spatial.shape[0]
        H = W = self.pos_len
        L = H * W
        dev = input_spatial.device

        mask = input_spatial[:, 0:1, :, :]
        mask_sum_hw = torch.sum(mask, dim=(2, 3), keepdim=True)

        x = self.conv_spatial(input_spatial)
        x = x + self.linear_global(input_global).unsqueeze(-1).unsqueeze(-1)

        for block in self.blocks:
            x = block(x, mask, mask_sum_hw)

        x = self.norm_trunkfinal(x, mask)
        x = _act(x, self.act_kind)

        out_policy = self.policy_head(x, mask, mask_sum_hw)
        out_value, out_misc_6, out_ownership = self.value_head(x, mask, mask_sum_hw)

        # Map to nano 9-tuple: fill missing heads with zeros
        out_misc = torch.zeros(N, 10, device=dev)
        out_misc[:, :4] = out_misc_6[:, :4]  # scoremean, scorestdev, lead, vtime

        out_moremisc = torch.zeros(N, 8, device=dev)
        out_moremisc[:, :2] = out_misc_6[:, 4:6]  # stverr, stserr

        out_scoring = torch.zeros(N, 1, H, W, device=dev)
        out_futurepos = torch.zeros(N, 2, H, W, device=dev)
        out_seki = torch.zeros(N, 4, H, W, device=dev)

        scorebelief_len = (L + EXTRA_SCORE_DISTR_RADIUS) * 2
        out_scorebelief = torch.zeros(N, scorebelief_len, device=dev)

        return (
            out_policy.float(), out_value.float(), out_misc.float(), out_moremisc.float(),
            out_ownership.float(), out_scoring, out_futurepos, out_seki,
            out_scorebelief,
        )
