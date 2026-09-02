"""RTSRNet: paper-facing, checkpoint-compatible refactor.

This file reorganizes the original GoodRT_v2 implementation using the module
names used in the RTSRNet manuscript:

    RTSRNet  : Residual Target-Sensitive Refinement Network
    SGDEM    : Semantic-Geometric Detail Enhancement Module
    DCUB     : Depth-Conditioned Update Block
    TGMM     : Target-guided Geometry Modulation Module (decoder-stage update)
    GCDUB    : Geometry-Conditioned Decoder Update Block

IMPORTANT: the forward tensor operations are intentionally preserved from the
provided GoodRT_v2 source so existing checkpoints remain usable. Old state_dict
keys are translated automatically by RTSRNet.load_state_dict().

The historical ablation switches are retained only for backward compatibility.
For the paper-facing model use ablation_mode="full".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.EfficientNet import EfficientNet_B0
from Model.modules import BasicConv2d, DepthwiseSeparableConv


VALID_ABLATION_MODES = ("baseline", "dgp", "dgp_dls", "idls", "goodr", "full")
ABLATION_DIR_NAMES = {
    "baseline": "RTSRNet_baseline",
    "dgp": "RTSRNet_legacy_dgp",
    "dgp_dls": "RTSRNet_legacy_dgp_dls",
    "idls": "RTSRNet_legacy_idls",
    "goodr": "RTSRNet_legacy_goodr",
    "full": "RTSRNet",
}


def normalize_ablation_mode(mode):
    if mode is None or mode == "":
        mode = "full"
    mode = str(mode).lower().strip()
    aliases = {
        "none": "baseline",
        "base": "baseline",
        "+dgp": "dgp",
        "dgp+dls": "dgp_dls",
        "idls": "idls",
        "i_dls": "idls",
        "i-dls": "idls",
        "iter_dls": "idls",
        "iterative_dls": "idls",
        "dgp+idls": "idls",
        "dgp+i-dls": "idls",
        "rdls": "goodr",
        "r_dls": "goodr",
        "r-dls": "goodr",
        "irdls": "goodr",
        "ir_dls": "goodr",
        "ir-dls": "goodr",
        "iterative_rdls": "goodr",
        "goodr": "goodr",
        "dgp+rdls": "goodr",
        "dgp+r-dls": "goodr",
        "dgp_dls_spr": "full",
        "dgp+dls+spr": "full",
        "dgp+rdls+spr": "full",
        "dgp+r-dls+spr": "full",
        "spr": "full",
    }
    mode = aliases.get(mode, mode)
    if mode not in VALID_ABLATION_MODES:
        raise ValueError(f"Unsupported ablation_mode={mode!r}. Choose one of {VALID_ABLATION_MODES}.")
    return mode


def ablation_flags(mode):
    mode = normalize_ablation_mode(mode)
    return {
        "use_dgp": mode in ("dgp", "dgp_dls", "idls", "goodr", "full"),
        "use_dls": mode in ("dgp_dls", "idls", "goodr", "full"),
        "use_idls": mode in ("idls", "goodr", "full"),
        "use_rdls": mode in ("goodr", "full"),
        "use_spr": mode == "full",
    }



class ReliabilitySuppressor(nn.Module):
    """Suppress-only reliability gate for R-DLS.

    This is intentionally used only inside TGMM.
    It never amplifies DLS. The modulation factor is:

        factor = 1 - alpha * (1 - reliability)

    where reliability in [0, 1] and alpha in [0, 1].
    Thus factor <= 1. Reliable regions keep ordinary DLS nearly unchanged,
    while unreliable depth-edge regions suppress liquid updates.
    """

    def __init__(self, in_channels, hidden_channels, suppression_init=0.20, reliability_bias=2.0):
        super().__init__()
        self.net = nn.Sequential(
            BasicConv2d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            DepthwiseSeparableConv(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, reliability_bias)

        suppression_init = float(max(1e-4, min(1.0 - 1e-4, suppression_init)))
        self.suppress_logit = nn.Parameter(torch.logit(torch.tensor(suppression_init, dtype=torch.float32)))

        self.last_alpha = None

    def forward(self, x):
        logits = self.net(x)
        reliability = torch.sigmoid(logits)
        alpha = torch.sigmoid(self.suppress_logit)
        factor = 1.0 - alpha * (1.0 - reliability)
        factor = torch.clamp(factor, 0.0, 1.0)
        self.last_alpha = alpha
        return logits, reliability, factor


class RGBDProjection(nn.Module):
    """Lightweight 1x1 adapter A([I, D]) for pseudo-RGB-D input."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(4, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.GELU(),
        )

    def forward(self, rgb, depth):
        return self.proj(torch.cat([rgb, depth], dim=1))


class FeatureProjector(nn.Module):
    """Lightweight channel projection P_i used before decoder fusion."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            BasicConv2d(in_channels, out_channels, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.proj(x)


class SGDB(nn.Module):
    """Semantic-guided Detail Block (SGDB).

    The shallow-detail representation queries high-level semantic features
    through cross-attention, followed by the context modulation branch used by
    SGDEM. This class is a structural refactor only; the original operations are
    preserved exactly for checkpoint compatibility.
    """

    def __init__(self, channels, num_heads=4, ffn_ratio=2):
        super().__init__()
        self.q_norm = nn.LayerNorm(channels)
        self.kv_norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        hidden_channels = channels * ffn_ratio
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, channels),
        )
        self.attn_proj = nn.Sequential(
            BasicConv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.semantic_region_gate = nn.Sequential(
            BasicConv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    @staticmethod
    def _flatten_tokens(x):
        return x.flatten(2).transpose(1, 2)

    @staticmethod
    def _unflatten_tokens(tokens, height, width):
        return tokens.transpose(1, 2).contiguous().view(
            tokens.shape[0], tokens.shape[2], height, width
        )

    def forward(self, detail, semantic):
        detail_low = F.adaptive_avg_pool2d(detail, output_size=semantic.shape[2:])
        _, _, hs, ws = semantic.shape

        q = self.q_norm(self._flatten_tokens(detail_low))
        kv = self.kv_norm(self._flatten_tokens(semantic))
        attn_tokens, _ = self.attn(q, kv, kv, need_weights=False)
        attn_tokens = attn_tokens + self.ffn(self.ffn_norm(attn_tokens))
        attn_map = self._unflatten_tokens(attn_tokens, hs, ws)

        attn_map = F.interpolate(attn_map, size=detail.shape[2:], mode='bilinear', align_corners=False)
        semantic_guided_detail = self.attn_proj(attn_map)

        semantic_up = F.interpolate(semantic, size=detail.shape[2:], mode='bilinear', align_corners=False)
        region_gate = self.semantic_region_gate(semantic_up)
        region_context = semantic_guided_detail * region_gate
        return semantic_guided_detail, region_gate, region_context


class DCUB(nn.Module):
    """Depth-Conditioned Update Block (DCUB), checkpoint-compatible form.

    This is the trained encoder-side update cell used by SGDEM. Its numerical
    operations are preserved so legacy checkpoints remain valid.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        in_channels = channels * 3 + 1  # semantic, detail, diff, depth geometry

        self.local_alpha = nn.Sequential(
            BasicConv2d(in_channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.local_update = nn.Sequential(
            BasicConv2d(in_channels, channels, kernel_size=1),
            nn.GELU(),
            DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.update_gate = nn.Sequential(
            BasicConv2d(in_channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        hidden = max(channels // reduction, 4)
        self.global_tau = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.depth_tau = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.refine = nn.Sequential(
            DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        self.last_depth_condition = None
        self.last_update_gate = None
        self.last_decay = None
        self.last_tau = None
        self.last_alpha = None
        self.last_liquid_state = None
        self.last_raw_delta = None
        self.last_refined_delta = None
        self.last_delta = None

    def forward(self, semantic, detail, depth_geometry):
        if semantic.shape[-2:] != detail.shape[-2:]:
            semantic = F.interpolate(semantic, size=detail.shape[-2:], mode='bilinear', align_corners=False)
        if depth_geometry.shape[-2:] != detail.shape[-2:]:
            depth_geometry = F.interpolate(depth_geometry, size=detail.shape[-2:], mode='bilinear', align_corners=False)

        depth_geometry = depth_geometry.clamp(0.0, 1.0)
        diff = torch.abs(semantic - detail)
        local_in = torch.cat([semantic, detail, diff, depth_geometry], dim=1)

        tau = 0.5 * (self.global_tau(semantic) + self.depth_tau(depth_geometry))
        alpha = F.softplus(self.local_alpha(local_in))
        decay = torch.exp(-alpha * tau)
        update_gate = self.update_gate(local_in)

        candidate = detail + torch.tanh(self.local_update(local_in))
        liquid_state = semantic * decay + candidate * (1.0 - decay)
        raw_delta = update_gate * (liquid_state - detail)
        delta = self.refine(raw_delta)

        self.last_depth_condition = depth_geometry.detach()
        self.last_update_gate = update_gate.detach()
        self.last_tau = tau.detach()
        self.last_alpha = alpha.detach()
        self.last_decay = decay.detach()
        self.last_liquid_state = liquid_state.detach()
        self.last_raw_delta = raw_delta.detach()
        self.last_delta = delta.detach()

        return delta


class GeometryCalibration(nn.Module):
    """Depth Geometry Prior (DGP).

    DGP is an explicit, non-liquid depth-geometry calibration path. It converts
    Sobel depth geometry into a bounded residual gate and uses that gate to
    modulate the local feature. The final gate conv is zero-initialized, so DGP
    starts as an identity mapping and learns only when the geometric cue is
    useful.
    """

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            BasicConv2d(channels + 1, channels, kernel_size=1),
            nn.GELU(),
            DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

        self.last_depth_geo = None
        self.last_gate_logits = None
        self.last_gate = None
        self.last_delta = None
        self.last_state = None

    def forward(self, feature, depth_geo):
        if depth_geo.shape[-2:] != feature.shape[-2:]:
            depth_geo = F.interpolate(depth_geo, size=feature.shape[-2:], mode='bilinear', align_corners=False)
        depth_geo = depth_geo.clamp(0.0, 1.0)
        gate_logits = self.gate(torch.cat([feature, depth_geo], dim=1))
        gate = torch.tanh(gate_logits)
        delta = gate * feature * depth_geo
        state = feature + delta

        self.last_depth_geo = depth_geo.detach()
        self.last_gate_logits = gate_logits.detach()
        self.last_gate = gate.detach()
        self.last_delta = delta.detach()
        self.last_state = state.detach()
        return state

class SGDEM(nn.Module):
    """Semantic-Geometric Detail Enhancement Module (SGDEM).

    Paper-facing name for the encoder-side shallow refinement module. The
    computation is kept identical to the trained GoodRT_v2 implementation.
    """

    def __init__(self, f1_channels, f2_channels, f4_channels, f5_channels,
                 out_channels=32, num_heads=4, ffn_ratio=2, eps=1e-6, ablation_mode="full"):
        super().__init__()
        self.ablation_mode = normalize_ablation_mode(ablation_mode)
        flags = ablation_flags(self.ablation_mode)
        self.use_dgp = flags["use_dgp"]
        self.use_dls = flags["use_dls"]
        self.use_idls = flags["use_idls"]
        self.use_rdls = flags["use_rdls"]
        self.use_spr = flags["use_spr"]
        self.eps = eps

        self.detail_f1 = BasicConv2d(
            f1_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        self.detail_f2 = BasicConv2d(
            f2_channels, out_channels, kernel_size=1
        )
        self.detail_fuse = nn.Sequential(
            BasicConv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.semantic_f4 = BasicConv2d(
            f4_channels, out_channels, kernel_size=1
        )
        self.semantic_f5 = BasicConv2d(
            f5_channels, out_channels, kernel_size=1
        )
        self.semantic_fuse = nn.Sequential(
            BasicConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.sgdb = SGDB(out_channels, num_heads=num_heads, ffn_ratio=ffn_ratio)
        self.geometry_calibration = GeometryCalibration(out_channels)

        self.out_proj = nn.Sequential(
            BasicConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Liquid core branch.  No extra hand-tuned scalar is introduced here:
        # semantic-detail fusion is performed by the liquid state update itself.
        self.dcub = DCUB(out_channels)
        self.liquid_proj = nn.Sequential(
            BasicConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0],
              [-2.0, 0.0, 2.0],
              [-1.0, 0.0, 1.0]]],
            dtype=torch.float32,
        ).unsqueeze(0)
        sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0],
              [ 0.0,  0.0,  0.0],
              [ 1.0,  2.0,  1.0]]],
            dtype=torch.float32,
        ).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def _depth_geometry(self, depth, size):
        kx = self.sobel_x.to(device=depth.device, dtype=depth.dtype)
        ky = self.sobel_y.to(device=depth.device, dtype=depth.dtype)

        gx = F.conv2d(depth, kx, padding=1)
        gy = F.conv2d(depth, ky, padding=1)
        grad = torch.sqrt(gx * gx + gy * gy + self.eps)
        grad = grad / (grad.amax(dim=(2, 3), keepdim=True) + self.eps)
        grad = F.interpolate(grad, size=size, mode='bilinear', align_corners=False)
        return grad

    @staticmethod
    def _liquid_energy_gate(x, eps=1e-6):
        """Non-learnable liquid modulation gate.

        The gate is computed from the channel-averaged absolute liquid response
        and normalized per image. It cannot invert the liquid response with
        a learnable Conv+BN gate.
        """
        energy = x.abs().mean(dim=1, keepdim=True)
        energy = energy / (energy.amax(dim=(2, 3), keepdim=True) + eps)
        return energy.clamp(0.0, 1.0)

    def forward(self, f1, f2, f4, f5, depth, return_aux=False):
        detail_f1 = self.detail_f1(f1)
        if detail_f1.shape[2:] != f2.shape[2:]:
            detail_f1 = F.interpolate(detail_f1, size=f2.shape[2:], mode='bilinear', align_corners=False)
        detail_f2 = self.detail_f2(f2)
        detail = self.detail_fuse(torch.cat([detail_f1, detail_f2], dim=1))

        semantic_f4 = self.semantic_f4(f4)
        semantic_f5 = self.semantic_f5(f5)
        semantic_f5 = F.interpolate(semantic_f5, size=semantic_f4.shape[2:], mode='bilinear', align_corners=False)
        semantic = self.semantic_fuse(semantic_f4 + semantic_f5)

        semantic_guided_detail, region_gate, region_context = self.sgdb(detail, semantic)

        depth_condition = self._depth_geometry(depth, size=detail.shape[2:])

        # DGP/DLS ablation logic for the bridge:
        #   baseline: attention/semantic bridge only;
        #   dgp: add an explicit depth-geometry prior, but no liquid transition;
        #   dgp_dls/full: run the original depth-guided liquid state bridge.
        liquid_core_delta = None
        liquid_delta = None
        liquid_gate = None
        liquid_core = None
        bridge_liquid_bypass = None

        if self.use_dls:
            liquid_core_delta = self.dcub(semantic_guided_detail, detail, depth_condition)
            liquid_delta = self.liquid_proj(liquid_core_delta)
            liquid_gate = self._liquid_energy_gate(liquid_delta)
            liquid_core = detail + liquid_gate * liquid_delta
            core_refined = liquid_core + region_context
            bridge_projected_delta = self.out_proj(core_refined)
            bridge_liquid_bypass = liquid_gate * liquid_core_delta
            delta_f = bridge_projected_delta + bridge_liquid_bypass
        elif self.use_dgp:
            dgp_detail = self.geometry_calibration(detail, depth_condition)
            core_refined = dgp_detail + region_context
            bridge_projected_delta = self.out_proj(core_refined)
            delta_f = bridge_projected_delta
        else:
            core_refined = detail + region_context
            bridge_projected_delta = self.out_proj(core_refined)
            delta_f = bridge_projected_delta

        if return_aux:
            def _detach_or_none(x):
                return x.detach() if x is not None else None

            aux = {
                "ablation_mode": self.ablation_mode,
                "use_dgp": self.use_dgp,
                "use_dls": self.use_dls,
                "use_idls": self.use_idls,
                "use_rdls": self.use_rdls,
                "use_spr": self.use_spr,
                "detail_feature": detail.detach(),
                "semantic_feature": semantic.detach(),
                "semantic_guided_detail": semantic_guided_detail.detach(),
                "depth_condition": depth_condition.detach(),
                "dgp_gate": self.geometry_calibration.last_gate.detach() if self.geometry_calibration.last_gate is not None else None,
                "dgp_delta": self.geometry_calibration.last_delta.detach() if self.geometry_calibration.last_delta is not None else None,
                "dgp_state": self.geometry_calibration.last_state.detach() if self.geometry_calibration.last_state is not None else None,
                "semantic_region_gate": region_gate.detach(),
                "region_context": region_context.detach(),
                "liquid_core_delta": _detach_or_none(liquid_core_delta),
                "liquid_delta": _detach_or_none(liquid_delta),
                "liquid_gate": _detach_or_none(liquid_gate),
                "liquid_core": _detach_or_none(liquid_core),
                "bridge_projected_delta": bridge_projected_delta.detach(),
                "bridge_liquid_bypass": _detach_or_none(bridge_liquid_bypass),
                "liquid_update_gate": self.dcub.last_update_gate.detach() if self.dcub.last_update_gate is not None else None,
                "liquid_depth_condition": self.dcub.last_depth_condition.detach() if self.dcub.last_depth_condition is not None else None,
                "liquid_decay": self.dcub.last_decay.detach() if self.dcub.last_decay is not None else None,
                "liquid_state": self.dcub.last_liquid_state.detach() if self.dcub.last_liquid_state is not None else None,
                "liquid_raw_delta": self.dcub.last_raw_delta.detach() if self.dcub.last_raw_delta is not None else None,
                "sgdem_residual": delta_f.detach(),
            }
            return delta_f, aux

        return delta_f


class DecoderRefinementBlock(nn.Module):
    """Identity-preserving liquid update for an intermediate decoder scale.

    The module updates a detail/top-down decoder feature using the higher-level
    semantic feature and a depth-geometry condition.  The residual refinement is
    zero-initialized, so the decoder initially behaves like the original
    top-down pathway and learns multi-scale liquid updates during training.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        in_channels = channels * 3 + 1  # semantic, detail, diff, depth_geo

        self.semantic_proj = BasicConv2d(channels, channels, kernel_size=1)
        self.detail_proj = BasicConv2d(channels, channels, kernel_size=1)

        self.local_alpha = nn.Sequential(
            BasicConv2d(in_channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.local_update = nn.Sequential(
            BasicConv2d(in_channels, channels, kernel_size=1),
            nn.GELU(),
            DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.update_gate = nn.Sequential(
            BasicConv2d(in_channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        hidden = max(channels // reduction, 4)
        self.global_tau = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.depth_tau = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.refine = nn.Sequential(
            DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        # Identity-preserving residual initialization for the added p4/p3 updates.
        nn.init.zeros_(self.refine[1].weight)
        nn.init.zeros_(self.refine[1].bias)

        # Diagnostics
        self.last_depth_geo = None
        self.last_learned_update_gate = None
        self.last_depth_gate = None
        self.last_update_gate = None
        self.last_decay = None
        self.last_alpha = None
        self.last_tau = None
        self.last_raw_delta = None
        self.last_delta = None

    def forward(self, detail, semantic, depth_geo):
        if semantic.shape[-2:] != detail.shape[-2:]:
            semantic = F.interpolate(semantic, size=detail.shape[-2:], mode='bilinear', align_corners=False)
        if depth_geo.shape[-2:] != detail.shape[-2:]:
            depth_geo = F.interpolate(depth_geo, size=detail.shape[-2:], mode='bilinear', align_corners=False)

        semantic = self.semantic_proj(semantic)
        detail_state = self.detail_proj(detail)
        depth_geo = depth_geo.clamp(0.0, 1.0)
        diff = torch.abs(semantic - detail_state)
        local_in = torch.cat([semantic, detail_state, diff, depth_geo], dim=1)

        tau = 0.5 * (self.global_tau(semantic) + self.depth_tau(depth_geo))
        alpha = F.softplus(self.local_alpha(local_in))
        decay = torch.exp(-alpha * tau)
        learned_update_gate = self.update_gate(local_in)

        # Depth is already part of the learned liquid dynamics.  We do not add a
        # hard max lower bound here because previous experiments showed that raw
        # max gates can be numerically ineffective or unstable.
        depth_gate = depth_geo
        update_gate = learned_update_gate

        candidate = detail_state + torch.tanh(self.local_update(local_in))
        liquid_state = semantic * decay + candidate * (1.0 - decay)
        raw_delta = update_gate * (liquid_state - detail_state)
        delta = self.refine(raw_delta)

        self.last_depth_geo = depth_geo.detach()
        self.last_learned_update_gate = learned_update_gate.detach()
        self.last_depth_gate = depth_gate.detach()
        self.last_update_gate = update_gate.detach()
        self.last_tau = tau.detach()
        self.last_alpha = alpha.detach()
        self.last_decay = decay.detach()
        self.last_raw_delta = raw_delta.detach()
        self.last_delta = delta.detach()

        return detail + delta


class DepthGeometryExtractor(nn.Module):
    """Non-learnable Sobel geometry from depth."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0],
              [-2.0, 0.0, 2.0],
              [-1.0, 0.0, 1.0]]],
            dtype=torch.float32,
        ).unsqueeze(0)
        sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0],
              [ 0.0,  0.0,  0.0],
              [ 1.0,  2.0,  1.0]]],
            dtype=torch.float32,
        ).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def forward(self, depth, size=None):
        kx = self.sobel_x.to(device=depth.device, dtype=depth.dtype)
        ky = self.sobel_y.to(device=depth.device, dtype=depth.dtype)
        gx = F.conv2d(depth, kx, padding=1)
        gy = F.conv2d(depth, ky, padding=1)
        grad = torch.sqrt(gx * gx + gy * gy + self.eps)
        grad = grad / (grad.amax(dim=(2, 3), keepdim=True) + self.eps)
        if size is not None and grad.shape[-2:] != size:
            grad = F.interpolate(grad, size=size, mode='bilinear', align_corners=False)
        return grad.clamp(0.0, 1.0)


class TGMM(nn.Module):
    """Target-guided Geometry Modulation Module (TGMM), trained implementation.

    The class is the paper-facing name for the decoder-stage target-conditioned
    geometry update in the provided checkpoint-compatible implementation.
    Tensor operations are intentionally unchanged from GoodRT_v2.
    """

    def __init__(self, channels, reduction=16, suppression_init=0.20, steps=1):
        super().__init__()
        in_channels = channels * 4 + 1  # original branch: semantic, detail, prev_state, diff, depth_geo
        state_channels = channels * 4    # GoodRT content branch: semantic, detail, prev_state, diff
        control_channels = channels * 2 + 2  # GoodRT control branch: diff_md, diff_dp, target_depth_geo, target_support
        self.steps = max(1, int(steps))

        self.semantic_proj = BasicConv2d(channels, channels, kernel_size=1)
        self.detail_proj = BasicConv2d(channels, channels, kernel_size=1)
        self.prev_proj = BasicConv2d(channels, channels, kernel_size=1)
        self.memory_fuse = BasicConv2d(channels * 2, channels, kernel_size=1)

        self.local_alpha = nn.Sequential(
            BasicConv2d(in_channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.local_update = nn.Sequential(
            BasicConv2d(in_channels, channels, kernel_size=1),
            nn.GELU(),
            DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.update_gate = nn.Sequential(
            BasicConv2d(in_channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # GoodRT target-conditioned decoupled liquid cell.
        # Candidate liquid content is generated from RGB semantic/detail states,
        # while target-conditioned depth only controls liquid dynamics.
        prior_hidden = max(channels // 4, 8)
        self.target_prior_head = nn.Sequential(
            BasicConv2d(channels, prior_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(prior_hidden, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.target_prior_head[-1].weight)
        nn.init.zeros_(self.target_prior_head[-1].bias)

        self.state_update = nn.Sequential(
            BasicConv2d(state_channels, channels, kernel_size=1),
            nn.GELU(),
            DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.control_alpha = nn.Sequential(
            BasicConv2d(control_channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.control_update_gate = nn.Sequential(
            BasicConv2d(control_channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        hidden = max(channels // reduction, 4)
        self.global_tau = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.depth_tau = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.refine = nn.Sequential(
            DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        # Suppress-only R-DLS reliability. This is DLS-only:
        # bridge liquid, MS refinement and LDR are not reliability-routed.
        self.reliability_suppressor = ReliabilitySuppressor(
            in_channels,
            max(channels // 2, 8),
            suppression_init=suppression_init,
            reliability_bias=2.0,
        )
        self.control_reliability_suppressor = ReliabilitySuppressor(
            control_channels,
            max(channels // 2, 8),
            suppression_init=suppression_init,
            reliability_bias=2.0,
        )

        # Identity-preserving start: DLS initially leaves the decoder feature
        # unchanged, then learns depth-conditioned liquid restorations.
        nn.init.zeros_(self.refine[1].weight)
        nn.init.zeros_(self.refine[1].bias)

        # Diagnostics
        self.last_depth_geo = None
        self.last_target_prior_logits = None
        self.last_target_prior = None
        self.last_target_uncertainty = None
        self.last_target_support = None
        self.last_target_depth_geo = None
        self.last_control_in = None
        self.last_memory_state = None
        self.last_prev_state = None
        self.last_learned_update_gate = None
        self.last_reliability_logits = None
        self.last_reliability = None
        self.last_reliability_factor = None
        self.last_suppress_alpha = None
        self.last_iter_steps = None
        self.last_update_gate = None
        self.last_decay = None
        self.last_alpha = None
        self.last_tau = None
        self.last_raw_delta = None
        self.last_delta = None
        self.last_state = None

    def forward(self, detail, semantic, depth_geo, prev_state=None,
                reliability_enabled=False, iterative_enabled=False):
        if semantic.shape[-2:] != detail.shape[-2:]:
            semantic = F.interpolate(semantic, size=detail.shape[-2:], mode='bilinear', align_corners=False)
        if depth_geo.shape[-2:] != detail.shape[-2:]:
            depth_geo = F.interpolate(depth_geo, size=detail.shape[-2:], mode='bilinear', align_corners=False)
        if prev_state is None:
            prev_state = semantic
        elif prev_state.shape[-2:] != detail.shape[-2:]:
            prev_state = F.interpolate(prev_state, size=detail.shape[-2:], mode='bilinear', align_corners=False)

        semantic_state = self.semantic_proj(semantic)
        prev_state = self.prev_proj(prev_state)
        memory_state = self.memory_fuse(torch.cat([semantic_state, prev_state], dim=1))
        depth_geo = depth_geo.clamp(0.0, 1.0)

        # Ordinary DLS keeps the original single-step update.
        # I-DLS / GoodR / Full use scale-specific iterative state evolution.
        iter_steps = self.steps if iterative_enabled else 1
        state = detail
        step_scale = 1.0 / float(iter_steps)

        last_detail_state = None
        last_target_prior_logits = None
        last_target_prior = None
        last_target_uncertainty = None
        last_target_support = None
        last_target_depth_geo = None
        last_control_in = None
        last_learned_update_gate = None
        last_rel_logits = None
        last_reliability = None
        last_rel_factor = None
        last_suppress_alpha = None
        last_update_gate = None
        last_tau = None
        last_alpha = None
        last_decay = None
        last_raw_delta = None
        last_delta = None

        # GoodRT is enabled for iterative/R-DLS modes. Ordinary dgp_dls keeps
        # the original single-step DLS branch for clean comparison.
        target_conditioned_enabled = iterative_enabled or reliability_enabled

        for _ in range(iter_steps):
            detail_state = self.detail_proj(state)
            diff = torch.abs(memory_state - detail_state)

            if target_conditioned_enabled:
                # GoodRT_v2 foreground-anchored target support.
                # GoodRT used support = p + 4p(1-p), which opens all uncertain
                # regions and may activate background depth distractions.
                # Here uncertainty is anchored by foreground probability:
                #   support = p + p(1-p) = p(2-p)
                # Weak background responses therefore cannot easily open depth
                # stimulus, while true target/boundary regions remain supported.
                target_prior_logits = self.target_prior_head(memory_state)
                target_prior = torch.sigmoid(target_prior_logits)
                target_uncertainty = target_prior * (1.0 - target_prior)
                target_support = torch.clamp(target_prior + target_uncertainty, 0.0, 1.0)

                # Detach support for liquid control so the segmentation loss does
                # not directly push the prior head to open background shortcuts.
                # The prior head is still supervised by the existing auxiliary
                # control loss in train.py.
                target_support_ctrl = target_support.detach()
                target_depth_geo = depth_geo * target_support_ctrl

                state_in = torch.cat([memory_state, detail_state, prev_state, diff], dim=1)
                diff_prev = torch.abs(detail_state - prev_state)
                control_in = torch.cat([diff, diff_prev, target_depth_geo, target_support_ctrl], dim=1)

                tau = 0.5 * (self.global_tau(memory_state) + self.depth_tau(target_depth_geo))
                alpha = F.softplus(self.control_alpha(control_in))
                decay = torch.exp(-alpha * tau)

                learned_update_gate = self.control_update_gate(control_in)
                rel_logits, reliability, rel_factor = self.control_reliability_suppressor(control_in)
                update_gate = learned_update_gate * rel_factor if reliability_enabled else learned_update_gate

                candidate = detail_state + torch.tanh(self.state_update(state_in))
                last_suppress_alpha = self.control_reliability_suppressor.last_alpha
            else:
                # Original DLS branch.
                local_in = torch.cat([memory_state, detail_state, prev_state, diff, depth_geo], dim=1)
                target_prior_logits = depth_geo.new_zeros(depth_geo.shape)
                target_prior = torch.sigmoid(target_prior_logits)
                target_uncertainty = 4.0 * target_prior * (1.0 - target_prior)
                target_support = torch.ones_like(depth_geo)
                target_depth_geo = depth_geo
                control_in = local_in

                tau = 0.5 * (self.global_tau(memory_state) + self.depth_tau(depth_geo))
                alpha = F.softplus(self.local_alpha(local_in))
                decay = torch.exp(-alpha * tau)

                learned_update_gate = self.update_gate(local_in)
                rel_logits, reliability, rel_factor = self.reliability_suppressor(local_in)
                update_gate = learned_update_gate * rel_factor if reliability_enabled else learned_update_gate

                candidate = detail_state + torch.tanh(self.local_update(local_in))
                last_suppress_alpha = self.reliability_suppressor.last_alpha

            liquid_state = memory_state * decay + candidate * (1.0 - decay)
            raw_delta = update_gate * (liquid_state - detail_state)
            delta = self.refine(raw_delta)

            # Shared-parameter iterative liquid evolution. The 1/K step scale
            # prevents multi-step updates from simply becoming a larger residual.
            state = state + step_scale * delta

            last_detail_state = detail_state
            last_target_prior_logits = target_prior_logits
            last_target_prior = target_prior
            last_target_uncertainty = target_uncertainty
            last_target_support = target_support
            last_target_depth_geo = target_depth_geo
            last_control_in = control_in
            last_learned_update_gate = learned_update_gate
            last_rel_logits = rel_logits
            last_reliability = reliability
            last_rel_factor = rel_factor
            last_update_gate = update_gate
            last_tau = tau
            last_alpha = alpha
            last_decay = decay
            last_raw_delta = raw_delta
            last_delta = delta

        self.last_depth_geo = depth_geo.detach()
        self.last_target_prior_logits = last_target_prior_logits
        self.last_target_prior = last_target_prior.detach()
        self.last_target_uncertainty = last_target_uncertainty.detach()
        self.last_target_support = last_target_support.detach()
        self.last_target_depth_geo = last_target_depth_geo.detach()
        self.last_control_in = last_control_in.detach()
        self.last_memory_state = memory_state.detach()
        self.last_prev_state = prev_state.detach()
        self.last_learned_update_gate = last_learned_update_gate.detach()
        self.last_reliability_logits = last_rel_logits
        self.last_reliability = last_reliability.detach()
        self.last_reliability_factor = last_rel_factor.detach()
        self.last_suppress_alpha = last_suppress_alpha.detach()
        self.last_iter_steps = torch.tensor(float(iter_steps), device=state.device, dtype=state.dtype)
        self.last_update_gate = last_update_gate.detach()
        self.last_tau = last_tau.detach()
        self.last_alpha = last_alpha.detach()
        self.last_decay = last_decay.detach()
        self.last_raw_delta = last_raw_delta.detach()
        self.last_delta = last_delta.detach()
        self.last_state = state.detach()

        return state


class DecoderStateReadout(nn.Module):
    """LDR-anchored DLS state restoration.

    The final LDR output is treated as the semantic/shape anchor. The DLS state
    provides a depth-guided restoration target. The learned SPR gate decides
    where to pull the LDR readout back toward the DLS state. The gate is
    zero-initialized through the last 1x1 convolution, so the initial readout is
    exactly the original LiquidCore_DLS final LDR state.
    """

    def __init__(self, channels):
        super().__init__()
        in_channels = channels * 3 + 2  # dls_state, ldr_state, |diff|, depth_geo, edge_prob
        self.gate = nn.Sequential(
            BasicConv2d(in_channels, channels, kernel_size=1),
            nn.GELU(),
            DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

        # Diagnostics
        self.last_gate_logits = None
        self.last_gate = None
        self.last_restore = None
        self.last_delta = None
        self.last_state = None

    def forward(self, dls_state, ldr_state, depth_geo, edge_prob):
        if depth_geo.shape[-2:] != dls_state.shape[-2:]:
            depth_geo = F.interpolate(depth_geo, size=dls_state.shape[-2:], mode='bilinear', align_corners=False)
        if edge_prob.shape[-2:] != dls_state.shape[-2:]:
            edge_prob = F.interpolate(edge_prob, size=dls_state.shape[-2:], mode='bilinear', align_corners=False)

        depth_geo = depth_geo.clamp(0.0, 1.0)
        edge_prob = edge_prob.clamp(0.0, 1.0)
        restore = dls_state - ldr_state
        diff = torch.abs(restore)
        local_in = torch.cat([dls_state, ldr_state, diff, depth_geo, edge_prob], dim=1)

        gate_logits = self.gate(local_in)
        # tanh gives zero restoration at initialization and bounded learned
        # restoration strength without adding a hand-tuned scalar.
        spr_gate = torch.tanh(gate_logits)
        delta = spr_gate * restore
        state = ldr_state + delta

        self.last_gate_logits = gate_logits.detach()
        self.last_gate = spr_gate.detach()
        self.last_restore = restore.detach()
        self.last_delta = delta.detach()
        self.last_state = state.detach()

        return state


class GCDUB(nn.Module):
    """LiquidCore_DLS_SPR_V2 decoder update.

    The final p2 feature is directly updated by a depth/edge-conditioned liquid
    transition.  This is no longer a zero-gamma residual restoration.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        in_channels = channels * 3 + 2  # semantic, detail, diff, depth_geo, edge_prob

        self.semantic_proj = BasicConv2d(channels, channels, kernel_size=1)
        self.detail_proj = BasicConv2d(channels, channels, kernel_size=1)

        self.local_alpha = nn.Sequential(
            BasicConv2d(in_channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.local_update = nn.Sequential(
            BasicConv2d(in_channels, channels, kernel_size=1),
            nn.GELU(),
            DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.update_gate = nn.Sequential(
            BasicConv2d(in_channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        hidden = max(channels // reduction, 4)
        self.global_tau = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.cue_tau = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.refine = nn.Sequential(
            DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        # Diagnostics
        self.last_cue = None
        self.last_learned_update_gate = None
        self.last_cue_gate_raw = None
        self.last_cue_gate = None
        self.last_update_gate = None
        self.last_decay = None
        self.last_alpha = None
        self.last_tau = None
        self.last_raw_delta = None
        self.last_refined_delta = None
        self.last_delta = None

    def forward(self, p2, p3, depth_geo, edge_prob):
        if p3.shape[-2:] != p2.shape[-2:]:
            p3 = F.interpolate(p3, size=p2.shape[-2:], mode='bilinear', align_corners=False)
        if depth_geo.shape[-2:] != p2.shape[-2:]:
            depth_geo = F.interpolate(depth_geo, size=p2.shape[-2:], mode='bilinear', align_corners=False)
        if edge_prob.shape[-2:] != p2.shape[-2:]:
            edge_prob = F.interpolate(edge_prob, size=p2.shape[-2:], mode='bilinear', align_corners=False)

        semantic = self.semantic_proj(p3)
        detail = self.detail_proj(p2)
        depth_geo = depth_geo.clamp(0.0, 1.0)
        edge_prob = edge_prob.clamp(0.0, 1.0)
        cue = torch.cat([depth_geo, edge_prob], dim=1)

        diff = torch.abs(semantic - detail)
        local_in = torch.cat([semantic, detail, diff, cue], dim=1)

        tau = 0.5 * (self.global_tau(semantic) + self.cue_tau(cue))
        alpha = F.softplus(self.local_alpha(local_in))
        decay = torch.exp(-alpha * tau)
        learned_update_gate = self.update_gate(local_in)
        cue_gate_raw = torch.maximum(depth_geo, edge_prob)
        cue_gate = cue_gate_raw / (cue_gate_raw.amax(dim=(2, 3), keepdim=True) + 1e-6)
        update_gate = torch.maximum(learned_update_gate, cue_gate)

        candidate = detail + torch.tanh(self.local_update(local_in))
        liquid_state = semantic * decay + candidate * (1.0 - decay)
        raw_delta = update_gate * (liquid_state - detail)
        refined_delta = self.refine(raw_delta)

        # V2: preserve the raw liquid state update after refinement.  The raw
        # delta carried stronger boundary selectivity in LiquidCore_MS analysis,
        # while refine(raw_delta) could wash it out.
        delta = raw_delta + refined_delta

        self.last_cue = cue.detach()
        self.last_learned_update_gate = learned_update_gate.detach()
        self.last_cue_gate_raw = cue_gate_raw.detach()
        self.last_cue_gate = cue_gate.detach()
        self.last_update_gate = update_gate.detach()
        self.last_decay = decay.detach()
        self.last_alpha = alpha.detach()
        self.last_tau = tau.detach()
        self.last_raw_delta = raw_delta.detach()
        self.last_refined_delta = refined_delta.detach()
        self.last_delta = delta.detach()

        return p2 + delta


class RTSRDecoder(nn.Module):
    """RTSRNet top-down decoder with multi-stage target-sensitive refinement."""

    def __init__(self, channels=(32, 64, 96, 160), fpn_channels=64, use_edge_refine=True, ablation_mode="full"):
        super().__init__()
        self.ablation_mode = normalize_ablation_mode(ablation_mode)
        flags = ablation_flags(self.ablation_mode)
        self.use_dgp = flags["use_dgp"]
        self.use_dls = flags["use_dls"]
        self.use_idls = flags["use_idls"]
        self.use_rdls = flags["use_rdls"]
        self.use_spr = flags["use_spr"]
        c1, c2, c3, c4 = channels
        self.use_edge_refine = use_edge_refine
        self.l4 = BasicConv2d(c4, fpn_channels, kernel_size=1)
        self.l3 = BasicConv2d(c3, fpn_channels, kernel_size=1)
        self.l2 = BasicConv2d(c2, fpn_channels, kernel_size=1)
        self.l1 = BasicConv2d(c1, fpn_channels, kernel_size=1)
        self.s4 = DepthwiseSeparableConv(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.s3 = DepthwiseSeparableConv(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.s2 = DepthwiseSeparableConv(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.s1 = DepthwiseSeparableConv(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.edge_proj = nn.Sequential(
            nn.Conv2d(1, fpn_channels, 1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.GELU(),
        )
        self.refine1 = nn.Sequential(
            DepthwiseSeparableConv(fpn_channels, fpn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fpn_channels),
            nn.GELU(),
        )
        self.pred_head2 = nn.Conv2d(fpn_channels, 1, 1)
        self.pred_head3 = nn.Conv2d(fpn_channels, 1, 1)
        self.pred_head4 = nn.Conv2d(fpn_channels, 1, 1)
        self.pred_head5 = nn.Conv2d(fpn_channels, 1, 1)
        self.edge_head = nn.Sequential(
            DepthwiseSeparableConv(fpn_channels, fpn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fpn_channels),
            nn.GELU(),
            nn.Conv2d(fpn_channels, 1, 1),
        )

        # DGP, DLS, and SPR branches. They are always instantiated so checkpoints
        # share one stable state_dict; the ablation switch controls which branches
        # are active in forward().
        self.geometry_extractor = DepthGeometryExtractor()
        self.geometry_calibration4 = GeometryCalibration(fpn_channels)
        self.geometry_calibration3 = GeometryCalibration(fpn_channels)
        self.geometry_calibration2 = GeometryCalibration(fpn_channels)
        # p4 is high-level/coarse and more risky for false positives, so its
        # suppress strength starts weaker. p3/p2 carry boundary restoration.
        self.tgmm4 = TGMM(fpn_channels, suppression_init=0.05, steps=1)
        self.tgmm3 = TGMM(fpn_channels, suppression_init=0.20, steps=2)
        self.tgmm2 = TGMM(fpn_channels, suppression_init=0.20, steps=2)
        self.decoder_refine4 = DecoderRefinementBlock(fpn_channels)
        self.decoder_refine3 = DecoderRefinementBlock(fpn_channels)
        # Keep the final p2 depth/edge-conditioned LiquidCore update, now fed by DLS state.
        self.gcdub2 = GCDUB(fpn_channels)
        # State-Preserving Readout: constrain final LDR as a restoration to DLS state.
        self.state_readout2 = DecoderStateReadout(fpn_channels)

    def forward(self, feats, rect_edge, out_size, depth=None, return_aux=False):
        f2, f3, f4, f5 = feats

        p5 = self.s4(self.l4(f5))

        p4_base = self.s3(self.l3(f4) + F.interpolate(p5, size=f4.shape[2:], mode='bilinear', align_corners=False))
        if depth is None:
            depth_geo_p4 = torch.zeros(p4_base.shape[0], 1, p4_base.shape[2], p4_base.shape[3],
                                       device=p4_base.device, dtype=p4_base.dtype)
        else:
            depth_geo_p4 = self.geometry_extractor(depth, size=p4_base.shape[2:])
        if self.use_dgp:
            p4_seed = self.geometry_calibration4(p4_base, depth_geo_p4)
        else:
            p4_seed = p4_base
        if self.use_dls:
            tgmm4 = self.tgmm4(p4_seed, p5, depth_geo_p4, prev_state=None,
                                  reliability_enabled=self.use_rdls,
                                  iterative_enabled=self.use_idls)
            p4 = self.decoder_refine4(tgmm4, p5, depth_geo_p4)
        else:
            tgmm4 = p4_seed
            p4 = p4_seed

        p3_base = self.s2(self.l2(f3) + F.interpolate(p4, size=f3.shape[2:], mode='bilinear', align_corners=False))
        if depth is None:
            depth_geo_p3 = torch.zeros(p3_base.shape[0], 1, p3_base.shape[2], p3_base.shape[3],
                                       device=p3_base.device, dtype=p3_base.dtype)
        else:
            depth_geo_p3 = self.geometry_extractor(depth, size=p3_base.shape[2:])
        if self.use_dgp:
            p3_seed = self.geometry_calibration3(p3_base, depth_geo_p3)
        else:
            p3_seed = p3_base
        if self.use_dls:
            tgmm3 = self.tgmm3(p3_seed, p4, depth_geo_p3, prev_state=tgmm4,
                                  reliability_enabled=self.use_rdls,
                                  iterative_enabled=self.use_idls)
            p3 = self.decoder_refine3(tgmm3, p4, depth_geo_p3)
        else:
            tgmm3 = p3_seed
            p3 = p3_seed

        p2_topdown = self.s1(self.l1(f2) + F.interpolate(p3, size=f2.shape[2:], mode='bilinear', align_corners=False))
        edge1 = F.interpolate(rect_edge, size=p2_topdown.shape[2:], mode='bilinear', align_corners=False)
        if self.use_edge_refine:
            p2_base = self.refine1(p2_topdown + self.edge_proj(edge1))
        else:
            p2_base = self.refine1(p2_topdown)

        edge_low = self.edge_head(p2_base)

        if depth is None:
            depth_geo = torch.zeros(p2_base.shape[0], 1, p2_base.shape[2], p2_base.shape[3],
                                    device=p2_base.device, dtype=p2_base.dtype)
        else:
            depth_geo = self.geometry_extractor(depth, size=p2_base.shape[2:])

        edge_prob_low = torch.sigmoid(edge_low.detach())
        if self.use_dgp:
            p2_seed = self.geometry_calibration2(p2_base, depth_geo)
        else:
            p2_seed = p2_base
        if self.use_dls:
            tgmm2 = self.tgmm2(p2_seed, p3, depth_geo, prev_state=tgmm3,
                                  reliability_enabled=self.use_rdls,
                                  iterative_enabled=self.use_idls)
            p2_ldr = self.gcdub2(tgmm2, p3, depth_geo, edge_prob_low)
            if self.use_spr:
                p2 = self.state_readout2(tgmm2, p2_ldr, depth_geo, edge_prob_low)
            else:
                p2 = p2_ldr
        else:
            tgmm2 = p2_seed
            p2_ldr = p2_seed
            p2 = p2_seed

        m1 = F.interpolate(self.pred_head2(p2), size=out_size, mode='bilinear', align_corners=False)
        m2 = F.interpolate(self.pred_head3(p3), size=out_size, mode='bilinear', align_corners=False)
        m3 = F.interpolate(self.pred_head4(p4), size=out_size, mode='bilinear', align_corners=False)
        m4 = F.interpolate(self.pred_head5(p5), size=out_size, mode='bilinear', align_corners=False)
        edge = F.interpolate(edge_low, size=out_size, mode='bilinear', align_corners=False)

        if return_aux:
            aux = {
                "ablation_mode": self.ablation_mode,
                "use_dgp": self.use_dgp,
                "use_dls": self.use_dls,
                "use_idls": self.use_idls,
                "use_rdls": self.use_rdls,
                "use_spr": self.use_spr,
                "decoder_p4_base": p4_base.detach(),
                "decoder_p4_dgp_state": self.geometry_calibration4.last_state.detach() if self.geometry_calibration4.last_state is not None else None,
                "decoder_p4_dgp_gate": self.geometry_calibration4.last_gate.detach() if self.geometry_calibration4.last_gate is not None else None,
                "decoder_p4_dgp_delta": self.geometry_calibration4.last_delta.detach() if self.geometry_calibration4.last_delta is not None else None,
                "decoder_p4_dls_state": tgmm4.detach(),
                "decoder_p4_refined": p4.detach(),
                "tgmm4_depth_geo": self.tgmm4.last_depth_geo.detach() if self.tgmm4.last_depth_geo is not None else None,
                "tgmm4_target_prior": self.tgmm4.last_target_prior.detach() if self.tgmm4.last_target_prior is not None else None,
                "tgmm4_target_support": self.tgmm4.last_target_support.detach() if self.tgmm4.last_target_support is not None else None,
                "tgmm4_target_depth_geo": self.tgmm4.last_target_depth_geo.detach() if self.tgmm4.last_target_depth_geo is not None else None,
                "tgmm4_memory_state": self.tgmm4.last_memory_state.detach() if self.tgmm4.last_memory_state is not None else None,
                "tgmm4_prev_state": self.tgmm4.last_prev_state.detach() if self.tgmm4.last_prev_state is not None else None,
                "tgmm4_learned_update_gate": self.tgmm4.last_learned_update_gate.detach() if self.tgmm4.last_learned_update_gate is not None else None,
                "tgmm4_reliability": self.tgmm4.last_reliability.detach() if self.tgmm4.last_reliability is not None else None,
                "tgmm4_reliability_factor": self.tgmm4.last_reliability_factor.detach() if self.tgmm4.last_reliability_factor is not None else None,
                "tgmm4_suppress_alpha": self.tgmm4.last_suppress_alpha.detach().view(1, 1, 1, 1) if self.tgmm4.last_suppress_alpha is not None else None,
                "tgmm4_iter_steps": self.tgmm4.last_iter_steps.detach().view(1, 1, 1, 1) if self.tgmm4.last_iter_steps is not None else None,
                "tgmm4_update_gate": self.tgmm4.last_update_gate.detach() if self.tgmm4.last_update_gate is not None else None,
                "tgmm4_decay": self.tgmm4.last_decay.detach() if self.tgmm4.last_decay is not None else None,
                "tgmm4_raw_delta": self.tgmm4.last_raw_delta.detach() if self.tgmm4.last_raw_delta is not None else None,
                "tgmm4_delta": self.tgmm4.last_delta.detach() if self.tgmm4.last_delta is not None else None,
                "ms_p4_depth_geo": depth_geo_p4.detach(),
                "ms_p4_learned_update_gate": self.decoder_refine4.last_learned_update_gate.detach() if self.decoder_refine4.last_learned_update_gate is not None else None,
                "ms_p4_depth_gate": self.decoder_refine4.last_depth_gate.detach() if self.decoder_refine4.last_depth_gate is not None else None,
                "ms_p4_update_gate": self.decoder_refine4.last_update_gate.detach() if self.decoder_refine4.last_update_gate is not None else None,
                "ms_p4_decay": self.decoder_refine4.last_decay.detach() if self.decoder_refine4.last_decay is not None else None,
                "ms_p4_raw_delta": self.decoder_refine4.last_raw_delta.detach() if self.decoder_refine4.last_raw_delta is not None else None,
                "ms_p4_delta": self.decoder_refine4.last_delta.detach() if self.decoder_refine4.last_delta is not None else None,

                "decoder_p3_base": p3_base.detach(),
                "decoder_p3_dgp_state": self.geometry_calibration3.last_state.detach() if self.geometry_calibration3.last_state is not None else None,
                "decoder_p3_dgp_gate": self.geometry_calibration3.last_gate.detach() if self.geometry_calibration3.last_gate is not None else None,
                "decoder_p3_dgp_delta": self.geometry_calibration3.last_delta.detach() if self.geometry_calibration3.last_delta is not None else None,
                "decoder_p3_dls_state": tgmm3.detach(),
                "decoder_p3_refined": p3.detach(),
                "tgmm3_depth_geo": self.tgmm3.last_depth_geo.detach() if self.tgmm3.last_depth_geo is not None else None,
                "tgmm3_target_prior": self.tgmm3.last_target_prior.detach() if self.tgmm3.last_target_prior is not None else None,
                "tgmm3_target_support": self.tgmm3.last_target_support.detach() if self.tgmm3.last_target_support is not None else None,
                "tgmm3_target_depth_geo": self.tgmm3.last_target_depth_geo.detach() if self.tgmm3.last_target_depth_geo is not None else None,
                "tgmm3_memory_state": self.tgmm3.last_memory_state.detach() if self.tgmm3.last_memory_state is not None else None,
                "tgmm3_prev_state": self.tgmm3.last_prev_state.detach() if self.tgmm3.last_prev_state is not None else None,
                "tgmm3_learned_update_gate": self.tgmm3.last_learned_update_gate.detach() if self.tgmm3.last_learned_update_gate is not None else None,
                "tgmm3_reliability": self.tgmm3.last_reliability.detach() if self.tgmm3.last_reliability is not None else None,
                "tgmm3_reliability_factor": self.tgmm3.last_reliability_factor.detach() if self.tgmm3.last_reliability_factor is not None else None,
                "tgmm3_suppress_alpha": self.tgmm3.last_suppress_alpha.detach().view(1, 1, 1, 1) if self.tgmm3.last_suppress_alpha is not None else None,
                "tgmm3_iter_steps": self.tgmm3.last_iter_steps.detach().view(1, 1, 1, 1) if self.tgmm3.last_iter_steps is not None else None,
                "tgmm3_update_gate": self.tgmm3.last_update_gate.detach() if self.tgmm3.last_update_gate is not None else None,
                "tgmm3_decay": self.tgmm3.last_decay.detach() if self.tgmm3.last_decay is not None else None,
                "tgmm3_raw_delta": self.tgmm3.last_raw_delta.detach() if self.tgmm3.last_raw_delta is not None else None,
                "tgmm3_delta": self.tgmm3.last_delta.detach() if self.tgmm3.last_delta is not None else None,
                "ms_p3_depth_geo": depth_geo_p3.detach(),
                "ms_p3_learned_update_gate": self.decoder_refine3.last_learned_update_gate.detach() if self.decoder_refine3.last_learned_update_gate is not None else None,
                "ms_p3_depth_gate": self.decoder_refine3.last_depth_gate.detach() if self.decoder_refine3.last_depth_gate is not None else None,
                "ms_p3_update_gate": self.decoder_refine3.last_update_gate.detach() if self.decoder_refine3.last_update_gate is not None else None,
                "ms_p3_decay": self.decoder_refine3.last_decay.detach() if self.decoder_refine3.last_decay is not None else None,
                "ms_p3_raw_delta": self.decoder_refine3.last_raw_delta.detach() if self.decoder_refine3.last_raw_delta is not None else None,
                "ms_p3_delta": self.decoder_refine3.last_delta.detach() if self.decoder_refine3.last_delta is not None else None,

                "decoder_p2_topdown": p2_topdown.detach(),
                "decoder_p2_base": p2_base.detach(),
                "decoder_p2_dgp_state": self.geometry_calibration2.last_state.detach() if self.geometry_calibration2.last_state is not None else None,
                "decoder_p2_dgp_gate": self.geometry_calibration2.last_gate.detach() if self.geometry_calibration2.last_gate is not None else None,
                "decoder_p2_dgp_delta": self.geometry_calibration2.last_delta.detach() if self.geometry_calibration2.last_delta is not None else None,
                "decoder_p2_dls_state": tgmm2.detach(),
                "decoder_p2_ldr_state": p2_ldr.detach(),
                "decoder_p2_refined": p2.detach(),
                "spr_gate_logits": self.state_readout2.last_gate_logits.detach() if self.state_readout2.last_gate_logits is not None else None,
                "spr_gate": self.state_readout2.last_gate.detach() if self.state_readout2.last_gate is not None else None,
                "spr_restore": self.state_readout2.last_restore.detach() if self.state_readout2.last_restore is not None else None,
                "spr_delta": self.state_readout2.last_delta.detach() if self.state_readout2.last_delta is not None else None,
                "spr_state": self.state_readout2.last_state.detach() if self.state_readout2.last_state is not None else None,
                "tgmm2_depth_geo": self.tgmm2.last_depth_geo.detach() if self.tgmm2.last_depth_geo is not None else None,
                "tgmm2_target_prior": self.tgmm2.last_target_prior.detach() if self.tgmm2.last_target_prior is not None else None,
                "tgmm2_target_support": self.tgmm2.last_target_support.detach() if self.tgmm2.last_target_support is not None else None,
                "tgmm2_target_depth_geo": self.tgmm2.last_target_depth_geo.detach() if self.tgmm2.last_target_depth_geo is not None else None,
                "tgmm2_memory_state": self.tgmm2.last_memory_state.detach() if self.tgmm2.last_memory_state is not None else None,
                "tgmm2_prev_state": self.tgmm2.last_prev_state.detach() if self.tgmm2.last_prev_state is not None else None,
                "tgmm2_learned_update_gate": self.tgmm2.last_learned_update_gate.detach() if self.tgmm2.last_learned_update_gate is not None else None,
                "tgmm2_reliability": self.tgmm2.last_reliability.detach() if self.tgmm2.last_reliability is not None else None,
                "tgmm2_reliability_factor": self.tgmm2.last_reliability_factor.detach() if self.tgmm2.last_reliability_factor is not None else None,
                "tgmm2_suppress_alpha": self.tgmm2.last_suppress_alpha.detach().view(1, 1, 1, 1) if self.tgmm2.last_suppress_alpha is not None else None,
                "tgmm2_iter_steps": self.tgmm2.last_iter_steps.detach().view(1, 1, 1, 1) if self.tgmm2.last_iter_steps is not None else None,
                "tgmm2_update_gate": self.tgmm2.last_update_gate.detach() if self.tgmm2.last_update_gate is not None else None,
                "tgmm2_decay": self.tgmm2.last_decay.detach() if self.tgmm2.last_decay is not None else None,
                "tgmm2_raw_delta": self.tgmm2.last_raw_delta.detach() if self.tgmm2.last_raw_delta is not None else None,
                "tgmm2_delta": self.tgmm2.last_delta.detach() if self.tgmm2.last_delta is not None else None,
                "ms_p2_depth_geo": depth_geo.detach(),
                "ms_p2_edge_prob": edge_prob_low.detach(),
                "ms_p2_learned_update_gate": self.gcdub2.last_learned_update_gate.detach() if self.gcdub2.last_learned_update_gate is not None else None,
                "ms_p2_cue_gate_raw": self.gcdub2.last_cue_gate_raw.detach() if self.gcdub2.last_cue_gate_raw is not None else None,
                "ms_p2_cue_gate": self.gcdub2.last_cue_gate.detach() if self.gcdub2.last_cue_gate is not None else None,
                "ms_p2_update_gate": self.gcdub2.last_update_gate.detach() if self.gcdub2.last_update_gate is not None else None,
                "ms_p2_decay": self.gcdub2.last_decay.detach() if self.gcdub2.last_decay is not None else None,
                "ms_p2_raw_delta": self.gcdub2.last_raw_delta.detach() if self.gcdub2.last_raw_delta is not None else None,
                "ms_p2_refined_delta": self.gcdub2.last_refined_delta.detach() if self.gcdub2.last_refined_delta is not None else None,
                "ms_p2_delta": self.gcdub2.last_delta.detach() if self.gcdub2.last_delta is not None else None,

                # Backward-compatible names for the final p2 liquid update.
                "ldr_depth_geo": depth_geo.detach(),
                "ldr_edge_prob": edge_prob_low.detach(),
                "ldr_cue": self.gcdub2.last_cue.detach() if self.gcdub2.last_cue is not None else None,
                "ldr_decay": self.gcdub2.last_decay.detach() if self.gcdub2.last_decay is not None else None,
                "ldr_raw_delta": self.gcdub2.last_raw_delta.detach() if self.gcdub2.last_raw_delta is not None else None,
                "gcdub2d_delta": self.gcdub2.last_refined_delta.detach() if self.gcdub2.last_refined_delta is not None else None,
                "ldr_delta": self.gcdub2.last_delta.detach() if self.gcdub2.last_delta is not None else None,
                "ldr_learned_update_gate": self.gcdub2.last_learned_update_gate.detach() if self.gcdub2.last_learned_update_gate is not None else None,
                "ldr_cue_gate_raw": self.gcdub2.last_cue_gate_raw.detach() if self.gcdub2.last_cue_gate_raw is not None else None,
                "ldr_cue_gate": self.gcdub2.last_cue_gate.detach() if self.gcdub2.last_cue_gate is not None else None,
                "ldr_update_gate": self.gcdub2.last_update_gate.detach() if self.gcdub2.last_update_gate is not None else None,
            }
            return m1, m2, m3, m4, edge, aux

        return m1, m2, m3, m4, edge



# -----------------------------------------------------------------------------
# Legacy checkpoint compatibility
# -----------------------------------------------------------------------------
LEGACY_STATE_PREFIX_MAP = (
    ("early_rgbd_adapter.", "rgbd_projection.proj."),
    ("rgb_backbone.", "encoder."),
    ("rgb_projectors.", "feature_projectors."),
    ("pseudo_rgbd_adapter.", "rgbd_projection."),
    ("gsdt.detail_r2.", "sgdem.detail_f1."),
    ("gsdt.detail_r3.", "sgdem.detail_f2."),
    ("gsdt.semantic_r6.", "sgdem.semantic_f4."),
    ("gsdt.semantic_r8.", "sgdem.semantic_f5."),
    ("gsdt.", "sgdem."),
    ("sgdem.q_norm.", "sgdem.sgdb.q_norm."),
    ("sgdem.kv_norm.", "sgdem.sgdb.kv_norm."),
    ("sgdem.attn.", "sgdem.sgdb.attn."),
    ("sgdem.ffn_norm.", "sgdem.sgdb.ffn_norm."),
    ("sgdem.ffn.", "sgdem.sgdb.ffn."),
    ("sgdem.attn_proj.", "sgdem.sgdb.attn_proj."),
    ("sgdem.semantic_region_gate.", "sgdem.sgdb.semantic_region_gate."),
    ("sgdem.dgp_prior.", "sgdem.geometry_calibration."),
    ("sgdem.liquid_residual.", "sgdem.dcub."),
    ("decoder.depth_geometry.", "decoder.geometry_extractor."),
    ("decoder.dgp_p4.", "decoder.geometry_calibration4."),
    ("decoder.dgp_p3.", "decoder.geometry_calibration3."),
    ("decoder.dgp_p2.", "decoder.geometry_calibration2."),
    ("decoder.dls_p4.", "decoder.tgmm4."),
    ("decoder.dls_p3.", "decoder.tgmm3."),
    ("decoder.dls_p2.", "decoder.tgmm2."),
    ("decoder.ms_refine_p4.", "decoder.decoder_refine4."),
    ("decoder.ms_refine_p3.", "decoder.decoder_refine3."),
    ("decoder.ldr_refine.", "decoder.gcdub2."),
    ("decoder.spr_readout.", "decoder.state_readout2."),
    ("decoder.mask1.", "decoder.pred_head2."),
    ("decoder.mask2.", "decoder.pred_head3."),
    ("decoder.mask3.", "decoder.pred_head4."),
    ("decoder.mask4.", "decoder.pred_head5."),
)


def remap_legacy_state_dict(state_dict):
    """Return a paper-named state_dict without changing tensor values.

    Handles checkpoints saved by the provided GoodRT_v2 code, DataParallel
    prefixes, and the older v4b naming used by the bundled inference script.
    """
    try:
        from collections import OrderedDict
        out = OrderedDict()
    except Exception:
        out = {}

    for key, value in state_dict.items():
        new_key = key
        while new_key.startswith("module."):
            new_key = new_key[len("module."):]
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]

        # Some transformations are chained (e.g. gsdt -> sgdem -> dcub).
        changed = True
        while changed:
            changed = False
            for old, new_prefix in LEGACY_STATE_PREFIX_MAP:
                if new_key.startswith(old):
                    new_key = new_prefix + new_key[len(old):]
                    changed = True
                    break
        out[new_key] = value

    if hasattr(state_dict, "_metadata"):
        out._metadata = getattr(state_dict, "_metadata")
    return out


def extract_state_dict(checkpoint, use_ema=True):
    """Extract weights from common RTSRNet/legacy checkpoint containers."""
    if not isinstance(checkpoint, dict):
        return checkpoint
    if use_ema and "ema_model" in checkpoint:
        return checkpoint["ema_model"]
    for key in ("model", "state_dict"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


class RTSRNet(nn.Module):
    """Residual Target-Sensitive Refinement Network (RTSRNet).

    This paper-facing refactor preserves the original forward computation and
    adds automatic legacy checkpoint key conversion.
    """

    def __init__(self, pretrained=True, fuse_channels=(32, 64, 96, 160), ablation_mode="full"):
        super().__init__()
        self.ablation_mode = normalize_ablation_mode(ablation_mode)
        flags = ablation_flags(self.ablation_mode)
        self.use_dgp = flags["use_dgp"]
        self.use_dls = flags["use_dls"]
        self.use_idls = flags["use_idls"]
        self.use_rdls = flags["use_rdls"]
        self.use_spr = flags["use_spr"]
        self.rgbd_projection = RGBDProjection()
        self.encoder = EfficientNet_B0(pretrained=pretrained)
        stage_channels = self.encoder.get_stage_channels()

        # Decoder uses F2/F3/F4/F5, corresponding to EfficientNet stages r3/r4/r6/r8.
        projector_stage_indices = [1, 2, 3, 4]
        self.feature_projectors = nn.ModuleList([
            FeatureProjector(stage_channels[i], fuse_channels[j])
            for j, i in enumerate(projector_stage_indices)
        ])

        self.sgdem = SGDEM(
            f1_channels=stage_channels[0],
            f2_channels=stage_channels[1],
            f4_channels=stage_channels[3],
            f5_channels=stage_channels[4],
            out_channels=fuse_channels[0],
            num_heads=4,
            ablation_mode=self.ablation_mode,
        )
        self.decoder = RTSRDecoder(fuse_channels, 64, use_edge_refine=True, ablation_mode=self.ablation_mode)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load both paper-named and legacy GoodRT_v2 checkpoints.

        The compatibility layer only renames keys; parameter tensors are not
        altered. Therefore an original .pth can be loaded directly with
        ``model.load_state_dict(old_state, strict=True)``.
        """
        state_dict = remap_legacy_state_dict(state_dict)
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            # PyTorch versions before ``assign`` was added.
            return super().load_state_dict(state_dict, strict=strict)

    def _project_decoder_features(self, f2, f3, f4, f5):
        return (
            self.feature_projectors[0](f2),
            self.feature_projectors[1](f3),
            self.feature_projectors[2](f4),
            self.feature_projectors[3](f5),
        )

    def forward(self, rgb, depth, return_aux=False):
        rectified_depth = depth
        rect_edge = torch.zeros_like(depth)

        x = self.rgbd_projection(rgb, rectified_depth)
        f1, f2, f3, f4, f5 = self.encoder(x)

        dec_f2, dec_f3, dec_f4, dec_f5 = self._project_decoder_features(f2, f3, f4, f5)

        if return_aux:
            delta_f, sgdem_aux = self.sgdem(f1, f2, f4, f5, depth, return_aux=True)
        else:
            delta_f = self.sgdem(f1, f2, f4, f5, depth, return_aux=False)
            sgdem_aux = None

        dec_f2 = dec_f2 + delta_f
        if return_aux:
            m1, m2, m3, m4, edge_pred, decoder_aux = self.decoder(
                (dec_f2, dec_f3, dec_f4, dec_f5),
                rect_edge,
                rgb.shape[2:],
                depth=depth,
                return_aux=True,
            )
            sgdem_aux = dict(sgdem_aux)
            sgdem_aux.update({
                "ablation_mode": self.ablation_mode,
                "enhanced_f2": dec_f2.detach(),
            })
            return m1, m2, m3, m4, edge_pred, rectified_depth, rect_edge, {
                "sgdem": sgdem_aux,
                "decoder": decoder_aux,
            }

        m1, m2, m3, m4, edge_pred = self.decoder(
            (dec_f2, dec_f3, dec_f4, dec_f5),
            rect_edge,
            rgb.shape[2:],
            depth=depth,
            return_aux=False,
        )

        return m1, m2, m3, m4, edge_pred



# Deprecated source-level aliases. Existing external imports continue to work,
# while README/examples use the paper-facing names above.
GSDTNet = RTSRNet
GSDT = SGDEM
# SGDB has no legacy top-level class; it was previously inlined inside GSDT/SGDEM.
CGGDecoder = RTSRDecoder
PseudoRGBDAdapter = RGBDProjection
LiquidCore_DLS_SPR_V2Fuse = DCUB
DepthGeometryPrior = GeometryCalibration
DepthGeometry = DepthGeometryExtractor
DepthLiquidStateUpdate = TGMM
LiquidMSDecoderRefinement = DecoderRefinementBlock
StatePreservingReadout = DecoderStateReadout
LiquidCore_DLS_SPR_V2DecoderRefinement = GCDUB
DLSReliabilitySuppressor = ReliabilitySuppressor

if __name__ == '__main__':
    model = RTSRNet(pretrained=False, ablation_mode='full')
    total = sum(p.numel() for p in model.parameters())
    backbone = sum(p.numel() for p in model.encoder.parameters())
    print('total params:', total)
    print('backbone params:', backbone)
    print('extra params:', total - backbone)
    rgb = torch.randn(1, 3, 384, 384)
    dep = torch.randn(1, 1, 384, 384)
    with torch.no_grad():
        outs = model(rgb, dep)
    for i, y in enumerate(outs, 1):
        print(f'out{i}:', y.shape)
