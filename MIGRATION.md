# Migration from GoodRT_v2 to RTSRNet Names

This refactor changes public names while preserving the supplied implementation's forward computation.

| GoodRT_v2 name | RTSRNet paper-facing name |
|---|---|
| `GSDTNet` | `RTSRNet` |
| `PseudoRGBDAdapter` | `RGBDProjection` |
| `GSDT` | `SGDEM` |
| inlined semantic/detail cross-attention | `SGDB` |
| `LiquidCore_DLS_SPR_V2Fuse` | `DCUB` |
| `DepthGeometry` | `DepthGeometryExtractor` |
| `DepthLiquidStateUpdate` | `TGMM` |
| `LiquidCore_DLS_SPR_V2DecoderRefinement` | `GCDUB` |
| `CGGDecoder` | `RTSRDecoder` |
| `gsdt` | `sgdem` |
| `decoder.dls_p4` | `decoder.tgmm4` |
| `decoder.dls_p3` | `decoder.tgmm3` |
| `decoder.dls_p2` | `decoder.tgmm2` |
| `decoder.mask1` | `decoder.pred_head2` |
| `decoder.mask2` | `decoder.pred_head3` |
| `decoder.mask3` | `decoder.pred_head4` |
| `decoder.mask4` | `decoder.pred_head5` |

## Backward-compatible imports

Old code remains valid:

```python
from Model.GSDTNet import GSDTNet
model = GSDTNet(pretrained=False, ablation_mode="full")
```

The preferred public API is:

```python
from Model.RTSRNet import RTSRNet
model = RTSRNet(pretrained=False, ablation_mode="full")
```

## Backward-compatible checkpoints

`RTSRNet.load_state_dict()` automatically remaps old parameter prefixes. Use strict loading whenever possible:

```python
model.load_state_dict(old_state_dict, strict=True)
```

This is a key-only migration: parameter values are unchanged.
