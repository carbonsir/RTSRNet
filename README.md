# RTSRNet

Official code release for **RTSRNet: Residual Target-Sensitive Refinement Network for Depth-Assisted Camouflaged Object Detection**.

RTSRNet formulates refinement-stage cue utilization as **target-sensitive cue refinement**. Instead of directly treating shallow detail and depth-derived geometry as uniformly reliable content, the network uses target-related context to regulate how these ambiguous cues update task representations.

> **Code-release note.** This repository is a paper-facing refactor of the provided `GoodRT_v2` implementation. The module/class names have been aligned with the manuscript while the trained forward tensor operations are intentionally preserved so that existing `GoodRT_v2` checkpoints can still be loaded. Legacy `state_dict` keys are converted automatically.

## Highlights

- **SGDEM — Semantic-Geometric Detail Enhancement Module.** Establishes a semantic-guided reference for ambiguous shallow responses and performs depth-conditioned shallow refinement.
- **SGDB — Semantic-guided Detail Block.** Uses shallow detail to query high-level semantics and constructs a semantic-guided response.
- **DCUB — Depth-Conditioned Update Block.** Performs the trained encoder-side controlled update used by SGDEM.
- **TGMM — Target-guided Geometry Modulation Module.** Provides decoder-stage target-conditioned geometry refinement.
- **GCDUB — Geometry-Conditioned Decoder Update Block.** Paper-facing name for the trained decoder update/refinement block.
- **Legacy checkpoint compatibility.** Original `.pth` files saved with `GSDTNet`, `gsdt`, `dls_p*`, `mask*`, etc. are remapped on load without changing tensor values.

## Repository Structure

```text
RTSRNet/
├── Model/
│   ├── RTSRNet.py              # RTSRNet, SGDEM, SGDB, DCUB, TGMM, GCDUB
│   ├── GSDTNet.py              # deprecated compatibility import shim
│   ├── EfficientNet.py
│   ├── modules.py
│   └── __init__.py
├── utils/
│   ├── config.py
│   ├── edge_dataloader.py
│   ├── metrics.py
│   └── utils.py
├── tools/
│   └── check_checkpoint.py     # strict legacy/new checkpoint validation
├── tests/
│   └── test_checkpoint_compat.py
├── train.py
├── inference.py
├── evaluate.py
├── profile_model.py
├── analyze_internal_responses.py
├── go.py
├── go.sh
├── requirements.txt
├── MIGRATION.md
└── README.md
```

## Paper-to-Code Mapping

| Manuscript term | Code |
|---|---|
| RTSRNet | `Model.RTSRNet.RTSRNet` |
| RGB-D projection | `RGBDProjection` |
| SGDEM | `SGDEM` |
| SGDB | `SGDB` |
| DCUB | `DCUB` |
| Depth-derived geometry | `DepthGeometryExtractor` / SGDEM Sobel geometry |
| TGMM at decoder stages | `decoder.tgmm4`, `decoder.tgmm3`, `decoder.tgmm2` |
| GCDUB / final decoder refinement | `GCDUB` / `decoder.gcdub2` |
| \(P_2,P_3,P_4,P_5\) | `pred_head2`, `pred_head3`, `pred_head4`, `pred_head5` |
| Edge prediction \(P_e\) | `edge_head` |

The refactor is intentionally **name/organization compatible rather than a re-training rewrite**: forward tensor operations from the supplied implementation are preserved to protect the numerical behavior of existing checkpoints.

## Installation

A typical environment can be created with:

```bash
conda create -n rtsrnet python=3.10 -y
conda activate rtsrnet
pip install -r requirements.txt
```

The code was written for PyTorch and supports CUDA when available.

## Dataset Preparation

The training protocol in the manuscript uses the standard COD training split:

- 2,026 training images from COD10K
- 1,000 training images from CAMO

Evaluation is performed on CAMO, COD10K, and NC4K. The bundled evaluation code also supports CHAMELEON.

The repository expects the following layout:

```text
DATA_ROOT/
├── TrainDataset/
│   ├── Imgs/
│   ├── GT/
│   ├── Depth/
│   └── Edge/
└── TestDataset/
    ├── CAMO/
    │   ├── Imgs/
    │   ├── GT/
    │   ├── Depth/
    │   └── Edge/
    ├── COD10K/
    │   ├── Imgs/
    │   ├── GT/
    │   ├── Depth/
    │   └── Edge/
    ├── NC4K/
    │   ├── Imgs/
    │   ├── GT/
    │   ├── Depth/
    │   └── Edge/
    └── CHAMELEON/
        ├── Imgs/
        ├── GT/
        ├── Depth/
        └── Edge/
```

Pass the root directory with `--dataset_dir`.

### Depth Maps

The manuscript uses **Depth Anything V2** as the default monocular depth estimator. Depth maps are generated offline; the depth estimator is not part of RTSRNet and its cost is not included in COD-network complexity.

This repository expects the generated depth maps to already be present in the `Depth/` directories and spatially matched to the RGB images.

## Model Construction

```python
from Model.RTSRNet import RTSRNet

model = RTSRNet(
    pretrained=True,
    ablation_mode="full",
)
```

For the paper-facing RTSRNet-E model, use `ablation_mode="full"`.

The historical modes `baseline`, `dgp`, `dgp_dls`, `idls`, and `goodr` are retained only to keep old experiments/checkpoints usable. They correspond to internal development ablations and should not be confused with the manuscript's SGDEM/TGMM component-ablation table.

## Loading the Original `.pth`

Existing checkpoints saved by the original `GoodRT_v2` code can be loaded directly.

```python
import torch
from Model.RTSRNet import RTSRNet, extract_state_dict

ckpt = torch.load("best.pth", map_location="cpu")
state = extract_state_dict(ckpt, use_ema=True)

model = RTSRNet(pretrained=False, ablation_mode="full")
model.load_state_dict(state, strict=True)
model.eval()
```

The loader automatically translates legacy names such as:

```text
pseudo_rgbd_adapter.*  -> rgbd_projection.*
gsdt.*                 -> sgdem.*
gsdt.liquid_residual.* -> sgdem.dcub.*
decoder.dls_p4.*       -> decoder.tgmm4.*
decoder.dls_p3.*       -> decoder.tgmm3.*
decoder.dls_p2.*       -> decoder.tgmm2.*
decoder.mask1.*         -> decoder.pred_head2.*
...
```

No checkpoint tensor is numerically modified.

### Check a Checkpoint

```bash
python tools/check_checkpoint.py \
  --ckpt /path/to/best.pth \
  --ablation_mode full
```

A successful run reports strict loading and performs a small forward smoke test.

> The source archive used for this refactor did **not** include the actual trained `.pth`; therefore the repository includes an automated compatibility test and checker, but you should run the command above once on the final released checkpoint before publishing.

## Training

The manuscript-facing default uses four-scale structure-aware mask supervision plus auxiliary edge supervision:

\[
L = L_{\mathrm{mask}} + 0.05 L_{\mathrm{edge}}.
\]

Run:

```bash
python train.py \
  --dataset_dir /path/to/DATA_ROOT \
  --ablation_mode full \
  --epochs 180 \
  --batch_size 16 \
  --trainsize 384 \
  --amp
```

Default optimization settings are aligned with the manuscript:

- optimizer: AdamW
- epochs: 180
- batch size: 16
- input size: 384 × 384
- initial learning rate: \(3\times10^{-4}\)
- minimum learning rate: \(1\times10^{-6}\)
- weight decay: \(1\times10^{-4}\)
- backbone learning-rate multiplier: 0.2
- EMA decay: 0.999

### Legacy Training Loss

The original `GoodRT_v2` training script contained an additional reliability/target-control auxiliary loss. This refactor disables it by default so the public training command follows the manuscript loss.

To reproduce the **old training-script behavior** rather than the manuscript-facing default, add:

```bash
--legacy_control_loss --rel_loss_weight 0.05
```

This option does not affect checkpoint loading or inference.

## Inference

```bash
python inference.py \
  --ckpt /path/to/best.pth \
  --dataset_dir /path/to/DATA_ROOT \
  --datasets CAMO COD10K NC4K \
  --ablation_mode full
```

By default, if a checkpoint contains EMA weights, inference uses `ema_model`. Add `--use_raw` to use the raw `model` state instead.

Prediction maps are written to the configured output directory.

## Evaluation

If prediction maps already exist:

```bash
python evaluate.py \
  --ckpt /path/to/best.pth \
  --dataset_dir /path/to/DATA_ROOT \
  --datasets CAMO COD10K NC4K \
  --save_json
```

The bundled evaluator reports the metrics used in the manuscript:

- \(S_{\alpha}\): structure measure
- \(E_{\phi}^{ad}\): adaptive enhanced-alignment measure
- \(F_{\beta}^{\omega}\): weighted F-measure
- \(M\): mean absolute error

If predictions are missing, `evaluate.py` can invoke inference first.

## Profiling

```bash
python profile_model.py \
  --ckpt /path/to/best.pth \
  --img_size 384 \
  --batch_size 1 \
  --device cuda \
  --amp
```

Install `thop` to report MACs/FLOPs:

```bash
pip install thop
```

The script reports parameters, a module-level parameter breakdown, MACs/FLOPs when available, latency, and FPS.

## Internal Response Analysis

The manuscript's internal-response visualizations can be supported through:

```bash
python analyze_internal_responses.py --help
```

The analysis interface uses the paper-facing `sgdem` and `tgmm*` names. The old `analyze_bridge_diagnostics.py` filename is retained as a deprecated entry point.

## One-Command Pipeline

The bundled pipeline can run training, inference, evaluation, and analysis:

```bash
python go.py \
  --data /path/to/DATA_ROOT \
  --mode full
```

or:

```bash
bash go.sh /path/to/DATA_ROOT
```

## Checkpoint Compatibility Test

Run:

```bash
python tests/test_checkpoint_compat.py
```

The test creates a model, converts its state dictionary into the legacy GoodRT_v2 naming convention, strictly reloads it through the RTSRNet compatibility layer, and checks that evaluation outputs are bit-identical.

## Important Reproducibility Notes

1. **Checkpoint compatibility was prioritized.** The refactor changes names and organization but intentionally avoids changing trained forward tensor operations.
2. **The provided source archive contains the EfficientNet-B0 implementation.** The manuscript also reports SMT-B and PVTv2-B4 variants, but their backbone implementations were not present in the supplied archive and are therefore not invented in this release.
3. **Depth generation is external.** Generate depth maps offline and place them in the expected folders.
4. **Paper loss vs. historical training script.** The default training loss in this release follows the manuscript. The previous auxiliary control loss remains available only behind `--legacy_control_loss`.
5. Before public release, validate the final distributed checkpoint with `tools/check_checkpoint.py`.

## Legacy Name Migration

See [`MIGRATION.md`](MIGRATION.md) for a more detailed old-to-new naming table.

## Citation

The bibliographic record was not included in the supplied project archive. Add the final publisher-provided BibTeX entry here after the article is accepted/published; this README intentionally does not invent author, journal, DOI, or year metadata.

## Acknowledgements

The implementation uses an EfficientNet-B0 encoder and standard COD evaluation metrics. Please also cite the corresponding original works and datasets used in your experiments.

## License

No license file was present in the supplied source archive. Choose and add the intended repository license before making the GitHub repository public.
