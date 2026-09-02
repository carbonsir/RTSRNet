# RTSRNet

## Residual Target-Sensitive Refinement Network for Depth-Assisted Camouflaged Object Detection

**Tan Song, Jinbao Li**

> Manuscript submitted to **IEEE Transactions on Multimedia (TMM)**.

RTSRNet is a depth-assisted camouflaged object detection framework that formulates refinement-stage cue utilization as **target-sensitive cue refinement**. Instead of directly treating shallow detail and depth-derived geometry as uniformly reliable content, RTSRNet uses target-related context to regulate how ambiguous cues update task representations.

The framework contains two main modules:

- **SGDEM — Semantic-Geometric Detail Enhancement Module:** establishes a semantic-guided reference for ambiguous shallow responses and combines semantic discrepancy with depth-derived geometry to refine shallow representations.
- **TGMM — Target-guided Geometry Modulation Module:** transforms raw depth-derived geometry into target-aware geometry and uses it to progressively refine decoder representations.

---

## News

- **2026:** RTSRNet manuscript submitted to **IEEE Transactions on Multimedia (TMM)**.
- Code, trained weights, prepared RGB/depth data, prediction maps, and evaluation scripts are provided for reproducibility.

---

## Overview of RTSRNet

<p align="center">
  <img src="assets/Fig2_Overview.png" width="100%">
</p>

<p align="center">
  <b>Overview of RTSRNet.</b> RGB and depth inputs are encoded with SGDEM for shallow refinement and TGMM for multi-scale decoder refinement.
</p>

Given an RGB image and its estimated depth map, RTSRNet first performs RGB-D projection and hierarchical feature extraction. SGDEM refines the shallow encoder representation using detail, semantic, and depth-derived geometric information. The decoder then applies TGMM at multiple stages to convert raw geometry into target-aware guidance and progressively refine the decoder states.

---

## Main Results

### Table I. Quantitative Comparison

Quantitative comparison with representative COD methods on **CAMO**, **COD10K**, and **NC4K**. The best and second-best results within each group are shown in **bold** and <u>underlined</u>, respectively.

| Method | Pub. | Input | Backbone | CAMO Sα | CAMO Eφad | CAMO Fβω | CAMO M | COD10K Sα | COD10K Eφad | COD10K Fβω | COD10K M | NC4K Sα | NC4K Eφad | NC4K Fβω | NC4K M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **RGB-based COD Methods** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SINet | CVPR'20 | 352×352 | ResNet50 | 0.751 | 0.834 | 0.606 | 0.100 | 0.776 | 0.867 | 0.631 | 0.043 | 0.808 | 0.883 | 0.723 | 0.058 |
| SINetV2 | TPAMI'21 | 352×352 | ResNet50 | 0.820 | 0.884 | 0.743 | 0.070 | 0.815 | 0.864 | 0.680 | 0.037 | 0.847 | 0.901 | 0.770 | 0.048 |
| FEDER | CVPR'23 | 384×384 | ResNet50 | 0.802 | 0.877 | 0.738 | 0.071 | 0.822 | 0.902 | 0.716 | 0.032 | 0.847 | 0.913 | 0.789 | 0.044 |
| PUENet | TIP'23 | 512×512 | Res2Net50 | 0.794 | 0.861 | 0.722 | 0.077 | 0.809 | 0.884 | 0.687 | 0.037 | 0.835 | 0.892 | 0.762 | 0.049 |
| DINet | TMM'24 | 400×400 | Res2Net50 | 0.821 | 0.883 | 0.790 | 0.068 | 0.832 | 0.901 | 0.744 | 0.031 | 0.856 | 0.910 | 0.820 | 0.043 |
| FSEL | ECCV'24 | 416×416 | PVTv2-B4 | 0.822 | 0.892 | 0.758 | 0.067 | 0.838 | 0.900 | 0.724 | 0.029 | 0.855 | 0.913 | 0.792 | 0.042 |
| CamoFormer | TPAMI'24 | 384×384 | PVTv2-B4 | 0.816 | 0.884 | 0.756 | 0.066 | 0.836 | 0.898 | 0.730 | 0.029 | 0.858 | 0.914 | 0.793 | 0.041 |
| ESNet-S | KBS'25 | 384×384 | SMT-B | <u>0.877</u> | **0.934** | <u>0.861</u> | <u>0.044</u> | 0.871 | 0.935 | 0.811 | 0.022 | 0.893 | <u>0.941</u> | <u>0.870</u> | 0.030 |
| ESCNet | ICCV'25 | 416×416 | PVTv2-B4 | 0.871 | <u>0.932</u> | 0.843 | <u>0.044</u> | 0.873 | 0.936 | 0.804 | <u>0.021</u> | 0.892 | 0.938 | 0.859 | <u>0.028</u> |
| FBD-Net | TOMM'25 | 384×384 | PVTv2-B4 | **0.881** | 0.931 | 0.844 | **0.043** | 0.877 | 0.935 | 0.799 | 0.022 | 0.893 | 0.939 | 0.852 | 0.030 |
| CSFIN | ESWA'25 | 384×384 | SMT-T | 0.876 | 0.929 | 0.832 | 0.047 | 0.868 | 0.930 | 0.780 | 0.023 | 0.890 | 0.937 | 0.842 | 0.031 |
| MCSWA-Net | TMM'26 | 512×512 | PVTv2-B4 | <u>0.877</u> | 0.926 | **0.877** | 0.046 | **0.886** | <u>0.940</u> | <u>0.817</u> | **0.020** | <u>0.894</u> | 0.937 | 0.854 | 0.031 |
| PONet | PR'26 | 384×384 | PVTv2-B4 | 0.874 | 0.929 | 0.833 | 0.045 | 0.874 | 0.934 | 0.792 | 0.022 | 0.892 | 0.938 | 0.848 | 0.030 |
| ZoomingCOD | TMM'26 | 512×512 | Mamba | 0.874 | 0.929 | 0.833 | 0.045 | <u>0.880</u> | **0.941** | **0.825** | **0.020** | **0.906** | **0.945** | **0.872** | **0.022** |
| **Depth-assisted COD Methods** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| DaCOD | ACM MM'23 | 448×448 | SMT-B | 0.855 | 0.911 | 0.796 | 0.051 | 0.840 | 0.908 | 0.729 | 0.028 | 0.874 | 0.923 | 0.814 | 0.035 |
| PopNet | ICCV'23 | 352×352 | — | 0.806 | 0.869 | 0.821 | 0.073 | 0.827 | 0.897 | 0.789 | 0.031 | 0.852 | 0.908 | 0.851 | 0.043 |
| DAF-Net | IVC'24 | 352×352 | ResNet50 | 0.860 | 0.913 | 0.799 | 0.051 | 0.838 | 0.899 | 0.715 | 0.031 | 0.865 | 0.909 | 0.792 | 0.042 |
| DSAM | ACM MM'24 | 1024×1024 | SAM-B | 0.832 | 0.920 | 0.794 | 0.061 | 0.845 | 0.931 | 0.760 | 0.033 | 0.871 | 0.940 | 0.826 | 0.040 |
| CAM-Net | ICVRV'25 | 448×448 | ResNet50 | 0.842 | 0.908 | 0.785 | 0.047 | 0.838 | 0.910 | 0.730 | 0.028 | 0.871 | 0.931 | 0.815 | 0.035 |
| SAM-DSA | ICCV'25 | 1024×1024 | SAM-B | 0.875 | <u>0.952</u> | 0.849 | 0.044 | <u>0.887</u> | **0.948** | <u>0.827</u> | 0.022 | 0.896 | **0.959** | <u>0.866</u> | 0.029 |
| DASFF-Net | DSP'26 | 518×518 | SMT-B | **0.900** | 0.947 | <u>0.873</u> | **0.034** | 0.880 | 0.935 | 0.809 | <u>0.020</u> | 0.897 | 0.939 | 0.860 | <u>0.028</u> |
| DMLR-Net | EAAI'26 | 518×518 | SMT-B | 0.885 | 0.936 | 0.863 | 0.039 | **0.893** | <u>0.944</u> | **0.834** | **0.018** | **0.903** | 0.947 | <u>0.866</u> | <u>0.028</u> |
| **RTSRNet-SM** | Ours | 384×384 | SMT-B | <u>0.886</u> | **0.955** | **0.876** | <u>0.035</u> | 0.882 | 0.943 | 0.805 | 0.021 | <u>0.899</u> | <u>0.950</u> | **0.891** | **0.019** |
| **Compact-Backbone COD Methods** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| TinyCOD | ICASSP'23 | 384×384 | TinyNet | 0.822 | 0.890 | 0.752 | 0.065 | 0.811 | 0.877 | 0.678 | 0.036 | 0.843 | 0.903 | 0.766 | 0.047 |
| DGNet-S | MIR'23 | 352×352 | EfficientNet-B0 | 0.824 | 0.892 | 0.754 | 0.063 | 0.810 | 0.868 | 0.672 | 0.036 | 0.845 | 0.899 | 0.764 | 0.047 |
| ASBI | CVIU'23 | 384×384 | EfficientNet-B0 | 0.839 | 0.896 | 0.761 | 0.064 | 0.825 | 0.872 | 0.690 | 0.035 | 0.855 | 0.902 | 0.775 | 0.046 |
| FINet | SPL'24 | 384×384 | EfficientNet-B0 | 0.828 | 0.890 | 0.752 | 0.065 | 0.817 | 0.883 | 0.686 | 0.034 | 0.847 | 0.904 | 0.771 | 0.047 |
| BPNet | SPL'25 | 384×384 | EfficientNet-B0 | <u>0.864</u> | 0.913 | 0.815 | <u>0.048</u> | 0.847 | <u>0.909</u> | 0.748 | 0.027 | <u>0.867</u> | 0.919 | 0.812 | <u>0.039</u> |
| ESNet-E | KBS'25 | 384×384 | EfficientNet-B0 | 0.848 | <u>0.919</u> | <u>0.828</u> | 0.049 | 0.830 | 0.908 | 0.745 | 0.031 | 0.862 | 0.916 | 0.821 | 0.040 |
| FMLNet | ASOC'26 | 384×384 | MobileViT | 0.822 | 0.888 | 0.755 | 0.067 | 0.823 | 0.894 | 0.704 | 0.033 | 0.850 | 0.909 | 0.780 | 0.045 |
| ULCOD-Net | CAIS'26 | 384×384 | MobileViT | 0.823 | 0.890 | 0.758 | 0.067 | 0.829 | 0.893 | 0.714 | 0.033 | 0.854 | 0.908 | 0.787 | 0.045 |
| ECNet | PR'26 | 384×384 | EfficientNet-B0 | 0.856 | 0.913 | 0.787 | 0.049 | <u>0.860</u> | **0.927** | <u>0.780</u> | <u>0.026</u> | <u>0.867</u> | <u>0.927</u> | <u>0.878</u> | **0.020** |
| **RTSRNet-E** | Ours | 384×384 | EfficientNet-B0 | **0.879** | **0.923** | **0.829** | **0.046** | **0.872** | **0.927** | **0.786** | **0.025** | **0.884** | **0.930** | **0.888** | **0.020** |

---

## Qualitative Comparison

<p align="center">
  <img src="assets/Fig7_Qtt.png" width="100%">
</p>

<p align="center">
  <b>Fig. 7.</b> Qualitative comparison with representative COD methods under challenging camouflage conditions. Red boxes show enlarged regions for small-target comparison.
</p>

RTSRNet produces cleaner and more coherent predictions under texture ambiguity, geometry ambiguity, weak boundaries, small targets, large targets, and cluttered backgrounds.

---

## Ablation Visualization

<p align="center">
  <img src="assets/Fig8_Abla.png" width="72%">
</p>

<p align="center">
  <b>Fig. 8.</b> Qualitative ablation of SGDEM and TGMM under the same EfficientNet-B0 backbone. The single-module variants are compared with the baseline and the full RTSRNet-E.
</p>

The visual comparison shows the complementary effects of SGDEM and TGMM: SGDEM suppresses distracting shallow responses and preserves local structures, while TGMM improves target coherence through target-guided geometry refinement.

---

## Downloads

### Trained Model

| Model | Backbone | Input Size | Download |
|---|---|---:|---|
| RTSRNet-E | EfficientNet-B0 | 384×384 | [Google Drive](YOUR_RTSRNET_E_PTH_GOOGLE_DRIVE_URL) / [Baidu Netdisk](YOUR_RTSRNET_E_PTH_BAIDU_URL) |

### Training and Testing Data

RTSRNet is trained with **2,026 COD10K training images + 1,000 CAMO training images** and evaluated on **CAMO (250)**, **COD10K (2,026)**, and **NC4K (4,121)**.

For exact reproduction, release the RGB images, masks, edge maps, and the **Depth Anything V2 depth maps used by RTSRNet** in prepared packages:

| Resource | Content | Download |
|---|---|---|
| RTSRNet Training Set | CAMO-train + COD10K-train; RGB / GT / Depth / Edge | [Google Drive](YOUR_TRAIN_RGB_DEPTH_GOOGLE_DRIVE_URL) / [Baidu Netdisk](YOUR_TRAIN_RGB_DEPTH_BAIDU_URL) |
| RTSRNet Testing Sets | CAMO-test + COD10K-test + NC4K; RGB / GT / Depth / Edge | [Google Drive](YOUR_TEST_RGB_DEPTH_GOOGLE_DRIVE_URL) / [Baidu Netdisk](YOUR_TEST_RGB_DEPTH_BAIDU_URL) |

Original RGB/GT datasets:

- **CAMO:** https://drive.google.com/open?id=1h-OqZdwkuPhBvGcVAwmh0f1NGqlH_4B6
- **COD10K:** https://drive.google.com/file/d/1vRYAie0JcNStcSwagmCq55eirGyMYGm5/view?usp=sharing
- **NC4K:** https://drive.google.com/file/d/1kzpX_U3gbgO9MuwZIWTuRVpiB7V6yrAQ/view?usp=sharing

The default depth maps in the manuscript are generated offline with **Depth Anything V2**:

- https://github.com/DepthAnything/Depth-Anything-V2

### Prediction Maps / Test Results

| Results | Download |
|---|---|
| RTSRNet-E prediction maps on CAMO / COD10K / NC4K | [Google Drive](YOUR_RTSRNET_PREDICTIONS_GOOGLE_DRIVE_URL) / [Baidu Netdisk](YOUR_RTSRNET_PREDICTIONS_BAIDU_URL) |

---

## Environment

```bash
conda create -n rtsrnet python=3.10 -y
conda activate rtsrnet
pip install -r requirements.txt
```

---

## Repository Structure

```text
RTSRNet/
├── Model/
│   ├── RTSRNet.py
│   ├── EfficientNet.py
│   └── modules.py
├── utils/
│   ├── config.py
│   ├── edge_dataloader.py
│   ├── metrics.py
│   └── utils.py
├── train.py
├── inference.py
├── evaluate.py
├── profile_model.py
├── analyze_internal_responses.py
├── requirements.txt
└── README.md
```

### Paper-to-Code Mapping

| Manuscript | Code |
|---|---|
| RTSRNet | `Model.RTSRNet.RTSRNet` |
| RGB-D projection | `RGBDProjection` |
| SGDEM | `SGDEM` |
| SGDB | `SGDB` |
| DCUB | `DCUB` |
| Depth-derived geometry | `DepthGeometryExtractor` |
| TGMM | `TGMM` |
| GCDUB | `GCDUB` |
| Prediction maps P2/P3/P4/P5 | `pred_head2` / `pred_head3` / `pred_head4` / `pred_head5` |
| Edge prediction | `edge_head` |

---

## Dataset Organization

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
    └── NC4K/
        ├── Imgs/
        ├── GT/
        ├── Depth/
        └── Edge/
```

---

## Model

```python
from Model.RTSRNet import RTSRNet

model = RTSRNet(
    pretrained=True,
    ablation_mode="full",
)
```

### Load Trained Weights

```python
import torch
from Model.RTSRNet import RTSRNet, extract_state_dict

model = RTSRNet(pretrained=False, ablation_mode="full")
checkpoint = torch.load("RTSRNet-E.pth", map_location="cpu")
state_dict = extract_state_dict(checkpoint, use_ema=True)
model.load_state_dict(state_dict, strict=True)
model.eval()
```

---

## Training

```bash
python train.py \
  --dataset_dir /path/to/DATA_ROOT \
  --ablation_mode full \
  --epochs 180 \
  --batch_size 16 \
  --trainsize 384 \
  --amp
```

Main settings:

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Epochs | 180 |
| Batch size | 16 |
| Input size | 384×384 |
| Initial learning rate | 3×10^-4 |
| Minimum learning rate | 1×10^-6 |
| Weight decay | 1×10^-4 |
| Backbone LR multiplier | 0.2 |
| EMA decay | 0.999 |

---

## Inference

```bash
python inference.py \
  --ckpt /path/to/RTSRNet-E.pth \
  --dataset_dir /path/to/DATA_ROOT \
  --datasets CAMO COD10K NC4K \
  --ablation_mode full
```

---

## Evaluation

```bash
python evaluate.py \
  --ckpt /path/to/RTSRNet-E.pth \
  --dataset_dir /path/to/DATA_ROOT \
  --datasets CAMO COD10K NC4K \
  --save_json
```

Metrics:

- **Sα**: structure measure
- **Eφad**: adaptive enhanced-alignment measure
- **Fβω**: weighted F-measure
- **M**: mean absolute error

---

## Internal Response Visualization

```bash
python analyze_internal_responses.py --help
```

This script supports the supplementary qualitative analysis of SGDEM and TGMM internal responses.

---

## Citation

If you find RTSRNet useful in your research, please cite our work:

```bibtex
@misc{song2026rtsrnet,
  title={RTSRNet: Residual Target-Sensitive Refinement Network for Depth-Assisted Camouflaged Object Detection},
  author={Song, Tan and Li, Jinbao},
  year={2026},
  note={Manuscript submitted to IEEE Transactions on Multimedia (TMM)}
}
```

After acceptance/publication, replace this entry with the official IEEE Xplore BibTeX record and DOI.

---

## Contact

- **Tan Song:** 1223114@s.hlju.edu.cn
- **Jinbao Li:** lijinb@sdas.org

---

## Acknowledgements

We thank the authors of CAMO, COD10K, NC4K, Depth Anything V2, and the open-source COD community for their datasets and implementations.

## License

Please add the intended open-source license before making the repository public.
