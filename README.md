# RTSRNet

## Residual Target-Sensitive Refinement Network for Depth-Assisted Camouflaged Object Detection

**Tan Song, Jinbao Li**

> Manuscript submitted to **IEEE Transactions on Multimedia (TMM)**.

RTSRNet is a depth-assisted camouflaged object detection framework that formulates refinement-stage cue utilization as **target-sensitive cue refinement**. Instead of directly treating shallow detail and depth-derived geometry as uniformly reliable content, RTSRNet uses target-related context to regulate how ambiguous cues update task representations.

The framework contains two main modules:

- **SGDEM — Semantic-Geometric Detail Enhancement Module:** establishes a semantic-guided reference for ambiguous shallow responses and combines semantic discrepancy with depth-derived geometry to refine shallow representations.
- **TGMM — Target-Guided Geometry Modulation Module:** transforms raw depth-derived geometry into target-aware geometry and uses it to progressively refine decoder representations.

---

## News

- **2026:** RTSRNet manuscript submitted to **IEEE Transactions on Multimedia (TMM)**.
- Code, trained weights, prepared depth maps, training edge maps, prediction maps, and evaluation scripts are provided for reproducibility.

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

## Quantitative Comparison

Quantitative comparison with state-of-the-art camouflaged object detection methods on the benchmark datasets. The best results are highlighted in bold.

<p align="center">
  <img src="assets/table1.png" width="100%">
</p>

<p align="center">
  <b>Table I.</b> Quantitative comparison with state-of-the-art methods.
</p>

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

The visual comparison shows the complementary effects of SGDEM and TGMM. SGDEM suppresses distracting shallow responses and preserves local structures, while TGMM improves target coherence through target-guided geometry refinement.

---

## Downloads

### Trained Model

| Model | Backbone | Input Size | Download |
|---|---|---:|---|
| RTSRNet-E | EfficientNet-B0 | 384×384 | [Google Drive](https://drive.google.com/file/d/1u7_Jdv6xFtnHYQkn2B2jBnKzlB5ppc2J/view?usp=drive_link) |

### Original RGB Images and Ground-Truth Masks

The original RGB images and ground-truth (GT) masks are **not redistributed in this repository**. Please download them from the official dataset sources and follow the licenses and terms of use of the corresponding datasets.

| Dataset | Usage in RTSRNet | Official Source |
|---|---|---|
| CAMO | Training and testing | [CAMO Official Project Page](https://sites.google.com/view/ltnghia/research/camo) / [Official Download](https://drive.google.com/drive/folders/1h-OqZdwkuPhBvGcVAwmh0f1NGqlH_4B6?usp=drive_link) |
| COD10K | Training and testing | [COD10K / SINet Official Repository](https://github.com/DengPingFan/SINet) |
| NC4K | Testing | [NC4K Official Project Repository](https://github.com/JingZhang617/COD-Rank-Localize-and-Segment) / [Official Download](https://drive.google.com/file/d/1kzpX_U3gbgO9MuwZIWTuRVpiB7V6yrAQ/view?usp=sharing) |

For convenience, the official SINet repository provides the commonly used COD training and testing splits:

- [COD Training Set](https://drive.google.com/file/d/1D9bf1KeeCJsxxri6d2qAC7z6O1X_fxpt/view?usp=sharing)
- [COD Testing Sets](https://drive.google.com/file/d/1QEGnP9O7HbN_2tH999O3HRIsErIVYalx/view?usp=sharing)

RTSRNet is trained using the CAMO and COD10K training data and evaluated on CAMO, COD10K, and NC4K. Please use the same dataset split described in the manuscript when reproducing the reported results.

### Prepared Depth Maps and Edge Maps

To facilitate reproduction of the RGB-D inputs and edge supervision used in our experiments, we provide only the **prepared depth maps** and **training edge maps**. The original RGB images and GT masks should be obtained from the official dataset sources above.

| Resource | Content | Download |
|---|---|---|
| Depth Maps | Prepared train/test depth maps for CAMO, COD10K, and NC4K | [Google Drive](https://drive.google.com/file/d/16gyUM6YsjWGIE5YXqXViJTHfDiWPTohE/view?usp=drive_link) |
| Edge Maps | Edge supervision maps used for training | [Google Drive](https://drive.google.com/file/d/15wf2iEi7u1g3LMxcAHAt_EWl-erddMvS/view?usp=drive_link) |

The default depth maps used in the manuscript are generated offline with **Depth Anything V2**.

> **Note:** The edge-map download should contain only the derived edge supervision maps. It does not include or redistribute the original RGB images or GT masks.

### Prediction Maps / Test Results

| Results | Download |
|---|---|
| RTSRNet-E prediction maps on CAMO, COD10K, and NC4K | [Google Drive](https://drive.google.com/file/d/1SMPq3khh-XdGtE1ztP8npW6vGxyHU6VG/view?usp=drive_link) |

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
├── assets/
│   ├── Fig2_Overview.png
│   ├── table1.png
│   ├── Fig7_Qtt.png
│   └── Fig8_Abla.png
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
├── LICENSE
└── README.md
```

---

## Dataset Preparation

After downloading the original RGB images and GT masks from the official dataset sources, place the prepared depth maps and edge maps into the corresponding directories.

The expected directory structure is:

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
    │   └── Depth/
    ├── COD10K/
    │   ├── Imgs/
    │   ├── GT/
    │   └── Depth/
    └── NC4K/
        ├── Imgs/
        ├── GT/
        └── Depth/
```

The `Imgs/` and `GT/` directories should come from the official datasets. The `Depth/` directories should contain the prepared depth maps provided above. The training `Edge/` directory should contain the provided edge supervision maps.

Please make sure that the RGB image, GT mask, depth map, and training edge map corresponding to the same sample use matching filenames.

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

The generated prediction maps are saved according to the output path configured in the inference script.

---

## Evaluation

```bash
python evaluate.py \
  --ckpt /path/to/RTSRNet-E.pth \
  --dataset_dir /path/to/DATA_ROOT \
  --datasets CAMO COD10K NC4K \
  --save_json
```

The evaluation reports the following metrics:

- **Sα** — structure measure
- **Eφad** — adaptive enhanced-alignment measure
- **Fβω** — weighted F-measure
- **M** — mean absolute error

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

The citation entry will be updated with the official IEEE Xplore record and DOI after publication.

---

## Acknowledgements

We thank the authors of **CAMO**, **COD10K**, **NC4K**, **Depth Anything V2**, and the open-source camouflaged object detection community for making their datasets, models, and implementations available to the research community.

The original datasets remain subject to their respective licenses and terms of use. Users should download CAMO, COD10K, and NC4K from the official sources listed above and cite the corresponding dataset papers when using them.

---

## License

The source code of RTSRNet is released under the **MIT License**. Please see the `LICENSE` file for details.

The MIT License applies only to the RTSRNet source code for which the authors hold the necessary rights. Third-party code, pretrained models, datasets, and other external resources remain subject to their respective licenses and terms of use.
