***

# Photovoltaic Panel Segmentation using Satellite Imagery

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Accurate detection and segmentation of photovoltaic (PV) installations from high-resolution aerial imagery using a Hybrid Ensemble of CNNs and Vision Transformers.**

## 📄 Abstract
Accurate estimation of solar energy adoption requires robust automated mapping tools. This project benchmarks **12 segmentation architectures**—spanning classical CNNs (U-Net, FCN), modern Transformers (SegFormer, Mask2Former), and Foundation Models (SAM2, InternImage)—on high-resolution satellite imagery.

We address the challenge of **cross-domain generalization** by training on dataset from France and evaluating on a self-annotated dataset from California. Our final solution utilizes a **Weighted Hybrid Ensemble** combining the strengths of InternImage, HRNet, SegFormer, and PSPNet, achieving an **mIoU of 82.4%** on out-of-distribution test data.

---

## 👥 Team Members

| Name | Email | Role / Focus |
| :--- | :--- | :--- |
| **Dharmik Chellappa Naicker** | dnaicker@uci.edu | DMNet, SAM2, Report |
| **Yash Makarand Deole** | ydeole@uci.edu | HRNet-Seg, InternImage-B, Masks |
| **Shrushti Nikhil Mehta** | shrushtm@uci.edu | PSPNet, MaskDINO, Hybrid Model |
| **Aiden Wan** | zizhenw8@uci.edu | FCN, SegFormer, Labeling |
| **Shengjie Zhang** | shenz18@uci.edu | DeepLabV3+, Mask2Former, Labeling |
| **Yancheng Huang** | yancheh2@uci.edu | U-Net, Swin-UNet, Posters |

---

## 📊 Dataset

We utilized two distinct datasets to test generalization capabilities:

1.  **Training Domain (France):**
    *   Source: BDPV-Google subset (Jiang et al., 2021).
    *   Size: 8,019 high-resolution images (6,415 Train / 1,604 Val).
    *   Characteristics: Highly imbalanced (most masks <2% image area).
2.  **Testing Domain (California - Out of Distribution):**
    *   Source: U.S. Solar Photovoltaic Database (USPVDB) + Google Static API.
    *   Size: 765 images (Manually annotated by the team using AnyLabeling).
    *   Challenge: Different roof architectures (asphalt vs. clay tile), different atmospheric conditions, and sensor angles.

---

## 🧠 Model Architectures

We evaluated the evolution of segmentation models over the last decade:

*   **Classical CNNs:** FCN, U-Net, PSPNet, DeepLab v3+, HRNet-Seg, DMNet.
*   **Transformers:** SegFormer (MiT-B2), Swin-UNet, Mask2Former, MaskDINO.
*   **Foundation Models:** SAM2 (Segment Anything 2), InternImage-B.

### The Hybrid Ensemble (Ours)
To tackle the trade-off between boundary precision and texture recognition, we implemented a weighted voting ensemble:

| Model | Weight | Role |
| :--- | :---: | :--- |
| **InternImage-B** | **0.30** | **Micro-Cracks & Detail:** Captures fine structural anomalies via deformable convolutions. |
| **HRNet-Seg** | **0.25** | **Boundaries:** Preserves high-res spatial info for sharp grid edges. |
| **SegFormer** | **0.25** | **Texture/Shading:** Transformer attention handles shadows and dirt accumulation. |
| **PSPNet** | **0.20** | **Global Context:** Pools features to identify large panel regions. |

**Fusion Logic:** $\text{Pixel Class} = \text{argmax}(\sum (\text{ModelMap}_i \times \text{Weight}_i))$

---

## 📈 Results

Evaluation was performed on the **held-out California Test Set**.

| Model | mIoU | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| PSPNet (Baseline) | 71.2% | 68.5% | 84.1% | 75.5% |
| SegFormer | 76.4% | 74.2% | 79.5% | 76.8% |
| HRNet-Seg | 78.1% | **83.5%** | 76.2% | 79.7% |
| InternImage-B | 79.8% | 80.1% | 85.3% | 82.6% |
| **Hybrid Ensemble** | **82.4%** | 82.9% | **88.1%** | **85.4%** |

**Key Findings:**
*   The Hybrid model improved Recall by **~12%** over HRNet alone.
*   InternImage provided the strongest single-model performance.
*   The ensemble successfully bridged the domain gap between French slate roofs and Californian asphalt shingles.

---

## 🛠️ Repository Structure

```bash
.
├── data/
│   ├── france_train/      # BDPV subset
│   ├── california_test/   # Self-annotated test set
│   └── preprocessing/     # Scripts for cropping and normalization
├── models/
│   ├── definitions/       # PyTorch model definitions
│   ├── weights/           # Pre-trained weights (gitignored)
│   └── ensemble.py        # Logic for weighted voting
├── notebooks/
│   ├── 01_Exploratory_Data_Analysis.ipynb
│   ├── 02_Training_Loop.ipynb
│   └── 03_Evaluation_Metrics.ipynb
├── results/
│   ├── plots/             # Loss curves, IoU charts
│   └── predictions/       # Generated masks
└── requirements.txt
```

## 🚀 Getting Started

### Prerequisites
*   Linux (Ubuntu 20.04+)
*   Python 3.8+
*   CUDA 11.x / 12.x
*   GPUs: Recommended 11GB+ VRAM for Transformer models.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/solar-panel-segmentation.git
    cd solar-panel-segmentation
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Key libraries: `torch`, `torchvision`, `mmsegmentation`, `transformers`, `opencv-python`)*

3.  **Run Inference (Hybrid Model):**
    ```bash
    python scripts/inference_ensemble.py \
        --input_dir data/california_test/images \
        --weights_dir models/weights \
        --output_dir results/predictions
    ```

---

## 🔮 Future Work
*   **Knowledge Distillation:** Compressing the heavy hybrid model into a lightweight real-time network.
*   **Multimodal Fusion:** Integrating Digital Surface Models (DSM) / LiDAR to distinguish flat skylights from tilted solar panels.
*   **Foundation Training:** Pre-training backbones (MAE/DINO) specifically on unlabeled remote sensing data rather than ImageNet.

---

## 📚 References
1.  **InternImage:** Wang et al., CVPR 2023.
2.  **HRNet:** Wang et al., TPAMI 2020.
3.  **BDPV Dataset:** Jiang et al., ESSD 2021.
4.  **SAM2:** Ravi et al., arXiv 2024.
5.  **FCN (Fully Convolutional Networks):** Long et al., CVPR 2015.
6.  **SegFormer:** Xie et al., NeurIPS 2021.
7.  **AnyLabeling:** [GitHub Repository](https://anylabeling.nrl.ai/)
8.  **InternImage:** Cao, H., et al. 2022

---

*This project was completed as part of the Final Project for [Course Name] at University of California, Irvine.*
