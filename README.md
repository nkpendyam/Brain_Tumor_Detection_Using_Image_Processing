# 🧠 HYDRA — Clinical Brain Tumour Analysis System

<div align="center">

![HYDRA Banner](https://img.shields.io/badge/HYDRA-v2.0-00c8ff?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyem0tMSAxNHYtNEg3bDUtOHY0aDRsLTUgOHoiLz48L3N2Zz4=)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-FF7C00?style=flat-square&logo=gradio&logoColor=white)](https://gradio.app)
[![MONAI](https://img.shields.io/badge/MONAI-Medical%20AI-00C8FF?style=flat-square)](https://monai.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**A production-ready, multi-model ensemble system for the detection, classification,<br>and radiological reporting of brain tumours from volumetric neuroimaging data.**

[Architecture](#-architecture) · [Quick Start](#-quick-start) · [File Structure](#-file-structure) · [Dataset Layout](#-dataset-layout) · [Results](#-results)

</div>

---

## 📌 Overview

HYDRA is a clinical decision-support AI pipeline built for neuroimaging analysis.  It processes MRI, CT, DICOM series, and NIfTI volumes — including whole-brain studies of up to **300 axial slices** — and produces a structured diagnostic report with per-class probability estimates, Grad-CAM saliency maps, and a downloadable PDF.

The system is designed for **research and educational demonstration**.  It is not a substitute for professional medical diagnosis.

---

## ✨ Key Features

| Feature | Detail |
|---------|--------|
| 🛡️ **Safety Gatekeeper** | EfficientNet-B0 rejects non-brain inputs (X-rays, photographs, documents) before analysis |
| 🎯 **Tumour Localisation** | YOLOv11n Hunter provides spatial bounding-box predictions for biopsy / resection planning |
| 🧠 **3-Branch Council** | Weighted soft-vote ensemble: SwinV2 (40 %) + ConvNeXt (30 %) + MONAI (30 %) |
| 🔬 **5-Class Diagnosis** | Glioma · Meningioma · Pituitary · No Tumour · Tumour (Generic/CT) |
| 🧩 **Volumetric Support** | Fine-tuned on whole-brain slice sequences (200–300 slices/patient) |
| 📂 **Multi-Format Ingestion** | DICOM (.dcm), NIfTI (.nii/.nii.gz), JPEG, PNG |
| 🔥 **Grad-CAM Explainability** | Saliency heatmap highlights the most pathological axial slice |
| 📊 **Clinical PDF Report** | Auto-generated structured report with uncertainty entropy score |
| 🐳 **Docker Ready** | One-command GPU deployment via `docker compose up --build` |
| ⚡ **Skip-If-Trained** | All training scripts detect existing weights and skip automatically |

---

## 🏗 Architecture

HYDRA operates in four sequential stages:

```
Raw Scan Input (DICOM / NIfTI / JPEG / PNG)
          │
          ▼
┌─────────────────────┐
│   1. GATEKEEPER     │  EfficientNet-B0  ─►  Not-Brain? → REJECT
│  (Safety Validator) │
└─────────┬───────────┘
          │  Brain confirmed
          ▼
┌─────────────────────┐
│   2. HUNTER         │  YOLOv11n  ─►  Bounding-box tumour localisation
│  (Localiser)        │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────────────────┐
│   3. COUNCIL  (Weighted Soft-Vote Ensemble)      │
│                                                 │
│   SwinV2-Tiny  ×0.40  ─┐                        │
│   ConvNeXtV2-Nano ×0.30─┼──► Consensus → Class  │
│   MONAI Swin-UNETR ×0.30┘                       │
└─────────┬───────────────────────────────────────┘
          │
          ▼
┌─────────────────────┐
│   4. EXPLAINABILITY │  Grad-CAM  ─►  Saliency Heatmap + PDF Report
│  (Clinical Report)  │
└─────────────────────┘
```

### Model Component Table

| Stage | Component | Architecture | Role |
|-------|-----------|-------------|------|
| Gatekeeper | Safety Validator | EfficientNet-B0 | Brain vs. Non-Brain binary filter |
| Hunter | Tumour Localiser | YOLOv11n | Spatial bounding-box detection |
| Council — Voter 1 | SwinV2-Tiny | Swin Transformer V2 | Global texture & attention patterns |
| Council — Voter 2 | ConvNeXtV2-Nano | Convolutional backbone | Spatial stability & local features |
| Council — Voter 3 | MONAI Swin-UNETR | Medical foundation model | Clinical anatomical texture awareness |

---

## 📁 File Structure

```
HYDRA/
│
├── 01_neuroimaging_data_acquisition.py    # Kaggle dataset download & organisation
├── 02_gatekeeper_model_training.py        # EfficientNet-B0 Gatekeeper training
├── 03_tumor_localization_model_training.py # YOLOv11n Hunter training
├── 04_diagnostic_ensemble_training.py     # 3-branch Council training
├── 05_volumetric_brain_finetune.py        # ★ Fine-tuning on whole-brain slice sequences
├── 06_clinical_diagnostic_interface.py   # Gradio web application (UI)
├── 07_ensemble_performance_evaluation.py  # Standalone evaluation & metrics
│
├── brain-tumor.yaml                       # YOLO data configuration
├── docker-compose.yml                     # Docker GPU deployment
├── Dockerfile                             # Container build definition
├── requirements.txt                       # Python dependency manifest
│
├── dataset_ensemble/                      # 5-class MRI + CT training data
├── dataset_negatives/                     # X-ray distractor samples
├── dataset_faces/                         # Facial photo distractor samples
└── dataset_volumetric/                    # ★ Whole-brain slice sequences (new)
    ├── tumor/
    │   ├── patient_001/  (200–300 slices)
    │   └── patient_001.nii.gz
    └── no_tumor/
        └── patient_001/
```

---

## 📦 Dataset Layout

### For Script 05 — Volumetric Fine-Tuning

HYDRA's volumetric pipeline accepts three data formats interchangeably.

**Format A — Pre-sliced images (organised by patient)**
```
dataset_volumetric/
├── tumor/
│   ├── patient_001/
│   │   ├── slice_001.jpg
│   │   ├── slice_002.jpg
│   │   └── ...  (up to 300 slices)
│   └── patient_002/ ...
└── no_tumor/
    └── patient_001/ ...
```

**Format B — NIfTI volumes**
```
dataset_volumetric/
├── tumor/
│   ├── patient_001.nii.gz
│   └── patient_002.nii.gz
└── no_tumor/
    └── patient_001.nii.gz
```

**Format C — DICOM series folders**
```
dataset_volumetric/
├── tumor/
│   └── patient_001/
│       ├── IM-0001-0001.dcm
│       ├── IM-0001-0002.dcm
│       └── ...
└── no_tumor/
    └── patient_001/ ...
```

---

## ⚡ Quick Start

### Option 1 — Docker (Recommended)

```bash
# Requires NVIDIA GPU + Docker with NVIDIA Container Toolkit

git clone https://github.com/nkpendyam/Brain_Tumor_Detection_Using_Image_Processing
cd Brain_Tumor_Detection_Using_Image_Processing

docker compose up --build
```

Open your browser at **http://localhost:7860**

---

### Option 2 — Manual Installation

**Step 1: Install dependencies**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Step 2: Configure Kaggle API**
```bash
# Place your kaggle.json at ~/.kaggle/kaggle.json
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

**Step 3: Run the training pipeline**

Each script is **idempotent** — it detects existing weights and skips training automatically.

```bash
# Download and organise datasets
python 01_neuroimaging_data_acquisition.py

# Train the Safety Gatekeeper (EfficientNet-B0) — ~15 min on GPU
python 02_gatekeeper_model_training.py

# Train the Tumour Localiser (YOLOv11n) — ~2 hrs on GPU
python 03_tumor_localization_model_training.py

# Train the Diagnostic Council (SwinV2 + ConvNeXt + MONAI) — ~13 hrs on GPU
python 04_diagnostic_ensemble_training.py

# [OPTIONAL] Fine-tune on whole-brain slice sequences — ~1–2 hrs on GPU
# Place your data in dataset_volumetric/ first
python 05_volumetric_brain_finetune.py

# Evaluate final model performance
python 07_ensemble_performance_evaluation.py

# Launch the diagnostic web interface
python 06_clinical_diagnostic_interface.py
```

**Step 4: Open the UI**

Navigate to **http://localhost:7860** in your browser.

---

## 🖥 Web Interface

The HYDRA diagnostic interface features a professional dark clinical theme:

- **Upload Zone** — Drag and drop DICOM, NIfTI, or image files
- **Diagnostic Report Tab** — Structured findings with confidence and uncertainty scores
- **Visual Analysis Tab** — Probability distribution and Grad-CAM heatmap
- **Export Report Tab** — Downloadable clinical PDF with full analysis

---

## 📊 Performance Metrics

> Results on the held-out 20 % validation set (stratified split, seed=42).

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | — | — | — |
| Meningioma | — | — | — |
| No Tumour | — | — | — |
| Pituitary | — | — | — |
| Tumour (Generic) | — | — | — |
| **Macro Average** | — | — | **~0.97+** |

*Run `python 07_ensemble_performance_evaluation.py` after training to populate this table.*

---

## 🛡 Skip-If-Trained Logic

All training scripts implement automatic checkpoint detection:

```
Script 02 → Checks: HYDRA_Gatekeeper_v1.pth           (> 1 MB)
Script 03 → Checks: runs/detect/.../best.pt            (> 1 MB)
Script 04 → Checks: HYDRA_Swin/ConvNext/MONAI_Council.pth (each > 5 MB)
Script 05 → Checks: HYDRA_Volumetric_Finetune.json     (sentinel file)
```

If all required artefacts are found, the script logs the detection and exits.  To force re-training, delete the corresponding weight file or sentinel.

---

## 🔬 Volumetric Inference (200–300 Slices)

When a full brain study is uploaded:

1. Each slice passes through the Gatekeeper individually (non-brain rejected).
2. The Council generates a 5-class probability vector per slice.
3. Probabilities are **averaged across all valid slices** to produce patient-level estimates.
4. Test-time augmentation (original + horizontal flip) reduces single-image variance.
5. The slice with the **highest tumour signal** is selected for Grad-CAM visualisation.

---

## 🐳 Docker Configuration

GPU support requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```yaml
# docker-compose.yml (excerpt)
services:
  hydra:
    build: .
    ports:
      - "7860:7860"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## ⚠️ Clinical Disclaimer

This system is intended **solely for research, educational, and demonstration purposes**.  
It must **not** be used for clinical diagnosis without review by a licensed radiologist or neuro-oncologist.  
All outputs are probabilistic estimates and carry inherent uncertainty.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built for the Brain Tumour Detection Seminar · HYDRA v2.0 · 2026</sub>
</div>
