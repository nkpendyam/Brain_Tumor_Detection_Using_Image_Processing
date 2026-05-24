# Brain Tumor Detection — AI Clinical Analysis System

> **Research and educational project. Not a certified medical device.**
> All outputs must be reviewed by a qualified radiologist.

A multi-stage deep-learning pipeline for automated brain tumour detection and classification from MRI, CT, DICOM series, and NIfTI volumetric studies.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Model Weights](#model-weights)
- [Performance](#performance)
- [Environment Setup](#environment-setup)
- [Run Order](#run-order)
- [Dataset Layout](#dataset-layout)
- [Dashboard](#dashboard)
- [Docker](#docker)
- [GitHub — Clean Push Guide](#github--clean-push-guide)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

This system combines five specialised AI models into a single clinical pipeline:

1. **Safety Gatekeeper** — rejects non-brain inputs before they reach the diagnostic models
2. **Tumour Localizer (Hunter)** — spatial bounding-box detection with YOLOv11n
3. **Diagnostic Council** — three-branch weighted-voting ensemble for 5-class classification
4. **Volumetric Fine-tuner** — adapts the council to full patient studies (NIfTI/DICOM)
5. **Clinical Dashboard** — Gradio web interface with PDF report export

**Diagnostic Classes:**

| Class | Description |
|-------|-------------|
| Glioma | Aggressive primary brain tumour |
| Meningioma | Slow-growing tumour from meninges |
| No Tumor | Healthy brain tissue |
| Pituitary | Tumour of the pituitary gland |
| Tumor (Generic / CT) | Unspecified tumour on CT scan |

---

## Architecture

```
Patient Scan (DICOM / NIfTI / MRI / CT)
        │
        ▼
┌─────────────────────────────────────────┐
│  Ingest & Normalise                     │
│  • Parse DICOM / NIfTI / image formats  │
│  • Skull-strip via Otsu thresholding    │
│  • Uniform downsample → ≤ 300 slices   │
└──────────────────┬──────────────────────┘
                   │
        ▼
┌─────────────────────────────────────────┐
│  Safety Gatekeeper (EfficientNet-B0)    │
│  • Binary: Brain vs NotBrain            │
│  • Rejects non-brain inputs at 70%      │
│  Weight: Gatekeeper_v1.pth             │
└──────────────────┬──────────────────────┘
                   │  (per valid slice)
        ▼
┌─────────────────────────────────────────┐
│  Diagnostic Council — Weighted Vote     │
│                                         │
│  ┌───────────┐  ┌───────────┐  ┌──────┐│
│  │ SwinV2    │  │ ConvNeXt  │  │MONAI ││
│  │ Tiny (40%)│  │ Nano (30%)│  │ (30%)││
│  │Swin_5C.pth│  │ConvNext.. │  │MONAI.││
│  └───────────┘  └───────────┘  └──────┘│
│                                         │
│  P = 0.4·P_swin + 0.3·P_conv + 0.3·P_m│
└──────────────────┬──────────────────────┘
                   │
        ▼
┌─────────────────────────────────────────┐
│  Patient-Level Aggregation              │
│  Mean probabilities across all slices   │
└──────────────────┬──────────────────────┘
                   │
        ▼
┌─────────────────────────────────────────┐
│  Outputs                                │
│  • Diagnostic report (markdown)         │
│  • Grad-CAM saliency heatmap            │
│  • Downloadable PDF clinical report     │
└─────────────────────────────────────────┘
```

**Council Branch Details:**

| Branch | Architecture | Vote | Strength |
|--------|-------------|------|----------|
| SwinV2-Tiny | Shifted-window vision transformer | 40% | Global texture, long-range dependencies |
| ConvNeXtV2-Nano | Modernised convolutional network | 30% | Spatial stability, rotation invariance |
| MONAI Swin-UNETR | Medical-domain pretrained encoder | 30% | Clinical anatomical intuition, Grad-CAM |

---

## Model Weights

All weights are stored in the project root directory:

| File | Model | Size | Purpose |
|------|-------|------|---------|
| `Gatekeeper_v1.pth` | EfficientNet-B0 | ~20 MB | Brain / NotBrain classifier |
| `Gatekeeper_Clinical.pth` | EfficientNet-B0 | ~20 MB | Fine-tuned clinical version |
| `Swin_5C.pth` | SwinV2-Tiny | ~110 MB | Council branch 1 |
| `ConvNext_5C.pth` | ConvNeXtV2-Nano | ~50 MB | Council branch 2 |
| `MONAI_5C.pth` | MONAI Swin-UNETR | ~95 MB | Council branch 3 |

> Weights are excluded from git (see `.gitignore`). Store them on cloud storage or Git LFS.

---

## Performance

Results on 20% held-out validation set (2,005 samples, `random_state=42`):

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Glioma | 99.07% | 99.07% | 0.9907 | 324 |
| Meningioma | 95.45% | 95.74% | 0.9560 | 329 |
| No Tumor | 100.00% | 99.86% | 0.9993 | 700 |
| Pituitary | 98.31% | 98.86% | 0.9858 | 352 |
| Tumor (Generic / CT) | 96.31% | 95.67% | 0.9599 | 300 |
| **Overall** | **97.83%** | **97.84%** | **0.9783** | **2005** |

**Overall Accuracy: 98.25%** | Trained: 12 epochs | GPU: RTX 5060 Laptop | CUDA: 12.8

---

## Environment Setup

**Requirements:**
- WSL2 Ubuntu 22.04
- NVIDIA RTX 5060 (or any CUDA-capable GPU)
- CUDA 12.8 (Blackwell / sm_120 compatible)
- Miniforge3 / Conda
- Python 3.11

### Step 1 — Create conda environment

```bash
conda activate rtx50_env
# or create fresh:
conda create -n btd_env python=3.11 -y
conda activate btd_env
```

### Step 2 — Install PyTorch (CUDA 12.8 — RTX 5060 requires cu128 wheels)

```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 --upgrade
```

### Step 3 — Install remaining packages

```bash
pip install -r requirements.txt
```

### Step 4 — Set up Kaggle credentials

```bash
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Upgrade Kaggle CLI to avoid download corruption bug
pip install kaggle --upgrade
```

### Step 5 — Verify GPU

```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.__version__)"
# Expected: NVIDIA GeForce RTX 5060 Laptop GPU   2.x.x+cu128
```

---

## Run Order

```bash
conda activate rtx50_env
cd ~/brain_fn
```

> Scripts 01–04 auto-skip if their outputs already exist. Re-running is always safe.

```bash
# ── Download base 2-D training data (skip if dataset_ensemble/ already populated)
python 01_neuroimaging_data_acquisition.py

# ── Download volumetric patient studies for fine-tuning
python 01b_volumetric_dataset_download.py

# ── Train Gatekeeper (5 epochs, ~10 min)
python 02_gatekeeper_model_training.py

# ── Train YOLO Localizer (35 epochs, ~45 min)
python 03_tumor_localization_model_training.py

# ── Train Diagnostic Council (12 epochs, ~13 hrs — skip if weights exist)
python 04_diagnostic_ensemble_training.py

# ── Fine-tune on volumetric patient studies (4 epochs, ~2 hrs)
python 05_volumetric_brain_finetune.py

# ── Evaluate final performance
python 07_ensemble_performance_evaluation.py

# ── Launch clinical dashboard
python 06_clinical_diagnostic_interface.py
# Open: http://localhost:7860

# Optional remote tunnel; keep disabled for clinical/private scans
BTD_SHARE=1 python 06_clinical_diagnostic_interface.py
```

### Skip Logic Summary

| Script | Skips when |
|--------|-----------|
| `01` | `dataset_ensemble/` has ≥ 1000 images |
| `01b` | `dataset_volumetric/tumor/` and `no_tumor/` have ≥ 10 patient folders each |
| `02` | `Gatekeeper_v1.pth` exists and > 1 MB |
| `03` | `runs/detect/tumor_localizer/weights/best.pt` exists and > 1 MB |
| `04` | `Swin_5C.pth`, `ConvNext_5C.pth`, `MONAI_5C.pth` all exist and each > 5 MB |
| `05` | Council weights exist AND `Volumetric_Finetune.json` fingerprint matches |

---

## Dataset Layout

### Base training data (`dataset_ensemble/`)

```
dataset_ensemble/
├── Glioma/          (1,621 images — MRI)
├── Meningioma/      (1,645 images — MRI)
├── NoTumor/         (3,500 images — MRI + CT)
├── Pituitary/       (1,757 images — MRI)
└── Tumor_Generic/   (1,500 images — CT)
```

Sources: Kaggle `masoudnickparvar/brain-tumor-mri-dataset` + `ahmedhamada0/brain-tumor-detection`

### Volumetric fine-tuning data (`dataset_volumetric/`)

```
dataset_volumetric/
├── tumor/
│   ├── lgg_TCGA_CS_4941_19960909/   ← JPEG slices per patient
│   │   ├── slice_0000.jpg
│   │   └── ...
│   └── sartaj_0001/
└── no_tumor/
    ├── healthy_0001/
    └── ...
```

Sources: `mateuszbuda/lgg-mri-segmentation` + `navoneel/brain-mri-images-for-brain-tumor-detection`

### Supported input formats at inference

| Format | Extension | How to upload |
|--------|-----------|--------------|
| DICOM series | `.dcm` | Upload all `.dcm` files from one patient |
| NIfTI volume | `.nii`, `.nii.gz` | Upload single volume file |
| MRI/CT slices | `.jpg`, `.png` | Upload one or more slice images |

---

## Dashboard

```bash
python 06_clinical_diagnostic_interface.py
# Opens at: http://localhost:7860
```

**What it shows:**
- 🟢 / 🔴 Verdict with confidence percentage
- How many slices were uploaded vs how many were analysed
- Per-class probability distribution (5 classes)
- Grad-CAM saliency map on the highest tumour-signal slice
- Downloadable PDF clinical report

**Slice handling:**
- Single image → 1 slice analysed
- DICOM series → sorted by `ImagePositionPatient` / `InstanceNumber`, uniform downsampled to ≤ 300
- NIfTI → 15% trimmed from each end (skull/noise), ≤ 300 slices extracted
- Result always shows: `N slices uploaded — M analysed`

---

## Docker

Build and run the full system in a container (no conda needed):

```bash
# Build image (uses CUDA 12.8 base)
docker build -t brain-tumor-detection:latest .

# Run with GPU access and model weights
docker compose up

# Or manually
docker run --gpus all \
  -p 7860:7860 \
  -v ~/.kaggle:/root/.kaggle:ro \
  -v $(pwd)/Gatekeeper_v1.pth:/app/Gatekeeper_v1.pth \
  -v $(pwd)/Gatekeeper_Clinical.pth:/app/Gatekeeper_Clinical.pth \
  -v $(pwd)/Swin_5C.pth:/app/Swin_5C.pth \
  -v $(pwd)/ConvNext_5C.pth:/app/ConvNext_5C.pth \
  -v $(pwd)/MONAI_5C.pth:/app/MONAI_5C.pth \
  -v $(pwd)/runs:/app/runs \
  brain-tumor-detection:latest

# Open: http://localhost:7860
```

---

## GitHub — Clean Push Guide

### Delete everything from old repo and push fresh code

```bash
cd ~/brain_fn

# ── Step 1: Remove all old tracked files from git index
git rm -r --cached .

# ── Step 2: Apply new .gitignore (datasets/weights excluded)
git add .gitignore
git commit -m "Apply clean .gitignore — exclude datasets and weights"

# ── Step 3: Add all project code files
git add \
  01_neuroimaging_data_acquisition.py \
  01b_volumetric_dataset_download.py \
  02_gatekeeper_model_training.py \
  03_tumor_localization_model_training.py \
  04_diagnostic_ensemble_training.py \
  05_volumetric_brain_finetune.py \
  06_clinical_diagnostic_interface.py \
  07_ensemble_performance_evaluation.py \
  requirements.txt \
  Dockerfile \
  docker-compose.yml \
  brain-tumor.yaml \
  gatekeeper_class_map.json \
  gatekeeper_classes.json \
  README.md \
  index.html \
  LICENSE

git commit -m "Brain Tumor Detection — clean codebase v1.0"

# ── Step 4: Push to GitHub
git push origin main

# If push is rejected (history mismatch after cleanup):
git push origin main --force
```

### Remove specific old files that slipped through

```bash
# Remove a file from git tracking (keep local copy)
git rm --cached hydra_core.py
git rm --cached Confusion_Matrix.csv
git commit -m "Remove old naming artifacts"
git push origin main

# Remove a large file from ALL git history (permanent)
pip install git-filter-repo
git filter-repo --path dataset_ensemble/ --invert-paths --force
git push origin main --force
```

### Confirm what will be pushed (dry run)

```bash
git status
git diff --cached --name-only
```

---

## Project Structure

```
brain_fn/
├── 01_neuroimaging_data_acquisition.py   # Download base 2-D training data
├── 01b_volumetric_dataset_download.py    # Download patient-level volumetric data
├── 02_gatekeeper_model_training.py       # Train EfficientNet-B0 gatekeeper
├── 03_tumor_localization_model_training.py # Train YOLOv11n tumour localizer
├── 04_diagnostic_ensemble_training.py    # Train 3-branch council
├── 05_volumetric_brain_finetune.py       # Fine-tune on patient studies
├── 06_clinical_diagnostic_interface.py   # Gradio dashboard
├── 07_ensemble_performance_evaluation.py # Standalone evaluation
├── requirements.txt                      # Python dependencies
├── Dockerfile                            # CUDA 12.8 container
├── docker-compose.yml                    # Docker Compose config
├── brain-tumor.yaml                      # YOLO dataset config
├── gatekeeper_class_map.json             # Brain/NotBrain class indices
├── gatekeeper_classes.json               # Class definitions
├── index.html                            # Presentation / flowchart
├── README.md                             # This file
└── LICENSE
```

**Generated at runtime (not committed):**
```
Gatekeeper_v1.pth          ← trained gatekeeper weights
Gatekeeper_Clinical.pth    ← fine-tuned gatekeeper
Swin_5C.pth                ← council branch 1
ConvNext_5C.pth            ← council branch 2
MONAI_5C.pth               ← council branch 3
Volumetric_Finetune.json   ← fine-tune fingerprint sentinel
BrainTumor_Confusion_Matrix.csv
runs/                      ← YOLO training artifacts
dataset_ensemble/          ← base training images
dataset_volumetric/        ← patient study folders
```

---

## License

See [LICENSE](LICENSE). For research and educational use only.
