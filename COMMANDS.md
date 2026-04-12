# HYDRA — Complete Command Reference

Everything you need to run the project from scratch, skip already-completed steps,
and manage your repository. All commands assume WSL2 Ubuntu, `rtx50_env` conda environment,
and the project folder at `~/brain_fn/` (adjust if you placed it elsewhere).

---

## Table of Contents

1. [Environment Setup (run once)](#1-environment-setup)
2. [Git — Delete Old Files from GitHub](#2-git-cleanup)
3. [Run Order — Full Pipeline](#3-full-pipeline-run-order)
4. [Skip Logic — What Gets Skipped](#4-skip-logic)
5. [New Volumetric Datasets](#5-volumetric-datasets)
6. [Docker (Optional)](#6-docker)
7. [Dashboard](#7-dashboard)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Environment Setup

> Run this **once** on your machine. Skip if `rtx50_env` is already configured.

```bash
cd ~/brain_fn
chmod +x 00_environment_setup.sh
bash 00_environment_setup.sh
```

After setup, always activate the environment first:

```bash
conda activate rtx50_env
```

**Manual PyTorch install (if setup script skipped it):**

```bash
conda activate rtx50_env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --upgrade
```

**Verify GPU:**

```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.__version__)"
```

Expected output: `NVIDIA GeForce RTX 5060 Laptop GPU` and a version ending in `+cu128`.

---

## 2. Git Cleanup

### Delete specific files from GitHub remote AND local

```bash
cd ~/brain_fn

# Remove a specific file from git tracking (keeps local copy)
git rm --cached <filename>
git commit -m "Remove <filename> from tracking"
git push origin main

# Remove a specific file from git tracking AND delete locally
git rm <filename>
git commit -m "Delete <filename>"
git push origin main
```

### Delete entire folder from GitHub remote

```bash
git rm -r --cached dataset_ensemble/
git commit -m "Remove dataset from git (too large for GitHub)"
git push origin main
```

### Delete ALL untracked large files from local disk (careful!)

```bash
# Preview what would be deleted
git clean -n -d

# Actually delete untracked files
git clean -fd
```

### Force push after rewriting history (removes large files from all commits)

```bash
# Install git-filter-repo first:
pip install git-filter-repo

# Remove a file from ALL git history:
git filter-repo --path dataset_ensemble/ --invert-paths
git push origin main --force
```

### .gitignore — update to prevent re-adding large files

The provided `.gitignore` already excludes datasets, model weights, and runs.
To apply it retroactively to already-tracked files:

```bash
git rm -r --cached .
git add .
git commit -m "Apply updated .gitignore"
git push origin main
```

---

## 3. Full Pipeline Run Order

> All commands from inside `~/brain_fn/` with `rtx50_env` active.
> Each script has built-in skip logic — safe to re-run at any time.

```bash
conda activate rtx50_env
cd ~/brain_fn
```

### STEP 1 — Download base 2-D datasets (Kaggle MRI/CT)
> **SKIP THIS** if `dataset_ensemble/` already has images (it does from your previous run).

```bash
python 01_neuroimaging_data_acquisition.py
```

### STEP 2 — Download new volumetric datasets (BraTS / IXI / OASIS)
> **NEW** — Downloads patient-level NIfTI/DICOM data for fine-tuning.

```bash
python 01b_volumetric_dataset_download.py
```

### STEP 3 — Train Gatekeeper (EfficientNet-B0, binary)
> **SKIP THIS** — `HYDRA_Gatekeeper_v1.pth` already exists from your previous run.

```bash
python 02_gatekeeper_model_training.py
```

### STEP 4 — Train YOLO Localizer (Hunter)
> **SKIP THIS** — `runs/detect/hydra_tumor_localizer/weights/best.pt` already exists.

```bash
python 03_tumor_localization_model_training.py
```

### STEP 5 — Train Diagnostic Council (3-branch ensemble)
> **SKIP THIS** — All three `.pth` files already exist from your 12-epoch run (98.25% accuracy).

```bash
python 04_diagnostic_ensemble_training.py
```

### STEP 6 — Volumetric Fine-Tuning (NEW — run this!)
> Adapts existing Council weights to patient-level volumetric data.
> **Requires volumetric dataset** from Step 2.

```bash
python 05_volumetric_brain_finetune.py
```

### STEP 7 — Evaluate Ensemble Performance

```bash
python 07_ensemble_performance_evaluation.py
```

### STEP 8 — Launch Clinical Dashboard

```bash
python 06_clinical_diagnostic_interface.py
```

Then open: **http://localhost:7860**

---

## 4. Skip Logic

Every script checks before running. Here is what gets skipped automatically:

| Script | Skip Condition |
|--------|---------------|
| `01_neuroimaging_data_acquisition.py` | `dataset_ensemble/` has ≥ 1000 images |
| `01b_volumetric_dataset_download.py` | `dataset_volumetric/tumor/` and `no_tumor/` each have ≥ 20 studies |
| `02_gatekeeper_model_training.py` | `HYDRA_Gatekeeper_v1.pth` exists and is > 1 MB |
| `03_tumor_localization_model_training.py` | `runs/detect/hydra_tumor_localizer/weights/best.pt` exists and is > 1 MB |
| `04_diagnostic_ensemble_training.py` | All 3 council `.pth` files exist and each is > 5 MB |
| `05_volumetric_brain_finetune.py` | Council weights exist AND dataset fingerprint unchanged |

**Your current skip status** (based on your previous training run):

- ✅ Script 01 — SKIP (data downloaded)
- ⬇️ Script 01b — **RUN** (new volumetric data needed)
- ✅ Script 02 — SKIP (Gatekeeper trained, 5 epochs, converged)
- ✅ Script 03 — SKIP (Hunter trained, 35 epochs, mAP50=0.516)
- ✅ Script 04 — SKIP (Council trained, 12 epochs, **98.25% accuracy**)
- ⬇️ Script 05 — **RUN** (fine-tune on new volumetric data)
- ▶️ Script 07 — Run to see updated metrics
- ▶️ Script 06 — Run to use the dashboard

---

## 5. Volumetric Datasets

### What gets downloaded by `01b_volumetric_dataset_download.py`

| Dataset | Source | Type | Studies | Label |
|---------|--------|------|---------|-------|
| BraTS 2020 | Kaggle | NIfTI | 369 patients | tumor |
| LGG Segmentation | Kaggle | Image folders | 110 patients | tumor |
| IXI T1 | brain-development.org | NIfTI | 581 subjects | no_tumor |
| Healthy supplement | Kaggle | Images | varies | no_tumor |

### If downloads fail — manual placement

Place studies in this structure and rerun `05_volumetric_brain_finetune.py`:

```
dataset_volumetric/
├── tumor/
│   ├── patient_001/          ← folder of slice images (.jpg/.png)
│   │   ├── slice_0001.jpg
│   │   └── ...
│   ├── patient_002.nii.gz    ← NIfTI volume
│   └── patient_003/          ← DICOM series folder
│       ├── IM-0001-0001.dcm
│       └── ...
└── no_tumor/
    ├── patient_101/
    └── patient_102.nii.gz
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.nii`, `.nii.gz`, `.dcm`

### Manual dataset links

| Dataset | URL | Registration |
|---------|-----|-------------|
| BraTS 2020 | https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation | Kaggle |
| LGG | https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation | Kaggle |
| IXI T1 | http://brain-development.org/ixi-dataset/ | Free, direct download |
| OASIS-3 | https://www.oasis-brains.org | Free registration |
| TCGA-GBM | https://www.cancerimagingarchive.net/collection/tcga-gbm/ | Free registration |

---

## 6. Docker (Optional)

Build and run the dashboard inside Docker (no conda needed on target machine):

```bash
cd ~/brain_fn

# Build image
docker build -t hydra-brain-tumor-ai:latest .

# Run with GPU + model weight volumes
docker compose up

# Or run manually
docker run --gpus all \
  -p 7860:7860 \
  -v ~/.kaggle:/root/.kaggle:ro \
  -v $(pwd)/HYDRA_Gatekeeper_v1.pth:/app/HYDRA_Gatekeeper_v1.pth \
  -v $(pwd)/HYDRA_Swin_Council.pth:/app/HYDRA_Swin_Council.pth \
  -v $(pwd)/HYDRA_ConvNext_Council.pth:/app/HYDRA_ConvNext_Council.pth \
  -v $(pwd)/HYDRA_MONAI_Council.pth:/app/HYDRA_MONAI_Council.pth \
  -v $(pwd)/runs:/app/runs \
  hydra-brain-tumor-ai:latest
```

Open: **http://localhost:7860**

---

## 7. Dashboard

### Start

```bash
conda activate rtx50_env
cd ~/brain_fn
python 06_clinical_diagnostic_interface.py
```

### What it accepts

The dashboard handles 200-300 slices per patient via uniform downsampling:

| Upload type | How to upload |
|-------------|--------------|
| Single MRI/CT image | Upload one `.jpg` or `.png` |
| DICOM series | Upload all `.dcm` files from one patient folder |
| NIfTI volume | Upload one `.nii.gz` or `.nii` file |
| Pre-sliced folder | Upload multiple `.jpg`/`.png` files from one patient |

### Output

- **Diagnostic Report** tab: Structured markdown report with verdict, per-class probabilities, branch stability
- **Visual Review** tab: Grad-CAM saliency map + probability bar chart
- **Export** tab: Downloadable PDF clinical report

---

## 8. Troubleshooting

### CUDA not available after install

```bash
conda activate rtx50_env
python -c "import torch; print(torch.cuda.is_available())"
# If False:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --upgrade --force-reinstall
```

### "RuntimeError: CUDA error: no kernel image is available for execution on the device"

This means PyTorch was built for an older CUDA version. The RTX 5060 (Blackwell) requires:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128 --upgrade
```

### GradScaler / autocast deprecation warnings

These are fixed in the new code. If you see them, ensure you're using the **new files** from this release (not the originals from BB.zip).

### "kaggle.json not found"

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### "dataset_ensemble not found" when running 02

You need to run 01 first (or copy your existing dataset to `dataset_ensemble/`):

```bash
# If data is at ~/backup/datasets/
cp -r ~/backup/datasets/brain-tumor/* ~/brain_fn/dataset_ensemble/
```

### MLflow FutureWarning about filesystem backend

This is a harmless warning. To silence it:

```bash
yolo settings mlflow=False
```

### ClearML warning during YOLO training

Also harmless. To silence:

```bash
export CLEARML_WEB_HOST=""
```

Or add it to `~/.bashrc`.

### Port 7860 already in use

```bash
# Find and kill the process
lsof -i :7860
kill -9 <PID>

# Or run on a different port
python 06_clinical_diagnostic_interface.py --server-port 7861
```
