# HYDRA — Complete Command Reference

**Environment:** WSL2 Ubuntu 22.04 · rtx50_env · RTX 5060 Laptop · CUDA 12.8

---

## Files Changed in This Release

| File | Status | What Changed |
|------|--------|-------------|
| `01b_volumetric_dataset_download.py` | ✅ **REWRITTEN** | Kaggle-only. Removed IXI 25 GB HTTP download that caused corruption. 4 reliable Kaggle datasets. |
| `04_diagnostic_ensemble_training.py` | ✅ **FIXED** | `hydra_core.py` fully inlined — no separate import needed. |
| `05_volumetric_brain_finetune.py` | ✅ **FIXED** | `nibabel.loadsave.load()` (Pylance fix) + `hydra_core` inlined. |
| `06_clinical_diagnostic_interface.py` | ✅ **FIXED** | `fpdf2` XPos/YPos API + `nibabel.loadsave.load()` + `hydra_core` inlined + new dark UI. |
| `07_ensemble_performance_evaluation.py` | ✅ **FIXED** | `hydra_core` inlined. |
| `index.html` | ✅ **NEW** | Professional dark presentation with animated SVG flowchart. |
| All other files | — Unchanged — | `01`, `02`, `03`, `Dockerfile`, etc. are identical to your working versions. |

**DELETE `hydra_core.py`** — it is no longer needed or imported by anything.

---

## Step 1 — Setup (if not done already)

```bash
conda activate rtx50_env
cd ~/brain_fn
```

Install Kaggle credentials:
```bash
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Verify GPU:
```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.__version__)"
# Expected: NVIDIA GeForce RTX 5060 Laptop GPU  and  2.x.x+cu128
```

---

## Step 2 — Install PyTorch for CUDA 12.8 (RTX 5060 — ONCE ONLY)

```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 --upgrade
```

Then install remaining packages:
```bash
pip install -r requirements.txt
```

---

## Step 3 — Run Order

Scripts 01–04 already ran and produced your trained models. They will **auto-skip** on re-run.

```bash
# Already done — will auto-skip:
python 01_neuroimaging_data_acquisition.py
python 02_gatekeeper_model_training.py
python 03_tumor_localization_model_training.py
python 04_diagnostic_ensemble_training.py
```

```bash
# NEW — run these now:

# Download volumetric patient studies (Kaggle only, no HTTP corruption)
python 01b_volumetric_dataset_download.py

# Fine-tune Council on patient studies (4 epochs, head-only)
python 05_volumetric_brain_finetune.py

# Evaluate updated metrics
python 07_ensemble_performance_evaluation.py

# Launch dashboard  →  http://localhost:7860
python 06_clinical_diagnostic_interface.py
```

---

## Skip Logic Summary

| Script | Skips when |
|--------|-----------|
| `01` | `dataset_ensemble/` has ≥ 1000 images |
| `01b` | `dataset_volumetric/tumor/` and `no_tumor/` each have ≥ 15 patient folders |
| `02` | `Gatekeeper_v1.pth` exists and > 1 MB |
| `03` | `runs/detect/tumor_localizer/weights/best.pt` exists and > 1 MB |
| `04` | All three council `.pth` files exist and each > 5 MB |
| `05` | Council weights exist **AND** `Volumetric_Finetune.json` fingerprint matches dataset |

---

## Volumetric Dataset Layout (what 01b creates)

```
dataset_volumetric/
├── tumor/
│   ├── brats_BraTS20_Training_001/   ← JPEG slices from NIfTI
│   │   ├── slice_0000.jpg
│   │   ├── slice_0001.jpg
│   │   └── ...
│   └── lgg_TCGA_CS_4941_19960909/
│       └── slice_0000.jpg ...
└── no_tumor/
    ├── oasis_mild_0001/
    │   └── slice_0000.jpg ...
    └── healthy_0001/
        └── slice_0000.jpg ...
```

You can also add data manually in this format. Run `05` again — it will detect the change via fingerprint.

---

## Git Cleanup Commands

### Remove hydra_core.py from GitHub (no longer needed)
```bash
git rm hydra_core.py
git commit -m "Remove hydra_core.py — all code inlined into individual scripts"
git push origin main
```

### Remove a file from tracking but keep it locally
```bash
git rm --cached <filename>
git commit -m "Untrack <filename>"
git push origin main
```

### Remove large dataset folders from all git history (CAREFUL)
```bash
pip install git-filter-repo
git filter-repo --path dataset_ensemble/ --invert-paths
git filter-repo --path dataset_volumetric/ --invert-paths
git push origin main --force
```

### Update .gitignore and apply to already-tracked files
```bash
# Add to .gitignore first, then:
git rm -r --cached .
git add .
git commit -m "Apply updated .gitignore"
git push origin main
```

---

## Docker (Optional)

```bash
# Build
docker build -t hydra-brain-tumor:latest .

# Run with GPU and model weights
docker compose up

# Or manually
docker run --gpus all \
  -p 7860:7860 \
  -v ~/.kaggle:/root/.kaggle:ro \
  -v $(pwd)/Gatekeeper_v1.pth:/app/Gatekeeper_v1.pth \
  -v $(pwd)/Swin_Council.pth:/app/Swin_Council.pth \
  -v $(pwd)/ConvNext_Council.pth:/app/ConvNext_Council.pth \
  -v $(pwd)/MONAI_Council.pth:/app/MONAI_Council.pth \
  -v $(pwd)/runs:/app/runs \
  hydra-brain-tumor:latest
```

---

## Dashboard Upload Formats

| What you upload | How |
|-----------------|-----|
| DICOM series | Select all `.dcm` files from one patient folder |
| NIfTI volume | Upload one `.nii.gz` or `.nii` file |
| MRI/CT folder | Upload multiple `.jpg` / `.png` slice images |

The dashboard handles up to **300 slices** per patient with uniform downsampling.

---

## Troubleshooting

### "RuntimeError: CUDA error: no kernel image for this device"
RTX 5060 needs CUDA 12.8 wheels. Fix:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128 --upgrade --force-reinstall
```

### "dataset_volumetric not found" when running 05
Run `01b` first. Or add data manually (see layout above).

### Port 7860 already in use
```bash
lsof -ti:7860 | xargs kill -9
python 06_clinical_diagnostic_interface.py
```

### Kaggle download fails
```bash
cat ~/.kaggle/kaggle.json   # should show your username and key
# If empty: cp kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json
```

### MLflow / ClearML warnings during YOLO training
These are harmless. To silence:
```bash
yolo settings mlflow=False
export CLEARML_WEB_HOST=""
```
