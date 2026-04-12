# HYDRA — Clinical Brain Tumour Analysis System

**Research and educational project — not a medical device.**

Multi-stage brain tumour analysis pipeline for MRI, CT, DICOM series, and NIfTI studies.
Handles 200-300 slices per patient with patient-level aggregation.

---

## Quick Status

| Component | Status | Detail |
|-----------|--------|--------|
| Gatekeeper (EfficientNet-B0) | ✅ Trained | 5 epochs, binary Brain/NotBrain |
| Hunter (YOLOv11n) | ✅ Trained | 35 epochs, mAP50 = 0.516 |
| Council (3-branch ensemble) | ✅ Trained | 12 epochs, **98.25% accuracy** |
| Volumetric fine-tuning | ⬇️ Pending | Run `01b` then `05` |
| Dashboard | ✅ Ready | `python 06_clinical_diagnostic_interface.py` |

---

## Run Order

```bash
conda activate rtx50_env
cd ~/brain_fn

# Already completed (will auto-skip):
python 01_neuroimaging_data_acquisition.py
python 02_gatekeeper_model_training.py
python 03_tumor_localization_model_training.py
python 04_diagnostic_ensemble_training.py

# NEW — run these:
python 01b_volumetric_dataset_download.py   # Download BraTS/IXI/OASIS
python 05_volumetric_brain_finetune.py      # Fine-tune on patient studies
python 07_ensemble_performance_evaluation.py

# Launch dashboard:
python 06_clinical_diagnostic_interface.py
# → http://localhost:7860
```

See **COMMANDS.md** for the complete reference including Git cleanup commands and troubleshooting.

---

## Architecture

```
Patient Scan (DICOM / NIfTI / MRI / CT)
        ↓
[Gatekeeper] EfficientNet-B0  ← rejects non-brain inputs
        ↓
[Hunter]     YOLOv11n         ← tumour bounding box localisation
        ↓
[Council]    Weighted vote (per slice, aggregated across study)
   ├── SwinV2-Tiny      40%   global texture
   ├── ConvNeXtV2-Nano  30%   spatial stability
   └── MONAI Swin-UNETR 30%   clinical domain
        ↓
Verdict + Grad-CAM + PDF Report
```

---

## Council Results (12 epochs, 2005 validation samples)

| Class | F1 | Accuracy |
|-------|----|----------|
| Glioma | 0.9907 | 99.07% |
| Meningioma | 0.9560 | 95.74% |
| NoTumor | 0.9993 | 99.86% |
| Pituitary | 0.9858 | 98.86% |
| Tumor Generic | 0.9599 | 95.67% |
| **Overall** | **0.9783** | **98.25%** |

---

## Dataset Layout

```
dataset_ensemble/          ← Base 2-D training data (already downloaded)
├── Glioma/
├── Meningioma/
├── NoTumor/
├── Pituitary/
└── Tumor_Generic/

dataset_volumetric/        ← Patient-level volumetric data (NEW — run 01b)
├── tumor/
│   ├── patient_001/       ← image folder OR
│   └── patient_001.nii.gz ← NIfTI volume
└── no_tumor/
    ├── patient_101/
    └── patient_101.nii.gz
```

---

## Environment

- **OS**: WSL2 Ubuntu 22.04
- **GPU**: RTX 5060 Laptop (8 GB VRAM)
- **CUDA**: 12.8 (Blackwell / SM_120)
- **Python**: 3.11
- **Conda env**: `rtx50_env`
- **PyTorch**: `torch-2.9.1+cu128`

---

## Files

| File | Purpose |
|------|---------|
| `00_environment_setup.sh` | One-time WSL2 + conda setup |
| `01_neuroimaging_data_acquisition.py` | Download base Kaggle 2-D data |
| `01b_volumetric_dataset_download.py` | Download BraTS/IXI/OASIS volumetric data |
| `02_gatekeeper_model_training.py` | Train EfficientNet-B0 gatekeeper |
| `03_tumor_localization_model_training.py` | Train YOLOv11n localizer |
| `04_diagnostic_ensemble_training.py` | Train 3-branch Council |
| `05_volumetric_brain_finetune.py` | Fine-tune Council on patient studies |
| `06_clinical_diagnostic_interface.py` | Gradio dashboard |
| `07_ensemble_performance_evaluation.py` | Standalone evaluation |
| `hydra_core.py` | Shared model adapters + helpers |
| `brain-tumor.yaml` | YOLO dataset config |
| `COMMANDS.md` | Complete command reference |
| `index.html` | Presentation + flowchart |

---

## License

See LICENSE. Research and educational use only.
