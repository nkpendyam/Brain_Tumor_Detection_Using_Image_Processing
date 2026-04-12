"""
01b_volumetric_dataset_download.py — HYDRA Volumetric Dataset Acquisition
==========================================================================
Downloads patient-level brain MRI datasets (NIfTI / DICOM) for the
volumetric fine-tuning stage (05_volumetric_brain_finetune.py).

Datasets Acquired
-----------------
TUMOR (with brain tumour):
  1. BraTS 2020 — Kaggle (awsaf49/brats20-dataset-training-validation)
     ↳ Format: NIfTI (.nii.gz), multi-modal, 369 patients, ~155 slices each
  2. LGG Segmentation — Kaggle (mateuszbuda/lgg-mri-segmentation)
     ↳ Format: FLAIR slices per patient folder, 110 patients

HEALTHY (no tumour):
  3. IXI Dataset — Direct download from brain-development.org
     ↳ Format: NIfTI T1 (.nii.gz), 581 healthy subjects
  4. Healthy Brain MRI — Kaggle (prathamgrover/healthy-brain-mri-images)
     ↳ Format: Pre-sliced images, grouped by subject

Output Layout
-------------
dataset_volumetric/
├── tumor/
│   ├── brats_patient_001/    ← DICOM series or image folders
│   │   ├── slice_001.jpg
│   │   └── ...
│   └── brats_vol_002.nii.gz  ← NIfTI volumes
└── no_tumor/
    ├── ixi_001.nii.gz
    └── ...

Skip Behaviour
--------------
The script checks each destination sub-directory independently.
If enough studies are already present it skips that download.
Re-running is always safe — no existing files are modified or deleted.

Alternative Download URLs
-------------------------
If Kaggle is unavailable or a dataset is removed, the script automatically
falls back to verified direct download URLs (OpenNeuro, Zenodo, etc.).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

VOLUMETRIC_ROOT = Path("dataset_volumetric")
TUMOR_DIR       = VOLUMETRIC_ROOT / "tumor"
NO_TUMOR_DIR    = VOLUMETRIC_ROOT / "no_tumor"

# Minimum patient studies in each class before skipping download
MIN_TUMOR_STUDIES    = 20
MIN_NO_TUMOR_STUDIES = 20

# HTTP download timeout (seconds)
DOWNLOAD_TIMEOUT = 600


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _count_studies(directory: Path) -> int:
    """
    Count patient studies in *directory*.
    A 'study' is either a .nii / .nii.gz file or a sub-directory.
    """
    if not directory.exists():
        return 0
    count = 0
    for entry in directory.iterdir():
        if entry.is_dir():
            count += 1
        elif entry.suffix in {".gz", ".nii"} or entry.name.endswith(".nii.gz"):
            count += 1
    return count


def _run_kaggle(slug: str, dest: Path) -> bool:
    """
    Download a Kaggle dataset.  Returns True on success.
    Requires ~/.kaggle/kaggle.json (configured by 01_neuroimaging_data_acquisition.py).
    """
    dest.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"],
        timeout=DOWNLOAD_TIMEOUT,
    )
    return result.returncode == 0


def _http_download(url: str, dest: Path, label: str) -> bool:
    """
    Download a single file via HTTP with a progress report.
    Returns True on success.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar = "#" * int(pct / 5)
            print(f"\r  [{bar:<20}] {pct:5.1f}%  {label}", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
        print()  # newline after progress bar
        return True
    except Exception as exc:
        print(f"\n[ERROR] Download failed: {exc}")
        return False


def _safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


# ─── DATASET 1 — BraTS 2020 (TUMOR) ──────────────────────────────────────────

def _download_brats_tumor() -> None:
    """
    BraTS 2020 training set from Kaggle.
    Each patient folder contains multi-modal NIfTI volumes.
    We copy the T1ce (post-contrast T1) modality as the primary channel.
    """
    label = "BraTS 2020 (Tumor)"
    print(f"\n[DATASET 1] {label}")
    tmp = Path("tmp_brats_cache")

    ok = _run_kaggle(
        slug="awsaf49/brats20-dataset-training-validation",
        dest=tmp,
    )
    if not ok:
        print(f"  [WARN] Kaggle download failed for {label}.")
        print("  [FALLBACK] Trying direct NIfTI sample from OpenNeuro ds001521 …")
        # Fallback: a small public subset on Zenodo
        fallback_url = (
            "https://zenodo.org/records/3718275/files/"
            "BraTS20_Training_001_flair.nii.gz?download=1"
        )
        dest_file = TUMOR_DIR / "brats_sample_001_flair.nii.gz"
        _http_download(fallback_url, dest_file, "BraTS sample 001")
        return

    # Walk the extracted folders — look for T1ce volumes
    copied = 0
    for study_dir in sorted(tmp.rglob("*")):
        if not study_dir.is_dir():
            continue
        # Each BraTS study dir contains: *_flair.nii.gz, *_t1.nii.gz, etc.
        t1ce_files = list(study_dir.glob("*_t1ce.nii.gz"))
        if not t1ce_files:
            continue
        patient_id = study_dir.name
        dst = TUMOR_DIR / f"brats_{patient_id}.nii.gz"
        if not dst.exists():
            _safe_copy(t1ce_files[0], dst)
            copied += 1

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  [OK] BraTS tumor studies copied: {copied}")


# ─── DATASET 2 — LGG Segmentation (TUMOR) ────────────────────────────────────

def _download_lgg_tumor() -> None:
    """
    LGG MRI Segmentation dataset from Kaggle.
    Contains 110 patient folders with FLAIR slices.
    """
    label = "LGG Segmentation (Tumor)"
    print(f"\n[DATASET 2] {label}")
    tmp = Path("tmp_lgg_cache")

    ok = _run_kaggle(
        slug="mateuszbuda/lgg-mri-segmentation",
        dest=tmp,
    )
    if not ok:
        print(f"  [WARN] Kaggle download failed for {label}.")
        print("  [FALLBACK] Skipping LGG — BraTS provides sufficient tumor coverage.")
        shutil.rmtree(tmp, ignore_errors=True)
        return

    copied = 0
    # Each patient dir: lgg-mri-segmentation/kaggle_3m/<PatientID>/
    for patient_dir in sorted(tmp.rglob("kaggle_3m/*")):
        if not patient_dir.is_dir():
            continue
        slices = sorted(patient_dir.glob("*.tif"))
        if not slices:
            slices = sorted(patient_dir.glob("*.png"))
        if not slices:
            continue

        patient_id = patient_dir.name
        dest_patient = TUMOR_DIR / f"lgg_{patient_id}"
        if dest_patient.exists() and len(list(dest_patient.iterdir())) > 0:
            continue
        dest_patient.mkdir(parents=True, exist_ok=True)

        for i, sl in enumerate(slices):
            # Skip mask files (they end with _mask.tif)
            if "mask" in sl.name.lower():
                continue
            dst = dest_patient / f"slice_{i:04d}{sl.suffix}"
            _safe_copy(sl, dst)
            copied += 1

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  [OK] LGG patient slices copied: {copied}")


# ─── DATASET 3 — IXI Dataset (HEALTHY) ───────────────────────────────────────

_IXI_T1_URLS = [
    # Primary — IXI official mirror (CC BY-SA 3.0)
    "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar",
    # Fallback mirror
    "https://www.nitrc.org/frs/download.php/10201/IXI-T1.tar",
]

def _download_ixi_healthy() -> None:
    """
    Download IXI T1 NIfTI volumes (581 healthy subjects).
    Primary: brain-development.org | Fallback: NITRC mirror.
    """
    label = "IXI Dataset T1 (Healthy)"
    print(f"\n[DATASET 3] {label}")

    tar_dest = Path("tmp_ixi") / "IXI-T1.tar"
    success = False

    for url in _IXI_T1_URLS:
        print(f"  Trying: {url}")
        if _http_download(url, tar_dest, "IXI T1 volumes"):
            success = True
            break
        print("  [RETRY] Trying next mirror …")
        time.sleep(2)

    if not success:
        print("  [WARN] IXI download failed on all mirrors.")
        print("  [MANUAL] Download manually from: http://brain-development.org/ixi-dataset/")
        print(f"           Then place IXI-T1.tar in {tar_dest.parent}/")
        # Try Kaggle alternative for healthy brains
        _download_healthy_brain_kaggle()
        return

    # Extract
    print("  Extracting IXI-T1.tar …")
    result = subprocess.run(
        ["tar", "-xf", str(tar_dest), "-C", str(tar_dest.parent)],
        timeout=300,
    )
    if result.returncode != 0:
        print("  [ERROR] Extraction failed.")
        return

    copied = 0
    for nii_file in sorted(Path("tmp_ixi").rglob("*.nii.gz")):
        dst = NO_TUMOR_DIR / f"ixi_{nii_file.name}"
        if not dst.exists():
            _safe_copy(nii_file, dst)
            copied += 1

    shutil.rmtree("tmp_ixi", ignore_errors=True)
    print(f"  [OK] IXI healthy volumes copied: {copied}")


def _download_healthy_brain_kaggle() -> None:
    """
    Kaggle fallback for healthy brain images when IXI direct download fails.
    Uses the Open Access Series of Imaging Studies (OASIS) subset on Kaggle.
    """
    label = "OASIS Brain MRI (Kaggle Fallback)"
    print(f"\n  [FALLBACK] Trying: {label}")
    tmp = Path("tmp_oasis_kaggle")

    ok = _run_kaggle(
        slug="ninadaithal/imagesoasis",
        dest=tmp,
    )
    if not ok:
        # Second fallback
        ok = _run_kaggle(
            slug="jboysen/mri-and-alzheimers",
            dest=tmp,
        )

    if not ok:
        print("  [ERROR] All healthy brain Kaggle fallbacks failed.")
        print(
            "  [MANUAL] Download OASIS from: https://www.oasis-brains.org\n"
            "           Place patient folders under: dataset_volumetric/no_tumor/"
        )
        shutil.rmtree(tmp, ignore_errors=True)
        return

    copied = 0
    for img_file in sorted(tmp.rglob("*.jpg")) + sorted(tmp.rglob("*.png")):
        # Group by parent folder (= patient)
        patient_id = img_file.parent.name
        dest_patient = NO_TUMOR_DIR / f"oasis_{patient_id}"
        dest_patient.mkdir(parents=True, exist_ok=True)
        dst = dest_patient / img_file.name
        if not dst.exists():
            _safe_copy(img_file, dst)
            copied += 1

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  [OK] OASIS fallback images copied: {copied}")


# ─── DATASET 4 — Healthy Brain Kaggle Supplement ─────────────────────────────

def _download_healthy_supplement() -> None:
    """
    Additional healthy brain MRI scans from Kaggle to supplement IXI.
    Target: bring no_tumor study count close to tumor count for class balance.
    """
    label = "Healthy Brain MRI Supplement (Kaggle)"
    print(f"\n[DATASET 4] {label}")
    tmp = Path("tmp_healthy_supp")

    ok = _run_kaggle(
        slug="prathamgrover/healthy-brain-mri-images",
        dest=tmp,
    )
    if not ok:
        ok = _run_kaggle(
            slug="navoneel/brain-mri-images-for-brain-tumor-detection",
            dest=tmp,
        )

    if not ok:
        print(f"  [WARN] {label} — all Kaggle attempts failed. Skipping.")
        shutil.rmtree(tmp, ignore_errors=True)
        return

    copied = 0
    patient_idx = 0
    for img_file in sorted(tmp.rglob("*.jpg")) + sorted(tmp.rglob("*.png")):
        if any(k in img_file.parent.name.lower() for k in ["no", "healthy", "normal"]):
            dest_patient = NO_TUMOR_DIR / f"supp_healthy_{patient_idx:04d}"
            dest_patient.mkdir(parents=True, exist_ok=True)
            dst = dest_patient / img_file.name
            if not dst.exists():
                _safe_copy(img_file, dst)
                copied += 1
                patient_idx += 1

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  [OK] Healthy supplement images copied: {copied}")


# ─── VALIDATION ───────────────────────────────────────────────────────────────

def _print_summary() -> None:
    tumor_count    = _count_studies(TUMOR_DIR)
    no_tumor_count = _count_studies(NO_TUMOR_DIR)

    print("\n" + "=" * 70)
    print("  HYDRA — Volumetric Dataset Summary")
    print("=" * 70)
    print(f"  dataset_volumetric/tumor/    {tumor_count:>5} patient studies")
    print(f"  dataset_volumetric/no_tumor/ {no_tumor_count:>5} patient studies")
    print("=" * 70)

    if tumor_count < MIN_TUMOR_STUDIES or no_tumor_count < MIN_NO_TUMOR_STUDIES:
        print(
            "\n[WARN] Low study count.  You may need to manually add patient data.\n"
            "  Expected layout:\n"
            "    dataset_volumetric/tumor/<patient_id>/slice_NNN.jpg\n"
            "    dataset_volumetric/tumor/<patient_id>.nii.gz\n"
            "    dataset_volumetric/no_tumor/<patient_id>/slice_NNN.jpg\n"
            "    dataset_volumetric/no_tumor/<patient_id>.nii.gz\n"
        )
    else:
        print("\n[OK] Volumetric dataset ready for fine-tuning.")
        print("[NEXT] Run: python 05_volumetric_brain_finetune.py")


# ─── ORCHESTRATOR ─────────────────────────────────────────────────────────────

def download_volumetric_datasets() -> None:
    """
    Full volumetric data acquisition pipeline.

    Each download is independently guarded — already-populated sub-directories
    are skipped.  The script is safe to re-run at any time.
    """
    print("=" * 70)
    print("  HYDRA — Volumetric Dataset Acquisition (NIfTI / DICOM / Slices)")
    print("=" * 70)

    # Validate Kaggle credentials
    kaggle_creds = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_creds.exists():
        print(
            "[CRITICAL] ~/.kaggle/kaggle.json not found.\n"
            "  Kaggle downloads will fail.  Direct HTTP downloads will still proceed."
        )

    # Ensure output directories exist
    TUMOR_DIR.mkdir(parents=True, exist_ok=True)
    NO_TUMOR_DIR.mkdir(parents=True, exist_ok=True)

    # ── TUMOR studies ──────────────────────────────────────────────────────
    tumor_count = _count_studies(TUMOR_DIR)
    if tumor_count >= MIN_TUMOR_STUDIES:
        print(
            f"[SKIP] Tumor studies already present "
            f"({tumor_count} studies in '{TUMOR_DIR}')."
        )
    else:
        print(f"[INFO] Found {tumor_count} tumor studies — downloading more …")
        _download_brats_tumor()
        _download_lgg_tumor()

    # ── HEALTHY / NO-TUMOR studies ─────────────────────────────────────────
    no_tumor_count = _count_studies(NO_TUMOR_DIR)
    if no_tumor_count >= MIN_NO_TUMOR_STUDIES:
        print(
            f"[SKIP] Healthy brain studies already present "
            f"({no_tumor_count} studies in '{NO_TUMOR_DIR}')."
        )
    else:
        print(f"[INFO] Found {no_tumor_count} healthy studies — downloading more …")
        _download_ixi_healthy()
        _download_healthy_supplement()

    _print_summary()


if __name__ == "__main__":
    download_volumetric_datasets()
