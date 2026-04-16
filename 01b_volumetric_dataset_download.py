"""
01b_volumetric_dataset_download.py
====================================
Downloads patient-level brain MRI studies for volumetric fine-tuning (Script 05).

FIXES IN THIS VERSION
─────────────────────
• Stale partial downloads cleaned before every kaggle call  → no more "corrupted zip"
• --force flag added to all kaggle commands                 → fresh download always
• BraTS 2020 (4.16 GB) removed — replaced with LGG (714 MB)
• OASIS (1.23 GB) removed — replaced with navoneel (~200 MB)
• Kaggle upgrade command added to setup
• _HAS_NIBABEL renamed to _has_nibabel (lowercase avoids constant-redefinition error)
• Unused `import os` removed
• nibabel type annotations corrected

Datasets
────────
TUMOR
  1. LGG Segmentation  (mateuszbuda/lgg-mri-segmentation)  ~714 MB
     Per-patient FLAIR slice folders, 110 patients

NO TUMOR
  2. navoneel brain MRI (navoneel/brain-mri-images-for-brain-tumor-detection)  ~200 MB
     'no/' subfolder = healthy brain scans, grouped into synthetic patient folders

Output layout
─────────────
dataset_volumetric/
├── tumor/
│   └── lgg_TCGA_CS_4941_19960909/   ← jpeg slices per patient
└── no_tumor/
    └── healthy_0001/                ← synthetic patient folders
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

try:
    import nibabel as _nib  # type: ignore[import-untyped]
    import numpy as _np
    from PIL import Image as _PILImage
    _has_nibabel = True
except ImportError:
    _has_nibabel = False

# ── Config ────────────────────────────────────────────────────────────────────

VOL_ROOT   = Path("dataset_volumetric")
TUMOR_DIR  = VOL_ROOT / "tumor"
NOTUM_DIR  = VOL_ROOT / "no_tumor"

MIN_STUDIES = 10          # skip a class when this many study folders already exist
MAX_SLICES  = 120         # slices to extract/copy per patient
IMG_EXT     = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_studies(d: Path) -> int:
    """Count immediate sub-directories = patient study folders."""
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir() if p.is_dir())


def _kaggle(slug: str, dest: Path) -> bool:
    """
    Download + unzip a Kaggle dataset.
    ALWAYS deletes dest first so stale partial zips cannot corrupt the result.
    Returns True on success.
    """
    # ── Critical fix: delete any previous partial download ─────────────────
    if dest.exists():
        print(f"  [CLEAN] Removing stale tmp dir: {dest}")
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", slug,
            "-p", str(dest),
            "--unzip",
            "--force",          # force fresh download, skip resume logic
        ],
        timeout=3600,           # 1 hour max per dataset
    )
    return result.returncode == 0


def _norm_arr(arr: Any) -> Any:
    """Normalise a numpy array to uint8 [0, 255]."""
    a = _np.array(arr, dtype=float)
    lo, hi = float(a.min()), float(a.max())
    return ((a - lo) / (hi - lo + 1e-8) * 255).clip(0, 255).astype("uint8")


def _nifti_to_jpegs(nii_path: Path, out_dir: Path) -> int:
    """
    Extract up to MAX_SLICES JPEG images from a NIfTI volume.
    Returns the number of slices written, or 0 on failure.
    Requires nibabel.
    """
    if not _has_nibabel:
        return 0
    try:
        img = _nib.load(str(nii_path))  # type: ignore[reportPrivateImportUsage]
        vol = img.get_fdata()           # type: ignore[attr-defined]
    except Exception as exc:
        print(f"    [WARN] Cannot read {nii_path.name}: {exc}")
        return 0

    if not hasattr(vol, "ndim") or vol.ndim < 3:  # type: ignore[union-attr]
        return 0

    depth = int(vol.shape[2])  # type: ignore[index]
    trim  = int(depth * 0.15)
    start, end = trim, max(depth - trim, trim + 1)
    total = end - start
    step  = max(1, total // MAX_SLICES)
    idxs  = list(range(start, end, step))[:MAX_SLICES]

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, z in enumerate(idxs):
        raw = _norm_arr(vol[:, :, z])  # type: ignore[index]
        if raw.ndim == 2:
            raw = _np.stack([raw, raw, raw], axis=2)
        _PILImage.fromarray(raw).save(out_dir / f"slice_{i:04d}.jpg")
    return len(idxs)


def _images_to_jpegs(src_dir: Path, out_dir: Path) -> int:
    """
    Copy or convert up to MAX_SLICES images from src_dir → out_dir as JPEGs.
    Skips mask files.  Returns number written.
    """
    imgs = sorted(
        f for f in src_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in IMG_EXT
        and "mask" not in f.name.lower()
    )[:MAX_SLICES]
    if not imgs:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(imgs):
        dst = out_dir / f"slice_{i:04d}.jpg"
        if dst.exists():
            continue
        if src.suffix.lower() in (".tif", ".tiff"):
            try:
                _PILImage.open(src).convert("RGB").save(dst)
            except Exception:
                shutil.copy(src, dst)
        else:
            shutil.copy(src, dst)
    return len(imgs)


# ── Dataset 1 — LGG Segmentation (tumor) ──────────────────────────────────────

def _lgg() -> None:
    """
    LGG MRI Segmentation — 110 patients, FLAIR slice folders.
    Size: ~714 MB  (manageable on most connections)
    Slug: mateuszbuda/lgg-mri-segmentation
    """
    print("\n[D1] LGG Segmentation  (mateuszbuda/lgg-mri-segmentation)")
    tmp = Path("tmp_lgg")

    if not _kaggle("mateuszbuda/lgg-mri-segmentation", tmp):
        print("  [ERROR] Kaggle download failed.")
        print("  Verify: kaggle --version  →  should NOT show '{current_version}'")
        print("  Fix:    pip install kaggle --upgrade")
        print("  Manual: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation")
        return

    copied = 0
    # Layout: kaggle_3m/<PatientID>/*.tif  (brain + *_mask.tif pairs)
    for pat_dir in sorted(tmp.rglob("kaggle_3m/*")):
        if not pat_dir.is_dir():
            continue
        out_dir = TUMOR_DIR / f"lgg_{pat_dir.name}"
        if out_dir.exists() and sum(1 for _ in out_dir.glob("*.jpg")) > 0:
            continue
        n = _images_to_jpegs(pat_dir, out_dir)
        if n > 0:
            copied += 1

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  [OK] LGG — {copied} patient study folders created in {TUMOR_DIR}")


# ── Dataset 2 — Healthy Brain MRI (no_tumor) ──────────────────────────────────

def _healthy_navoneel() -> None:
    """
    navoneel/brain-mri-images-for-brain-tumor-detection
    The 'no/' sub-folder contains healthy brain scans.
    Size: ~200 MB  (fast download)
    Grouped into synthetic patient folders of 20 slices each.
    """
    print("\n[D2] Healthy Brain MRI  (navoneel/brain-mri-images-for-brain-tumor-detection)")
    tmp = Path("tmp_healthy")

    if not _kaggle("navoneel/brain-mri-images-for-brain-tumor-detection", tmp):
        print("  [ERROR] Kaggle download failed.")
        print("  Manual: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection")
        return

    # Collect all images from 'no/' folders (healthy brains)
    imgs: list[Path] = []
    for no_dir in tmp.rglob("no"):
        if no_dir.is_dir():
            imgs.extend(
                f for f in no_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMG_EXT
            )

    # Group into synthetic patient folders of 20 images each
    BATCH = 20
    copied = 0
    for i in range(0, len(imgs), BATCH):
        batch = imgs[i: i + BATCH]
        if not batch:
            continue
        out_dir = NOTUM_DIR / f"healthy_{i // BATCH:04d}"
        if out_dir.exists() and sum(1 for _ in out_dir.glob("*.jpg")) > 0:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for j, src in enumerate(batch):
            shutil.copy(src, out_dir / f"slice_{j:04d}.jpg")
        copied += 1

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  [OK] Healthy — {copied} synthetic patient folders created in {NOTUM_DIR}")


# ── Dataset 3 — Sartaj Brain Tumour (tumor supplement, optional) ──────────────

def _sartaj_supplement() -> None:
    """
    sartajbhuvaji/brain-tumor-classification-mri  ~150 MB
    Additional glioma/meningioma/pituitary slices if LGG count is still low.
    Groups images into synthetic patient folders of 20 slices.
    """
    print("\n[D3] Tumour supplement  (sartajbhuvaji/brain-tumor-classification-mri)")
    tmp = Path("tmp_sartaj")

    if not _kaggle("sartajbhuvaji/brain-tumor-classification-mri", tmp):
        print("  [WARN] Sartaj supplement failed — LGG alone is sufficient.")
        shutil.rmtree(tmp, ignore_errors=True)
        return

    # Collect tumour-class images (glioma, meningioma, pituitary)
    tumour_kw = ("glioma", "meningioma", "pituitary")
    imgs: list[Path] = []
    for cls_dir in tmp.rglob("*"):
        if not cls_dir.is_dir():
            continue
        if any(k in cls_dir.name.lower() for k in tumour_kw):
            imgs.extend(
                f for f in cls_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMG_EXT
            )

    BATCH = 20
    copied = 0
    for i in range(0, len(imgs), BATCH):
        batch = imgs[i: i + BATCH]
        if not batch:
            continue
        out_dir = TUMOR_DIR / f"sartaj_{i // BATCH:04d}"
        if out_dir.exists() and sum(1 for _ in out_dir.glob("*.jpg")) > 0:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for j, src in enumerate(batch):
            shutil.copy(src, out_dir / f"slice_{j:04d}.jpg")
        copied += 1

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  [OK] Sartaj — {copied} synthetic patient folders → {TUMOR_DIR}")


# ── Summary ───────────────────────────────────────────────────────────────────

def _summary() -> None:
    t = _count_studies(TUMOR_DIR)
    n = _count_studies(NOTUM_DIR)
    print("\n" + "=" * 68)
    print("  Volumetric Dataset Summary")
    print("=" * 68)
    print(f"  dataset_volumetric/tumor/    {t:>5}  patient study folders")
    print(f"  dataset_volumetric/no_tumor/ {n:>5}  patient study folders")
    print("=" * 68)
    if t < MIN_STUDIES or n < MIN_STUDIES:
        print("\n[WARN] Low study count.  Add data manually:")
        print("  dataset_volumetric/tumor/<patient_id>/slice_0000.jpg")
        print("  dataset_volumetric/no_tumor/<patient_id>/slice_0000.jpg")
        print("  (min 8 slices per folder, each folder = one patient)")
    else:
        print("\n[OK]  Ready for: python 05_volumetric_brain_finetune.py")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 68)
    print("  Brain Tumor Detection — Volumetric Dataset Download")
    print("=" * 68)

    if not _has_nibabel:
        print("[WARN] nibabel not installed — NIfTI extraction disabled.")
        print("  pip install nibabel")

    creds = Path.home() / ".kaggle" / "kaggle.json"
    if not creds.exists():
        print("[ERROR] ~/.kaggle/kaggle.json not found.")
        print("  cp kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json")
        return

    # ── Upgrade kaggle CLI to fix the '{current_version}' corruption bug ──
    print("[INFO] Upgrading Kaggle CLI to fix download corruption bug …")
    subprocess.run(
        ["pip", "install", "kaggle", "--upgrade", "--quiet"],
        timeout=120,
    )

    TUMOR_DIR.mkdir(parents=True, exist_ok=True)
    NOTUM_DIR.mkdir(parents=True, exist_ok=True)

    # ── Tumor class ───────────────────────────────────────────────────────
    t_now = _count_studies(TUMOR_DIR)
    if t_now >= MIN_STUDIES:
        print(f"\n[SKIP] Tumor: {t_now} studies already present.")
    else:
        print(f"\n[INFO] Tumor: {t_now} studies — downloading …")
        _lgg()
        # If still low, add sartaj supplement
        if _count_studies(TUMOR_DIR) < MIN_STUDIES:
            _sartaj_supplement()

    # ── No-tumor class ────────────────────────────────────────────────────
    n_now = _count_studies(NOTUM_DIR)
    if n_now >= MIN_STUDIES:
        print(f"\n[SKIP] No-tumor: {n_now} studies already present.")
    else:
        print(f"\n[INFO] No-tumor: {n_now} studies — downloading …")
        _healthy_navoneel()

    _summary()


if __name__ == "__main__":
    main()
