"""
01_neuroimaging_data_acquisition.py — HYDRA Base Dataset Acquisition
=====================================================================
Downloads and organises the original 2-D Kaggle MRI/CT classification
datasets used for base model training (Stages 02–04).

Pipeline Stages
---------------
Stage A  → MRI Dataset  (Kaggle: masoudnickparvar/brain-tumor-mri-dataset)
           Classes: Glioma | Meningioma | Pituitary | NoTumor

Stage B  → CT Dataset   (Kaggle: ahmedhamada0/brain-tumor-detection)
           Classes: Tumor_Generic | NoTumor

Stage C  → Negative Safety Samples
           Chest X-rays (Kaggle: paultimothymooney/chest-xray-pneumonia)
           Facial Photos (Kaggle: jessicali9530/celeba-dataset)

Skip Behaviour
--------------
Each stage checks whether its target directory already contains images.
If so, the stage is skipped entirely — no re-downloading, no overwriting.
This script is fully idempotent and safe to run multiple times.

NOTE: This script only downloads the BASE 2-D slice data used by the
      Gatekeeper (02) and Council (04).  For the new VOLUMETRIC datasets
      (BraTS, IXI, OASIS-3) used by the fine-tuning stage (05), run:
          python 01b_volumetric_dataset_download.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

ENSEMBLE_DATASET_DIR  = "dataset_ensemble"
NEGATIVES_DATASET_DIR = "dataset_negatives"
FACES_DATASET_DIR     = "dataset_faces"

BRAIN_SAMPLES_PER_CLASS = 3_000
NEGATIVE_SAMPLES_CAP    = 1_500

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _count_images(directory: str) -> int:
    """Count image files recursively inside *directory*."""
    count = 0
    if not os.path.isdir(directory):
        return 0
    for _, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(SUPPORTED_EXTENSIONS):
                count += 1
    return count


def _dataset_exists(directory: str, min_images: int = 10) -> bool:
    """Return True when the directory contains at least *min_images* images."""
    return _count_images(directory) >= min_images


def _run_kaggle(slug: str, dest: str) -> bool:
    """
    Download and unzip a Kaggle dataset.
    Returns True on success, False on failure.
    """
    os.makedirs(dest, exist_ok=True)
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", dest, "--unzip"],
        capture_output=False,
    )
    return result.returncode == 0


# ─── STAGE A — MRI ────────────────────────────────────────────────────────────

def _ingest_mri() -> None:
    """Download MRI dataset and route images into semantic sub-directories."""
    print("[INFO] Stage A — Downloading MRI dataset …")
    ok = _run_kaggle(
        slug="masoudnickparvar/brain-tumor-mri-dataset",
        dest="tmp_mri_cache",
    )
    if not ok:
        print("[ERROR] Stage A — Kaggle download failed. Check kaggle.json.")
        return

    routing: dict[str, str] = {
        "glioma":     "Glioma",
        "meningioma": "Meningioma",
        "pituitary":  "Pituitary",
        "no":         "NoTumor",
    }

    copied = 0
    for root_dir, _, file_names in os.walk("tmp_mri_cache"):
        for file_name in file_names:
            if not file_name.lower().endswith(SUPPORTED_EXTENSIONS):
                continue
            root_lower = root_dir.lower()
            target_cls = next(
                (v for k, v in routing.items() if k in root_lower), None
            )
            if target_cls:
                dest_dir = os.path.join(ENSEMBLE_DATASET_DIR, target_cls)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy(
                    os.path.join(root_dir, file_name),
                    os.path.join(dest_dir, f"mri_{file_name}"),
                )
                copied += 1

    shutil.rmtree("tmp_mri_cache", ignore_errors=True)
    print(f"[INFO] Stage A — MRI ingestion complete ({copied} images).")


# ─── STAGE B — CT ─────────────────────────────────────────────────────────────

def _ingest_ct() -> None:
    """Download CT detection dataset → Tumor_Generic / NoTumor."""
    print("[INFO] Stage B — Downloading CT dataset …")
    ok = _run_kaggle(
        slug="ahmedhamada0/brain-tumor-detection",
        dest="tmp_ct_cache",
    )
    if not ok:
        print("[ERROR] Stage B — Kaggle download failed.")
        return

    os.makedirs(os.path.join(ENSEMBLE_DATASET_DIR, "Tumor_Generic"), exist_ok=True)
    os.makedirs(os.path.join(ENSEMBLE_DATASET_DIR, "NoTumor"),       exist_ok=True)

    copied = 0
    for root_dir, _, file_names in os.walk("tmp_ct_cache"):
        root_lower = root_dir.lower()
        for file_name in file_names:
            if not file_name.lower().endswith((".jpg", ".png")):
                continue
            if "yes" in root_lower:
                dest = os.path.join(ENSEMBLE_DATASET_DIR, "Tumor_Generic", f"ct_{file_name}")
            elif "no" in root_lower:
                dest = os.path.join(ENSEMBLE_DATASET_DIR, "NoTumor", f"ct_{file_name}")
            else:
                continue
            shutil.copy(os.path.join(root_dir, file_name), dest)
            copied += 1

    shutil.rmtree("tmp_ct_cache", ignore_errors=True)
    print(f"[INFO] Stage B — CT ingestion complete ({copied} images).")


# ─── STAGE C — NEGATIVES ──────────────────────────────────────────────────────

def _ingest_negatives() -> None:
    """Download distractor datasets (X-rays + faces) for Gatekeeper training."""
    print("[INFO] Stage C — Downloading negative safety samples …")
    os.makedirs(NEGATIVES_DATASET_DIR, exist_ok=True)
    os.makedirs(FACES_DATASET_DIR,     exist_ok=True)

    # Chest X-rays
    ok = _run_kaggle(
        slug="paultimothymooney/chest-xray-pneumonia",
        dest="tmp_xray_cache",
    )
    if ok:
        idx = 0
        for root_dir, _, files in os.walk("tmp_xray_cache"):
            for f in files:
                if f.lower().endswith(".jpeg") and idx < NEGATIVE_SAMPLES_CAP:
                    shutil.copy(
                        os.path.join(root_dir, f),
                        os.path.join(NEGATIVES_DATASET_DIR, f"xray_{idx:05d}.jpg"),
                    )
                    idx += 1
        print(f"[INFO] Stage C — X-ray distractors: {idx} images.")
        shutil.rmtree("tmp_xray_cache", ignore_errors=True)
    else:
        print("[WARN] Stage C — X-ray download failed.")

    # Facial photos
    ok = _run_kaggle(
        slug="jessicali9530/celeba-dataset",
        dest="tmp_faces_cache",
    )
    if ok:
        idx = 0
        for root_dir, _, files in os.walk("tmp_faces_cache"):
            for f in files:
                if f.lower().endswith(".jpg") and idx < NEGATIVE_SAMPLES_CAP:
                    shutil.copy(
                        os.path.join(root_dir, f),
                        os.path.join(FACES_DATASET_DIR, f"face_{idx:05d}.jpg"),
                    )
                    idx += 1
        print(f"[INFO] Stage C — Face distractors: {idx} images.")
        shutil.rmtree("tmp_faces_cache", ignore_errors=True)
    else:
        print("[WARN] Stage C — Face download failed.")


# ─── ORCHESTRATOR ─────────────────────────────────────────────────────────────

def prepare_neuroimaging_datasets() -> None:
    """
    Entry point for the full base data acquisition pipeline.
    Each stage independently checks for existing data before downloading.
    """
    print("=" * 70)
    print("  HYDRA — Base Neuroimaging Data Acquisition (2-D Slice Datasets)")
    print("=" * 70)

    # Validate Kaggle credentials
    kaggle_creds = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    if not os.path.exists(kaggle_creds):
        print(
            "[CRITICAL] ~/.kaggle/kaggle.json not found.\n"
            "  Copy kaggle.json to ~/.kaggle/ and run:\n"
            "    chmod 600 ~/.kaggle/kaggle.json"
        )
        sys.exit(1)

    # Stage A + B — MRI / CT ensemble dataset
    ensemble_count = _count_images(ENSEMBLE_DATASET_DIR)
    if ensemble_count >= 1_000:
        print(
            f"[SKIP] Ensemble dataset already populated "
            f"({ensemble_count} images in '{ENSEMBLE_DATASET_DIR}')."
        )
    else:
        os.makedirs(ENSEMBLE_DATASET_DIR, exist_ok=True)
        _ingest_mri()
        _ingest_ct()

    # Stage C — Negative safety samples
    neg_count = _count_images(NEGATIVES_DATASET_DIR)
    face_count = _count_images(FACES_DATASET_DIR)
    if neg_count >= 100 and face_count >= 100:
        print(
            f"[SKIP] Negative samples already present "
            f"(X-rays: {neg_count}, Faces: {face_count})."
        )
    else:
        _ingest_negatives()

    # Summary
    print("\n" + "=" * 70)
    print("  Acquisition Summary — dataset_ensemble/")
    print("=" * 70)
    if os.path.isdir(ENSEMBLE_DATASET_DIR):
        for cls in sorted(os.listdir(ENSEMBLE_DATASET_DIR)):
            cls_path = os.path.join(ENSEMBLE_DATASET_DIR, cls)
            if not os.path.isdir(cls_path):
                continue
            n = _count_images(cls_path)
            print(f"  {cls:<20}  {n:>6} images")
    print("=" * 70)
    print(
        "\n[NEXT] Run: python 01b_volumetric_dataset_download.py\n"
        "       to download the volumetric (BraTS/IXI/OASIS) datasets.\n"
        "       OR skip directly to: python 02_gatekeeper_model_training.py"
    )


if __name__ == "__main__":
    prepare_neuroimaging_datasets()
