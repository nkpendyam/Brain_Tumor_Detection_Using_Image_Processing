"""
Module  : 01_neuroimaging_data_acquisition.py
Project : HYDRA — Clinical Brain Tumor Analysis System
Purpose : Automated multi-modal neuroimaging dataset acquisition and
          semantic organisation into a 5-class clinical hierarchy.

Pipeline Overview
-----------------
Stage A  → MRI Dataset  (Kaggle: masoudnickparvar/brain-tumor-mri-dataset)
           Glioma | Meningioma | Pituitary | NoTumor
Stage B  → CT Dataset   (Kaggle: ahmedhamada0/brain-tumor-detection)
           Tumor_Generic | NoTumor
Stage C  → Negative Safety Samples (Chest X-rays + Facial Photographs)
           Used to train the Gatekeeper to reject out-of-distribution inputs.

Data Lineage
------------
All file copies are prefixed with their modality tag (mri_, ct_, xray_,
face_) so every image retains a traceable identity in the merged folder.
Augmentations are applied in-memory during training — raw data on disk
is kept as an immutable source of truth.

Run Behaviour
-------------
Idempotent: If the target directories already exist and are non-empty,
the script logs a status message and exits without re-downloading.
"""

import os
import shutil
import random


# ─── CONFIGURATION ────────────────────────────────────────────────────────────

ENSEMBLE_DATASET_DIR  = "dataset_ensemble"          # MRI + CT tumour classes
NEGATIVES_DATASET_DIR = "dataset_negatives"         # Chest X-ray distractors
FACES_DATASET_DIR     = "dataset_faces"             # Facial photo distractors

BRAIN_SAMPLES_PER_CLASS = 3000   # Positive class cap for Gatekeeper training
NEGATIVE_SAMPLES_CAP    = 1500   # Distractor cap (X-rays and faces, each)

SUPPORTED_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _dataset_exists(directory: str) -> bool:
    """Return True when the directory exists and contains at least one image."""
    if not os.path.isdir(directory):
        return False
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
                return True
    return False


def _download_kaggle_dataset(slug: str, cache_dir: str) -> None:
    """Invoke the Kaggle CLI to download and unzip a dataset."""
    os.system(
        f"kaggle datasets download -d {slug} "
        f"-p {cache_dir} --unzip"
    )


# ─── STAGE A: MRI ACQUISITION ─────────────────────────────────────────────────

def _ingest_mri_dataset() -> None:
    """
    Download the Brain Tumour MRI Dataset and route each image into one of
    four semantic sub-directories using keyword-based path matching.

    Routing Rules (in priority order)
    ──────────────────────────────────
    Source Folder Keyword → Target Sub-Directory
    'glioma'              → Glioma
    'meningioma'          → Meningioma
    'pituitary'           → Pituitary
    'no'                  → NoTumor
    """
    print("[INFO] Stage A — Downloading MRI Dataset …")
    _download_kaggle_dataset(
        slug="masoudnickparvar/brain-tumor-mri-dataset",
        cache_dir="tmp_mri_cache",
    )

    routing_map = {
        "glioma":      "Glioma",
        "meningioma":  "Meningioma",
        "pituitary":   "Pituitary",
        "no":          "NoTumor",
    }

    for current_root, _, file_names in os.walk("tmp_mri_cache"):
        for file_name in file_names:
            if not file_name.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
                continue

            root_lower = current_root.lower()
            target_class = next(
                (v for k, v in routing_map.items() if k in root_lower),
                None,
            )

            if target_class:
                dest_dir = os.path.join(ENSEMBLE_DATASET_DIR, target_class)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy(
                    os.path.join(current_root, file_name),
                    os.path.join(dest_dir, f"mri_{file_name}"),
                )

    shutil.rmtree("tmp_mri_cache", ignore_errors=True)
    print("[INFO] Stage A — MRI ingestion complete.")


# ─── STAGE B: CT ACQUISITION ──────────────────────────────────────────────────

def _ingest_ct_dataset() -> None:
    """
    Download the CT Brain Tumour Detection dataset.
    'yes' folders → Tumor_Generic class.
    'no'  folders → NoTumor class.
    """
    print("[INFO] Stage B — Downloading CT Dataset …")
    _download_kaggle_dataset(
        slug="ahmedhamada0/brain-tumor-detection",
        cache_dir="tmp_ct_cache",
    )

    os.makedirs(os.path.join(ENSEMBLE_DATASET_DIR, "Tumor_Generic"), exist_ok=True)
    os.makedirs(os.path.join(ENSEMBLE_DATASET_DIR, "NoTumor"),       exist_ok=True)

    for current_root, _, file_names in os.walk("tmp_ct_cache"):
        root_lower = current_root.lower()
        for file_name in file_names:
            if not file_name.lower().endswith(('.jpg', '.png')):
                continue
            if "yes" in root_lower:
                dest = os.path.join(ENSEMBLE_DATASET_DIR, "Tumor_Generic", f"ct_{file_name}")
            elif "no" in root_lower:
                dest = os.path.join(ENSEMBLE_DATASET_DIR, "NoTumor", f"ct_{file_name}")
            else:
                continue
            shutil.copy(os.path.join(current_root, file_name), dest)

    shutil.rmtree("tmp_ct_cache", ignore_errors=True)
    print("[INFO] Stage B — CT ingestion complete.")


# ─── STAGE C: NEGATIVE SAFETY SAMPLES ────────────────────────────────────────

def _ingest_negative_samples() -> None:
    """
    Download chest X-ray and facial photograph datasets to serve as
    out-of-distribution (OOD) distractors for Gatekeeper training.

    A 1 : 1 Brain-to-Non-Brain ratio avoids classifier bias.
    1 500 X-rays  + 1 500 Faces = 3 000 negative samples
    3 000 positive (brain) samples mirrors this count exactly.
    """
    print("[INFO] Stage C — Downloading negative safety samples …")

    os.makedirs(NEGATIVES_DATASET_DIR, exist_ok=True)
    os.makedirs(FACES_DATASET_DIR,     exist_ok=True)

    # X-ray distractors
    _download_kaggle_dataset(
        slug="paultimothymooney/chest-xray-pneumonia",
        cache_dir="tmp_xray_cache",
    )
    xray_index = 0
    for root, _, files in os.walk("tmp_xray_cache"):
        for f in files:
            if f.endswith('.jpeg') and xray_index < NEGATIVE_SAMPLES_CAP:
                shutil.copy(
                    os.path.join(root, f),
                    os.path.join(NEGATIVES_DATASET_DIR, f"xray_{xray_index}.jpg"),
                )
                xray_index += 1

    # Facial photo distractors
    _download_kaggle_dataset(
        slug="jessicali9530/celeba-dataset",
        cache_dir="tmp_faces_cache",
    )
    face_index = 0
    for root, _, files in os.walk("tmp_faces_cache"):
        for f in files:
            if f.endswith('.jpg') and face_index < NEGATIVE_SAMPLES_CAP:
                shutil.copy(
                    os.path.join(root, f),
                    os.path.join(FACES_DATASET_DIR, f"face_{face_index}.jpg"),
                )
                face_index += 1

    shutil.rmtree("tmp_xray_cache",  ignore_errors=True)
    shutil.rmtree("tmp_faces_cache", ignore_errors=True)
    print("[INFO] Stage C — Negative sample ingestion complete.")


# ─── ORCHESTRATOR ─────────────────────────────────────────────────────────────

def prepare_neuroimaging_datasets() -> None:
    """
    Entry point for the full data acquisition pipeline.
    Checks for existing datasets before downloading to remain idempotent.
    """
    print("=" * 70)
    print("  HYDRA — Neuroimaging Data Acquisition Pipeline")
    print("=" * 70)

    # Validate Kaggle credentials
    kaggle_creds = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    if not os.path.exists(kaggle_creds):
        print("[CRITICAL] kaggle.json not found at ~/.kaggle/. "
              "Please configure Kaggle API credentials and retry.")
        return

    # Stage A + B: MRI and CT ensemble dataset
    if _dataset_exists(ENSEMBLE_DATASET_DIR):
        print(f"[SKIP] Ensemble dataset already present at '{ENSEMBLE_DATASET_DIR}'.")
    else:
        os.makedirs(ENSEMBLE_DATASET_DIR, exist_ok=True)
        _ingest_mri_dataset()
        _ingest_ct_dataset()

    # Stage C: Negative safety samples
    if _dataset_exists(NEGATIVES_DATASET_DIR):
        print(f"[SKIP] Negative samples already present at '{NEGATIVES_DATASET_DIR}'.")
    else:
        _ingest_negative_samples()

    # Summary
    print("\n" + "=" * 70)
    print("  Acquisition Complete — Dataset Class Summary")
    print("=" * 70)
    if os.path.isdir(ENSEMBLE_DATASET_DIR):
        for cls in sorted(os.listdir(ENSEMBLE_DATASET_DIR)):
            cls_path = os.path.join(ENSEMBLE_DATASET_DIR, cls)
            count = len([
                f for f in os.listdir(cls_path)
                if f.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)
            ])
            print(f"  {cls:<20}  {count:>5} images")
    print("=" * 70)


if __name__ == "__main__":
    prepare_neuroimaging_datasets()
