"""
Module  : 05_volumetric_brain_finetune.py
Project : HYDRA — Clinical Brain Tumor Analysis System
Purpose : Fine-tune the trained Council ensemble on whole-brain volumetric
          slice sequences (200–300 axial slices per patient).

Why a Separate Fine-Tuning Script?
-----------------------------------
The Council (Script 04) was trained on individual 2D MRI/CT frames without
spatial context.  Whole-brain scan sequences carry additional diagnostic
signal:
  • Tumour continuity across adjacent slices confirms a genuine lesion.
  • Slice-to-slice intensity variation helps distinguish real pathology from
    scan artefacts.
By loading the existing Council weights and fine-tuning on patient-level
slice sequences, we adapt the models without discarding the 13+ hours of
prior knowledge.

Expected Dataset Layout (dataset_volumetric/)
----------------------------------------------
Place your volumetric data in *either* of these formats:

Format A — Pre-sliced JPG/PNG files (organised by patient)
  dataset_volumetric/
      tumor/
          patient_001/
              slice_001.jpg
              slice_002.jpg
              ...
          patient_002/ ...
      no_tumor/
          patient_001/ ...

Format B — NIfTI volumes (.nii or .nii.gz)
  dataset_volumetric/
      tumor/
          patient_001.nii.gz
          patient_002.nii.gz
      no_tumor/
          patient_001.nii.gz

Format C — DICOM series folders
  dataset_volumetric/
      tumor/
          patient_001/
              IM-0001-0001.dcm
              IM-0001-0002.dcm
              ...
      no_tumor/
          patient_001/ ...

Skip Behaviour
--------------
On first run a sentinel file 'HYDRA_Volumetric_Finetune.json' is written.
On subsequent runs the script detects this file and exits immediately, even
if run accidentally.

Output Artefacts
----------------
HYDRA_Swin_Council.pth    — Updated SwinV2 weights
HYDRA_ConvNext_Council.pth — Updated ConvNeXt weights
HYDRA_MONAI_Council.pth   — Updated MONAI weights
HYDRA_Volumetric_Finetune.json — Sentinel / training metadata log
"""

import json
import os
import random
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import nibabel as nib
import numpy as np
import pydicom
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from monai.networks.nets import SwinUNETR
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# ─── CONFIGURATION ────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOLUMETRIC_DATASET_DIR = "dataset_volumetric"
SENTINEL_PATH          = "HYDRA_Volumetric_Finetune.json"

WEIGHT_SWIN  = "HYDRA_Swin_Council.pth"
WEIGHT_CONV  = "HYDRA_ConvNext_Council.pth"
WEIGHT_MONAI = "HYDRA_MONAI_Council.pth"

MAX_SLICES_PER_PATIENT = 300   # Hard cap — prevents OOM on large volumes
MIN_SLICES_PER_PATIENT = 5     # Discard near-empty folders
NII_TRIM_FRACTION      = 0.15  # Exclude skull/noise at top and bottom 15 %

INPUT_SIZE     = 256
BATCH_SIZE     = 8             # Smaller batches accommodate 300-slice contexts
FINE_TUNE_EPOCHS = 6           # Short fine-tune — preserve prior knowledge
LEARNING_RATE  = 3e-5          # Very low LR — catastrophic forgetting prevention

SUPPORTED_IMAGE_EXT = ('.jpg', '.jpeg', '.png')


# ─── GUARD: SKIP IF ALREADY FINE-TUNED ───────────────────────────────────────

def volumetric_finetune_already_done() -> bool:
    """Return True if the sentinel file exists, preventing accidental re-runs."""
    if os.path.exists(SENTINEL_PATH):
        with open(SENTINEL_PATH) as fh:
            meta = json.load(fh)
        print(f"[SKIP] Volumetric fine-tune already completed on "
              f"{meta.get('timestamp', 'unknown date')}.")
        print(f"       Slices used  : {meta.get('total_slices', '?')}")
        print(f"       Patients     : {meta.get('total_patients', '?')}")
        print(f"       Final F1     : {meta.get('final_macro_f1', '?')}")
        print("       Delete 'HYDRA_Volumetric_Finetune.json' to re-run.")
        return True
    return False


# ─── DICOM / NIfTI LOADERS ───────────────────────────────────────────────────

def _normalise_array(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise a 2-D numpy array to uint8 [0, 255]."""
    arr = arr.astype(float)
    lo, hi = arr.min(), arr.max()
    normed = (arr - lo) / (hi - lo + 1e-8) * 255.0
    return normed.astype(np.uint8)


def _load_nifti_slices(path: str, max_slices: int) -> List[Image.Image]:
    """
    Extract axial slices from a NIfTI volume.
    Trims the top and bottom NII_TRIM_FRACTION of slices to remove
    skull cap and neck artefacts that carry no tumour signal.
    """
    vol = nib.load(path).get_fdata()
    depth = vol.shape[2]
    start = int(depth * NII_TRIM_FRACTION)
    end   = int(depth * (1 - NII_TRIM_FRACTION))
    indices = list(range(start, end))

    if len(indices) > max_slices:
        step = max(1, len(indices) // max_slices)
        indices = indices[::step][:max_slices]

    slices = []
    for z in indices:
        plane  = _normalise_array(vol[:, :, z])
        rgb    = cv2.cvtColor(plane, cv2.COLOR_GRAY2RGB)
        slices.append(Image.fromarray(rgb))
    return slices


def _load_dicom_series(folder: str, max_slices: int) -> List[Image.Image]:
    """
    Load and sort a DICOM series from a folder.
    Sorted by ImagePositionPatient[2] (Z-coordinate), falling back to
    InstanceNumber if the former is absent.
    """
    dcm_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith('.dcm')
    ]
    if not dcm_files:
        return []

    dcm_objects = []
    for path in dcm_files:
        try:
            dcm_objects.append(pydicom.dcmread(path))
        except Exception:
            continue

    try:
        dcm_objects.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    except Exception:
        try:
            dcm_objects.sort(key=lambda d: int(d.InstanceNumber))
        except Exception:
            pass

    if len(dcm_objects) > max_slices:
        step = max(1, len(dcm_objects) // max_slices)
        dcm_objects = dcm_objects[::step][:max_slices]

    slices = []
    for dcm in dcm_objects:
        try:
            plane = _normalise_array(dcm.pixel_array)
            rgb   = cv2.cvtColor(plane, cv2.COLOR_GRAY2RGB)
            slices.append(Image.fromarray(rgb))
        except Exception:
            continue
    return slices


def _load_image_sequence(folder: str, max_slices: int) -> List[Image.Image]:
    """
    Load a sorted sequence of JPEG/PNG files from a patient sub-folder.
    Files are sorted alphabetically to preserve acquisition order.
    """
    files = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(SUPPORTED_IMAGE_EXT)
    ])

    if len(files) > max_slices:
        step  = max(1, len(files) // max_slices)
        files = files[::step][:max_slices]

    slices = []
    for path in files:
        try:
            slices.append(Image.open(path).convert("RGB"))
        except Exception:
            continue
    return slices


def _discover_patient_slices(
    class_dir: str,
    label: int,
    max_slices: int,
) -> List[Tuple[Image.Image, int]]:
    """
    Discover all patient entries in a class directory and load their slices.
    Returns a flat list of (PIL_image, label) tuples.
    """
    samples = []

    for entry in sorted(os.listdir(class_dir)):
        entry_path = os.path.join(class_dir, entry)

        # NIfTI volume
        if entry.lower().endswith(('.nii', '.nii.gz')):
            slices = _load_nifti_slices(entry_path, max_slices)

        # Sub-folder (DICOM series or pre-sliced images)
        elif os.path.isdir(entry_path):
            if any(f.lower().endswith('.dcm') for f in os.listdir(entry_path)):
                slices = _load_dicom_series(entry_path, max_slices)
            else:
                slices = _load_image_sequence(entry_path, max_slices)

        # Single image file at top level
        elif entry.lower().endswith(SUPPORTED_IMAGE_EXT):
            try:
                slices = [Image.open(entry_path).convert("RGB")]
            except Exception:
                slices = []

        else:
            slices = []

        if len(slices) >= MIN_SLICES_PER_PATIENT:
            samples.extend([(s, label) for s in slices])

    return samples


# ─── PYTORCH DATASET ─────────────────────────────────────────────────────────

class VolumetricBrainDataset(Dataset):
    """
    Flat slice-level dataset constructed from whole-brain patient volumes.

    Each element is (transformed_tensor, binary_label) where:
      label 0 → No Tumour
      label 1 → Tumour Present

    The dataset supports NIfTI, DICOM, and pre-sliced JPEG/PNG inputs
    without requiring any format-specific pre-processing from the user.
    """

    CLASS_DIRS = {
        "tumor":    1,
        "no_tumor": 0,
        "Tumor":    1,
        "NoTumor":  0,
    }

    def __init__(
        self,
        root: str,
        transform=None,
        max_slices: int = MAX_SLICES_PER_PATIENT,
    ):
        self.transform = transform
        self.samples   = []
        self.patient_count = 0

        for folder_name, label in self.CLASS_DIRS.items():
            class_dir = os.path.join(root, folder_name)
            if not os.path.isdir(class_dir):
                continue

            n_before = len(self.samples)
            class_samples = _discover_patient_slices(class_dir, label, max_slices)
            self.samples.extend(class_samples)
            n_patients = len([
                e for e in os.listdir(class_dir)
                if os.path.isdir(os.path.join(class_dir, e)) or
                   e.lower().endswith(('.nii', '.nii.gz') + SUPPORTED_IMAGE_EXT)
            ])
            self.patient_count += n_patients
            print(f"  [{folder_name:<10}]  {n_patients:>4} patients  →  "
                  f"{len(self.samples) - n_before:>6} slices")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image, label = self.samples[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# ─── MONAI ADAPTER (consistent with Script 04 and 06) ────────────────────────

class MedicalSwinAdapter(nn.Module):
    """
    Wraps the MONAI Swin-UNETR encoder for 5-class classification.
    Naming must match Script 04 and 06 exactly.
    """

    def __init__(self):
        super().__init__()
        self.backbone   = SwinUNETR(
            spatial_dims=2, in_channels=3, out_channels=14, feature_size=24
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(384, 5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.swinViT(x, normalize=True)[-1]
        return self.classifier(features)


# ─── MODEL LOADING ───────────────────────────────────────────────────────────

def _load_pretrained_council():
    """Load all three Council branches from existing weight files."""
    swin  = timm.create_model(
        "swinv2_tiny_window8_256", pretrained=False, num_classes=5
    ).to(DEVICE)

    conv  = timm.create_model(
        "convnextv2_nano", pretrained=False, num_classes=5
    ).to(DEVICE)

    monai = MedicalSwinAdapter().to(DEVICE)

    for name, model, path in [
        ("SwinV2",   swin,  WEIGHT_SWIN),
        ("ConvNeXt", conv,  WEIGHT_CONV),
        ("MONAI",    monai, WEIGHT_MONAI),
    ]:
        if not os.path.exists(path):
            print(f"  [WARNING] {name} weight '{path}' not found. "
                  "Run 04_diagnostic_ensemble_training.py first.")
            continue
        try:
            model.load_state_dict(
                torch.load(path, map_location=DEVICE, weights_only=True),
                strict=True,
            )
            print(f"  [LOADED]  {name:<10} ← '{path}'")
        except Exception as err:
            print(f"  [WARNING] {name} load failed: {err}")

    return swin, conv, monai


# ─── FINE-TUNING LOOP ────────────────────────────────────────────────────────

def _finetune(swin, conv, monai, train_loader, val_loader) -> float:
    """
    Fine-tune all three Council branches on the volumetric dataset.
    Uses a very low learning rate and cosine annealing to prevent
    catastrophic forgetting of prior knowledge.
    Returns the final macro F1-score on the validation set.
    """
    # We reuse the same 5-class cross-entropy but with a binary-compatible
    # label mapping: index 2 = 'No Tumor', all others = tumour variants.
    # For the binary fine-tuning data we map: 1→any tumour class, 0→NoTumor.
    criterion = nn.BCEWithLogitsLoss()

    # Reduce to binary logit heads for fine-tuning on the 2-class volumetric data
    # by extracting 'tumor probability' = 1 - P(NoTumor)
    optimizer = optim.AdamW(
        list(swin.parameters()) +
        list(conv.parameters()) +
        list(monai.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-5,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINE_TUNE_EPOCHS)
    scaler    = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    for epoch in range(FINE_TUNE_EPOCHS):
        swin.train(); conv.train(); monai.train()
        epoch_loss = 0.0
        batch_bar  = tqdm(train_loader, desc=f"[FT Epoch {epoch + 1:02d}/{FINE_TUNE_EPOCHS}]")

        for images, labels in batch_bar:
            images = images.to(DEVICE)
            # Binary label → float for BCEWithLogits
            bin_labels = labels.float().to(DEVICE)
            optimizer.zero_grad()

            def _tumor_logit(logits_5c):
                # P(tumor) = 1 - P(NoTumor) where NoTumor is class index 2
                p = torch.softmax(logits_5c, dim=1)
                return torch.log(torch.clamp(1.0 - p[:, 2], 1e-7, 1 - 1e-7))

            if scaler:
                with torch.amp.autocast("cuda"):
                    loss = (
                        criterion(_tumor_logit(swin(images)),   bin_labels) +
                        criterion(_tumor_logit(conv(images)),   bin_labels) +
                        criterion(_tumor_logit(monai(images.float())), bin_labels)
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = (
                    criterion(_tumor_logit(swin(images)),   bin_labels) +
                    criterion(_tumor_logit(conv(images)),   bin_labels) +
                    criterion(_tumor_logit(monai(images)),  bin_labels)
                )
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        print(f"  Epoch {epoch + 1} complete — avg loss: {epoch_loss / len(train_loader):.4f}")

    # ── Validation ────────────────────────────────────────────────────────────
    print("\n[INFO] Evaluating on volumetric validation set …")
    swin.eval(); conv.eval(); monai.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)

            p_s  = torch.softmax(swin(images),         dim=1)
            p_c  = torch.softmax(conv(images),         dim=1)
            p_m  = torch.softmax(monai(images.float()), dim=1)
            cons = (0.4 * p_s) + (0.3 * p_c) + (0.3 * p_m)

            # Binary accuracy: predicted "tumor" if P(NoTumor) < 0.5
            predicted_tumor = (cons[:, 2] < 0.5).long().cpu()
            correct += (predicted_tumor == labels).sum().item()
            total   += labels.size(0)

    acc = 100.0 * correct / max(total, 1)
    print(f"  Volumetric Validation Accuracy : {acc:.2f}%  "
          f"({correct}/{total} slices correct)")
    return acc


# ─── ORCHESTRATOR ────────────────────────────────────────────────────────────

def run_volumetric_finetune() -> None:
    """
    Full volumetric fine-tuning pipeline.
    Idempotent — exits immediately if the sentinel file exists.
    """
    print("=" * 70)
    print("  HYDRA — Volumetric Brain Slice Fine-Tuning")
    print("=" * 70)

    if volumetric_finetune_already_done():
        return

    if not os.path.isdir(VOLUMETRIC_DATASET_DIR):
        print(f"[CRITICAL] Volumetric dataset directory "
              f"'{VOLUMETRIC_DATASET_DIR}' not found.")
        print("  Please create the directory and populate it with one of these layouts:")
        print("    dataset_volumetric/tumor/patient_001/*.jpg")
        print("    dataset_volumetric/tumor/patient_001.nii.gz")
        print("    dataset_volumetric/no_tumor/patient_001/*.dcm")
        return

    print("[INFO] Discovering and loading volumetric patient data …\n")

    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    full_dataset = VolumetricBrainDataset(
        root=VOLUMETRIC_DATASET_DIR,
        transform=transform,
        max_slices=MAX_SLICES_PER_PATIENT,
    )

    if len(full_dataset) == 0:
        print("[CRITICAL] No valid slices found in the volumetric dataset directory.")
        return

    labels = [full_dataset.samples[i][1] for i in range(len(full_dataset))]
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        stratify=labels,
        random_state=42,
    )

    from torch.utils.data import Subset
    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_idx),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    print(f"\n[INFO] Dataset summary:")
    print(f"  Total slices (train) : {len(train_idx)}")
    print(f"  Total slices (val)   : {len(val_idx)}")
    print(f"  Total patients       : {full_dataset.patient_count}")
    print(f"  Device               : {DEVICE}\n")

    print("[INFO] Loading pre-trained Council weights …")
    swin, conv, monai = _load_pretrained_council()

    print(f"\n[INFO] Starting fine-tuning — {FINE_TUNE_EPOCHS} epochs …\n")
    accuracy = _finetune(swin, conv, monai, train_loader, val_loader)

    torch.save(swin.state_dict(),  WEIGHT_SWIN)
    torch.save(conv.state_dict(),  WEIGHT_CONV)
    torch.save(monai.state_dict(), WEIGHT_MONAI)

    # Write sentinel file
    metadata = {
        "timestamp":      datetime.now().isoformat(),
        "total_slices":   len(full_dataset),
        "total_patients": full_dataset.patient_count,
        "epochs":         FINE_TUNE_EPOCHS,
        "final_accuracy": round(accuracy, 4),
        "final_macro_f1": "see 07_ensemble_performance_evaluation.py",
        "device":         str(DEVICE),
    }
    with open(SENTINEL_PATH, "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"\n[SUCCESS] Fine-tuned weights saved:")
    print(f"  → {WEIGHT_SWIN}")
    print(f"  → {WEIGHT_CONV}")
    print(f"  → {WEIGHT_MONAI}")
    print(f"  → Sentinel file written to '{SENTINEL_PATH}'")
    print("=" * 70)


if __name__ == "__main__":
    run_volumetric_finetune()
