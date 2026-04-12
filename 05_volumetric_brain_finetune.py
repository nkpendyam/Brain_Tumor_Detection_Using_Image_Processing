"""
05_volumetric_brain_finetune.py — HYDRA Volumetric Transfer Learning
=====================================================================
Fast patient-level transfer learning on whole-brain studies (NIfTI/DICOM/slices).

Reuses already-trained Council weights from Stage 04 and adapts only the
head layers for 200-300 slice patient studies — without full retraining.

Key Design Decisions
--------------------
• Patient-level train/val split  → no slice-level data leakage
• Lazy slice loading             → minimal RAM footprint
• Frozen backbones               → only heads & final layers are tuned
• Dataset fingerprinting         → unchanged data is never retrained

Skip Logic
----------
Skips entirely if:
  1. All three council weight files exist (> 5 MB each), AND
  2. HYDRA_Volumetric_Finetune.json sentinel exists, AND
  3. Dataset fingerprint in the sentinel matches current dataset

Fixes Applied
-------------
• torch.cuda.amp.GradScaler  → torch.amp.GradScaler('cuda')
• torch.cuda.amp.autocast    → torch.amp.autocast('cuda')
• All deprecated AMP usage removed
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import nibabel as nib
import numpy as np
import pydicom
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms  # type: ignore[import-untyped]
from tqdm import tqdm

from hydra_core import (
    BRANCH_WEIGHTS,
    MedicalSwinAdapter,
    NO_TUMOR_CLASS_INDEX,
    fingerprint_directory,
    load_monai_adapter_checkpoint,
)


# ─── CONFIGURATION ────────────────────────────────────────────────────────────

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"

VOLUMETRIC_DATASET_DIR = Path("dataset_volumetric")
SENTINEL_PATH          = Path("HYDRA_Volumetric_Finetune.json")

WEIGHT_SWIN  = Path("HYDRA_Swin_Council.pth")
WEIGHT_CONV  = Path("HYDRA_ConvNext_Council.pth")
WEIGHT_MONAI = Path("HYDRA_MONAI_Council.pth")
REQUIRED_COUNCIL_WEIGHTS = [WEIGHT_SWIN, WEIGHT_CONV, WEIGHT_MONAI]

MAX_SLICES_PER_PATIENT = 240
MIN_SLICES_PER_PATIENT = 8
NIFTI_TRIM_FRACTION    = 0.15   # Trim 15% each end — removes skull/noise slices

INPUT_SIZE       = 256
BATCH_SIZE       = 12
FINE_TUNE_EPOCHS = 4
LEARNING_RATE    = 1e-4
WEIGHT_DECAY     = 1e-4

SUPPORTED_IMAGE_EXTENSIONS  = (".jpg", ".jpeg", ".png")
SUPPORTED_VOLUME_EXTENSIONS = (".nii", ".nii.gz")


# ─── DATA STRUCTURES ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PatientStudy:
    patient_id:  str
    label:       int
    source_kind: str   # "nifti" | "dicom_series" | "image_series"
    source_path: str
    slice_count: int


@dataclass(frozen=True)
class SliceRecord:
    patient_id:  str
    label:       int
    source_kind: str
    source_path: str
    slice_index: int | None = None


# ─── ARRAY UTILITIES ──────────────────────────────────────────────────────────

def _normalise_array(arr: np.ndarray) -> np.ndarray:
    arr   = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    scaled = (arr - lo) / (hi - lo + 1e-8)
    return np.uint8(np.clip(scaled * 255.0, 0, 255))


def _choose_uniform_indices(length: int, cap: int) -> list[int]:
    if length <= 0:
        return []
    if length <= cap:
        return list(range(length))
    return np.linspace(0, length - 1, num=cap, dtype=int).tolist()  # type: ignore[return-value]


# ─── DICOM / NIfTI HELPERS ────────────────────────────────────────────────────

def _nifti_slice_indices(path: Path, cap: int) -> list[int]:
    volume = nib.load(str(path))
    depth  = int(volume.shape[2]) if len(volume.shape) >= 3 else 0
    start  = int(depth * NIFTI_TRIM_FRACTION)
    end    = int(depth * (1 - NIFTI_TRIM_FRACTION))
    return [start + i for i in _choose_uniform_indices(max(end - start, 0), cap)]


def _sorted_dicom_files(folder: Path) -> list[Path]:
    paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".dcm"]
    sortable: list[tuple[float, Path]] = []
    for path in paths:
        try:
            hdr = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            if hasattr(hdr, "ImagePositionPatient"):
                key = float(hdr.ImagePositionPatient[2])
            elif hasattr(hdr, "InstanceNumber"):
                key = float(hdr.InstanceNumber)
            else:
                key = float(len(sortable))
        except Exception:
            key = float(len(sortable))
        sortable.append((key, path))
    sortable.sort(key=lambda x: x[0])
    return [p for _, p in sortable]


def _sorted_image_files(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


# ─── PATIENT DISCOVERY ────────────────────────────────────────────────────────

def _discover_patient_studies(root: Path) -> list[PatientStudy]:
    """
    Walk dataset_volumetric/tumor/ and dataset_volumetric/no_tumor/
    and build a list of PatientStudy records (one per patient).
    """
    studies: list[PatientStudy] = []
    class_map = {
        "tumor": 1, "Tumor": 1,
        "no_tumor": 0, "NoTumor": 0, "no-tumor": 0,
    }

    for class_name, label in class_map.items():
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue

        for entry in sorted(class_dir.iterdir()):
            patient_id = f"{class_name}/{entry.stem}"

            # NIfTI volume
            if entry.is_file() and (
                entry.name.lower().endswith(".nii.gz") or
                entry.name.lower().endswith(".nii")
            ):
                indices = _nifti_slice_indices(entry, MAX_SLICES_PER_PATIENT)
                if len(indices) >= MIN_SLICES_PER_PATIENT:
                    studies.append(PatientStudy(patient_id, label, "nifti", str(entry), len(indices)))
                continue

            if not entry.is_dir():
                continue

            # DICOM series
            dcm_files = _sorted_dicom_files(entry)
            if dcm_files:
                selected = _choose_uniform_indices(len(dcm_files), MAX_SLICES_PER_PATIENT)
                if len(selected) >= MIN_SLICES_PER_PATIENT:
                    studies.append(PatientStudy(patient_id, label, "dicom_series", str(entry), len(selected)))
                continue

            # Pre-sliced image folder
            img_files = _sorted_image_files(entry)
            selected = _choose_uniform_indices(len(img_files), MAX_SLICES_PER_PATIENT)
            if len(selected) >= MIN_SLICES_PER_PATIENT:
                studies.append(PatientStudy(patient_id, label, "image_series", str(entry), len(selected)))

    return studies


# ─── DATASET ──────────────────────────────────────────────────────────────────

class VolumetricSliceDataset(Dataset[tuple[Any, int, str]]):
    """Lazy-loading slice dataset built from patient-level splits."""

    def __init__(self, studies: list[PatientStudy], transform: Any = None) -> None:
        self.transform = transform
        self.samples: list[SliceRecord] = []
        self._nifti_cache: dict[str, Any] = {}

        for study in studies:
            self.samples.extend(self._index_study(study))

    def _index_study(self, study: PatientStudy) -> list[SliceRecord]:
        path = Path(study.source_path)
        records: list[SliceRecord] = []

        if study.source_kind == "nifti":
            for idx in _nifti_slice_indices(path, MAX_SLICES_PER_PATIENT):
                records.append(SliceRecord(study.patient_id, study.label, "nifti", str(path), idx))

        elif study.source_kind == "dicom_series":
            files = _sorted_dicom_files(path)
            for idx in _choose_uniform_indices(len(files), MAX_SLICES_PER_PATIENT):
                records.append(SliceRecord(study.patient_id, study.label, "dicom_file", str(files[idx]), None))

        elif study.source_kind == "image_series":
            files = _sorted_image_files(path)
            for idx in _choose_uniform_indices(len(files), MAX_SLICES_PER_PATIENT):
                records.append(SliceRecord(study.patient_id, study.label, "image_file", str(files[idx]), None))

        return records

    def _load_slice(self, record: SliceRecord) -> Image.Image:
        if record.source_kind == "nifti":
            if record.source_path not in self._nifti_cache:
                self._nifti_cache[record.source_path] = nib.load(record.source_path).get_fdata()
            volume: np.ndarray = self._nifti_cache[record.source_path]
            raw = volume[:, :, record.slice_index or 0]
            arr = _normalise_array(raw)
            rgb = np.stack([arr, arr, arr], axis=2)
            return Image.fromarray(rgb)

        elif record.source_kind == "dicom_file":
            ds = pydicom.dcmread(record.source_path)
            arr = _normalise_array(ds.pixel_array.astype(np.float32))
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=2)
            return Image.fromarray(arr)

        else:  # image_file
            img = cv2.imread(record.source_path)
            if img is None:
                raise IOError(f"Cannot read image: {record.source_path}")
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, int, str]:
        record = self.samples[idx]
        try:
            image = self._load_slice(record)
        except Exception:
            image = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), color=0)

        if self.transform:
            image = self.transform(image)
        return image, record.label, record.patient_id


# ─── SKIP LOGIC ───────────────────────────────────────────────────────────────

def _dataset_fingerprint() -> str:
    return fingerprint_directory(
        VOLUMETRIC_DATASET_DIR,
        extensions=SUPPORTED_IMAGE_EXTENSIONS + SUPPORTED_VOLUME_EXTENSIONS + (".dcm",),
    )


def volumetric_finetune_already_done(fingerprint: str) -> bool:
    weights_ok = all(
        p.exists() and p.stat().st_size > 5 * 1024 * 1024
        for p in REQUIRED_COUNCIL_WEIGHTS
    )
    if not weights_ok or not SENTINEL_PATH.exists():
        return False

    try:
        meta = json.loads(SENTINEL_PATH.read_text())
    except json.JSONDecodeError:
        return False

    if meta.get("dataset_fingerprint") == fingerprint:
        print(f"[SKIP] Fine-tune already completed ({meta.get('timestamp', '?')}).")
        print(f"       Fingerprint : {fingerprint[:16]}...")
        print(f"       Patients    : {meta.get('patient_count', '?')}")
        print(f"       Patient F1  : {meta.get('patient_macro_f1', '?')}")
        return True

    print("[INFO] Dataset changed — re-running fine-tune …")
    return False


# ─── FREEZE STRATEGY ──────────────────────────────────────────────────────────

def _freeze_backbone(model: nn.Module, model_kind: str) -> None:
    """Freeze all parameters, then unfreeze classification head(s)."""
    for p in model.parameters():
        p.requires_grad = False

    if model_kind == "swin":
        for p in model.head.parameters():          # type: ignore[union-attr]
            p.requires_grad = True
        if hasattr(model, "norm"):
            for p in model.norm.parameters():      # type: ignore[union-attr]
                p.requires_grad = True

    elif model_kind == "convnext":
        for p in model.head.parameters():          # type: ignore[union-attr]
            p.requires_grad = True
        if hasattr(model, "norm_pre"):
            for p in model.norm_pre.parameters():  # type: ignore[union-attr]
                p.requires_grad = True

    elif model_kind == "monai":
        for p in model.classifier.parameters():    # type: ignore[union-attr]
            p.requires_grad = True
        if hasattr(model.backbone.swinViT, "layers"):  # type: ignore[union-attr]
            for p in model.backbone.swinViT.layers[-1].parameters():  # type: ignore[union-attr]
                p.requires_grad = True


# ─── COUNCIL LOADING ──────────────────────────────────────────────────────────

def _load_pretrained_council() -> tuple[nn.Module, nn.Module, nn.Module]:
    missing = [str(p) for p in REQUIRED_COUNCIL_WEIGHTS if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing council weights (run 04_diagnostic_ensemble_training.py first): "
            + ", ".join(missing)
        )

    swin: nn.Module  = timm.create_model("swinv2_tiny_window8_256", pretrained=False, num_classes=5).to(DEVICE)
    conv: nn.Module  = timm.create_model("convnextv2_nano",          pretrained=False, num_classes=5).to(DEVICE)
    monai: nn.Module = MedicalSwinAdapter().to(DEVICE)

    swin.load_state_dict(torch.load(WEIGHT_SWIN,  map_location=DEVICE, weights_only=True))
    conv.load_state_dict(torch.load(WEIGHT_CONV,  map_location=DEVICE, weights_only=True))
    load_monai_adapter_checkpoint(monai, WEIGHT_MONAI, DEVICE, strict=False)

    _freeze_backbone(swin,  "swin")
    _freeze_backbone(conv,  "convnext")
    _freeze_backbone(monai, "monai")

    print(f"  [LOADED] SwinV2   ← {WEIGHT_SWIN}")
    print(f"  [LOADED] ConvNeXt ← {WEIGHT_CONV}")
    print(f"  [LOADED] MONAI    ← {WEIGHT_MONAI}")
    return swin, conv, monai


# ─── BINARY TUMOUR LOGIT ──────────────────────────────────────────────────────

def _tumor_logit(logits_5c: torch.Tensor) -> torch.Tensor:
    """Convert 5-class logits to a binary tumour probability logit."""
    probs          = torch.softmax(logits_5c.float(), dim=1)
    tumor_prob     = 1.0 - probs[:, NO_TUMOR_CLASS_INDEX]
    return torch.logit(torch.clamp(tumor_prob, 1e-6, 1 - 1e-6))


# ─── EVALUATION ───────────────────────────────────────────────────────────────

def _evaluate(
    swin: nn.Module,
    conv: nn.Module,
    monai: nn.Module,
    val_loader: DataLoader[Any],
) -> dict[str, float]:
    slice_truth: list[int] = []
    slice_preds: list[int] = []
    patient_truth: dict[str, int] = {}
    patient_scores: dict[str, list[np.ndarray]] = {}

    for m in [swin, conv, monai]:
        m.eval()

    with torch.no_grad():
        for images, labels, patient_ids in val_loader:
            images = images.to(DEVICE)
            labels_list: list[int] = labels.tolist()

            # Use updated non-deprecated AMP
            with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                p_s = torch.softmax(swin(images), dim=1)
                p_c = torch.softmax(conv(images), dim=1)

            p_m = torch.softmax(monai(images.float()), dim=1)
            consensus = (
                BRANCH_WEIGHTS[0] * p_s.float() +
                BRANCH_WEIGHTS[1] * p_c.float() +
                BRANCH_WEIGHTS[2] * p_m.float()
            )

            preds: list[int] = (consensus[:, NO_TUMOR_CLASS_INDEX] < 0.5).long().cpu().tolist()
            slice_truth.extend(labels_list)
            slice_preds.extend(preds)

            for i, pid in enumerate(patient_ids):
                patient_truth[pid] = labels_list[i]
                patient_scores.setdefault(pid, []).append(consensus[i].cpu().numpy())

    pat_labels: list[int] = []
    pat_preds:  list[int] = []
    for pid, scores in patient_scores.items():
        avg = np.mean(scores, axis=0)
        pat_labels.append(patient_truth[pid])
        pat_preds.append(int(avg[NO_TUMOR_CLASS_INDEX] < 0.5))

    return {
        "slice_accuracy":   round(float(accuracy_score(slice_truth, slice_preds)), 4),
        "slice_macro_f1":   round(float(f1_score(slice_truth, slice_preds, average="macro")), 4),
        "patient_accuracy": round(float(accuracy_score(pat_labels, pat_preds)), 4),
        "patient_macro_f1": round(float(f1_score(pat_labels, pat_preds, average="macro")), 4),
    }


# ─── FINE-TUNE ────────────────────────────────────────────────────────────────

def _finetune(
    swin: nn.Module,
    conv: nn.Module,
    monai: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
) -> dict[str, float]:
    criterion = nn.BCEWithLogitsLoss()
    trainable = [
        p for m in (swin, conv, monai)
        for p in m.parameters() if p.requires_grad
    ]
    optimizer = optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINE_TUNE_EPOCHS)
    scaler    = torch.amp.GradScaler("cuda") if AMP_ENABLED else None

    for epoch in range(FINE_TUNE_EPOCHS):
        for m in [swin, conv, monai]:
            m.train()

        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"[FT Epoch {epoch + 1:02d}/{FINE_TUNE_EPOCHS}]")

        for images, labels, _pids in progress:
            images = images.to(DEVICE)
            labels_f = labels.float().to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss = (
                        criterion(_tumor_logit(swin(images)), labels_f) +
                        criterion(_tumor_logit(conv(images)), labels_f) +
                        criterion(_tumor_logit(monai(images.float())), labels_f)
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = (
                    criterion(_tumor_logit(swin(images)), labels_f) +
                    criterion(_tumor_logit(conv(images)), labels_f) +
                    criterion(_tumor_logit(monai(images.float())), labels_f)
                )
                loss.backward()
                optimizer.step()

            epoch_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        print(f"  Epoch {epoch + 1:02d} mean loss: {epoch_loss / max(len(train_loader), 1):.4f}")

    metrics = _evaluate(swin, conv, monai, val_loader)
    print("\n[INFO] Volumetric fine-tune metrics:")
    print(f"  Slice Accuracy   : {metrics['slice_accuracy'] * 100:.2f}%")
    print(f"  Slice Macro F1   : {metrics['slice_macro_f1']:.4f}")
    print(f"  Patient Accuracy : {metrics['patient_accuracy'] * 100:.2f}%")
    print(f"  Patient Macro F1 : {metrics['patient_macro_f1']:.4f}")
    return metrics


# ─── ORCHESTRATOR ─────────────────────────────────────────────────────────────

def run_volumetric_finetune() -> None:
    """Full volumetric fine-tune pipeline.  Idempotent."""
    print("=" * 70)
    print("  HYDRA — Volumetric Brain Transfer Learning (Patient-Level)")
    print("=" * 70)

    if not VOLUMETRIC_DATASET_DIR.is_dir():
        print(
            f"[CRITICAL] '{VOLUMETRIC_DATASET_DIR}' not found.\n"
            "  Run: python 01b_volumetric_dataset_download.py\n"
            "  Or manually place studies under:\n"
            "    dataset_volumetric/tumor/<patient_id>/  or  .nii.gz\n"
            "    dataset_volumetric/no_tumor/<patient_id>/  or  .nii.gz"
        )
        return

    fingerprint = _dataset_fingerprint()
    if not fingerprint:
        print("[CRITICAL] No volumetric files found in dataset_volumetric/.")
        return

    if volumetric_finetune_already_done(fingerprint):
        return

    studies = _discover_patient_studies(VOLUMETRIC_DATASET_DIR)
    print(f"[INFO] Discovered {len(studies)} patient studies.")
    if len(studies) < 4:
        print("[CRITICAL] Need at least 4 patient studies (2 per class) to proceed.")
        return

    study_labels = [s.label for s in studies]
    try:
        train_studies, val_studies = train_test_split(
            studies, test_size=0.2, stratify=study_labels, random_state=42
        )
    except ValueError as e:
        print(f"[CRITICAL] Cannot stratify split: {e}")
        return

    tfm_train = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.12, contrast=0.12),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    tfm_eval = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    train_ds = VolumetricSliceDataset(train_studies, tfm_train)
    val_ds   = VolumetricSliceDataset(val_studies,   tfm_eval)

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("[CRITICAL] Dataset is empty after indexing.")
        return

    kw: dict[str, Any] = {"num_workers": 2, "pin_memory": AMP_ENABLED}
    train_loader: DataLoader[Any] = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  **kw)
    val_loader:   DataLoader[Any] = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, **kw)

    print(f"[INFO] Train patients: {len(train_studies)} | Val patients: {len(val_studies)}")
    print(f"[INFO] Train slices  : {len(train_ds)}  | Val slices  : {len(val_ds)}")
    print(f"[INFO] Device        : {DEVICE}")
    print(f"[INFO] Fingerprint   : {fingerprint[:16]}...")

    try:
        print("\n[INFO] Loading pretrained Council weights …")
        swin, conv, monai = _load_pretrained_council()
    except FileNotFoundError as e:
        print(f"[CRITICAL] {e}")
        return

    print(f"\n[INFO] Fine-tuning for {FINE_TUNE_EPOCHS} epochs …\n")
    metrics = _finetune(swin, conv, monai, train_loader, val_loader)

    torch.save(swin.state_dict(),  WEIGHT_SWIN)
    torch.save(conv.state_dict(),  WEIGHT_CONV)
    torch.save(monai.state_dict(), WEIGHT_MONAI)

    meta: dict[str, Any] = {
        "timestamp":           datetime.now().isoformat(),
        "dataset_fingerprint": fingerprint,
        "dataset_path":        str(VOLUMETRIC_DATASET_DIR),
        "patient_count":       len(studies),
        "slice_count":         len(train_ds) + len(val_ds),
        "epochs":              FINE_TUNE_EPOCHS,
        "batch_size":          BATCH_SIZE,
        "learning_rate":       LEARNING_RATE,
        "max_slices_per_patient": MAX_SLICES_PER_PATIENT,
        "device":              str(DEVICE),
        **metrics,
    }
    SENTINEL_PATH.write_text(json.dumps(meta, indent=2))

    print("\n[SUCCESS] Updated council checkpoints saved:")
    print(f"  → {WEIGHT_SWIN}")
    print(f"  → {WEIGHT_CONV}")
    print(f"  → {WEIGHT_MONAI}")
    print(f"  → {SENTINEL_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    run_volumetric_finetune()
