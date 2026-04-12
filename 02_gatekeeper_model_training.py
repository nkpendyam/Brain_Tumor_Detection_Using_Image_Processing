"""
02_gatekeeper_model_training.py — HYDRA Safety Gatekeeper Training
===================================================================
Trains the binary EfficientNet-B0 classifier that rejects out-of-distribution
inputs (chest X-rays, photographs) before they reach the diagnostic ensemble.

Architecture
------------
Model  : EfficientNet-B0 (ImageNet pretrained, fine-tuned)
Task   : Binary — Brain vs. NotBrain
Why B0 : >99% binary accuracy at 10× lower latency than a transformer.

Skip Logic
----------
If HYDRA_Gatekeeper_v1.pth already exists and is > 1 MB, the entire training
step is skipped.  The Council (04) and fine-tune (05) stages are unaffected.

Fixes Applied
-------------
• torch.cuda.amp.GradScaler → torch.amp.GradScaler('cuda')  [deprecated]
• torch.cuda.amp.autocast  → torch.amp.autocast('cuda')     [deprecated]
• classifier in_features cast to int                        [type safety]
• DataLoader / Any imports                                  [Pylance clean]
"""

from __future__ import annotations

import glob
import json
import os
import random
import shutil
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms  # type: ignore[import-untyped]


# ─── CONFIGURATION ────────────────────────────────────────────────────────────

DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED        = DEVICE.type == "cuda"

WEIGHT_OUTPUT_PATH = "HYDRA_Gatekeeper_v1.pth"
CLASS_MAP_PATH     = "gatekeeper_class_map.json"
STAGING_DIR        = "staging_gatekeeper"

BRAIN_SAMPLE_CAP   = 3_000
NEGATIVE_SAMPLE_CAP = 1_500

INPUT_RESOLUTION   = 224
BATCH_SIZE         = 32
LEARNING_RATE      = 1e-4
NUM_EPOCHS         = 5


# ─── GUARD ────────────────────────────────────────────────────────────────────

def gatekeeper_already_trained() -> bool:
    """Return True when valid weight file exists (> 1 MB)."""
    if os.path.exists(WEIGHT_OUTPUT_PATH):
        size_mb = os.path.getsize(WEIGHT_OUTPUT_PATH) / (1024 * 1024)
        if size_mb > 1.0:
            print(
                f"[SKIP] Gatekeeper weights found → '{WEIGHT_OUTPUT_PATH}' "
                f"({size_mb:.1f} MB).  Training skipped."
            )
            return True
    return False


# ─── DATASET PREPARATION ──────────────────────────────────────────────────────

def _build_staging_directory() -> None:
    """
    Assemble a balanced Brain / NotBrain staging folder from existing
    ensemble images and distractor datasets.
    """
    if os.path.exists(STAGING_DIR):
        shutil.rmtree(STAGING_DIR)

    os.makedirs(f"{STAGING_DIR}/Brain",    exist_ok=True)
    os.makedirs(f"{STAGING_DIR}/NotBrain", exist_ok=True)

    # Positive class
    all_brain: list[str] = (
        glob.glob("dataset_ensemble/*/*.jpg")  +
        glob.glob("dataset_ensemble/*/*.jpeg") +
        glob.glob("dataset_ensemble/*/*.png")
    )
    random.shuffle(all_brain)
    selected = all_brain[:BRAIN_SAMPLE_CAP]
    print(f"[INFO] Gatekeeper — Brain samples selected: {len(selected)}")
    for path in selected:
        shutil.copy(path, f"{STAGING_DIR}/Brain/")

    # Negative class
    xray_paths = glob.glob("dataset_negatives/*.jpg")[:NEGATIVE_SAMPLE_CAP]
    face_paths  = glob.glob("dataset_faces/*.jpg")[:NEGATIVE_SAMPLE_CAP]
    negatives   = xray_paths + face_paths
    print(f"[INFO] Gatekeeper — NotBrain samples selected: {len(negatives)}")
    for path in negatives:
        shutil.copy(path, f"{STAGING_DIR}/NotBrain/")


def _build_data_loader() -> tuple[DataLoader[Any], int]:
    """Construct augmented DataLoader and persist the class-index mapping."""
    augment = transforms.Compose([
        transforms.Resize((INPUT_RESOLUTION, INPUT_RESOLUTION)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset: Any = datasets.ImageFolder(STAGING_DIR, transform=augment)

    with open(CLASS_MAP_PATH, "w") as fh:
        json.dump(dataset.class_to_idx, fh, indent=2)

    loader: DataLoader[Any] = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=AMP_ENABLED,
    )
    return loader, len(dataset.classes)


# ─── MODEL ────────────────────────────────────────────────────────────────────

def _build_model(num_classes: int) -> nn.Module:
    """
    EfficientNet-B0 with a fine-tuned classification head.
    in_features is explicitly cast to int to satisfy strict type checkers.
    """
    model = models.efficientnet_b0(weights="DEFAULT")
    in_features: int = int(model.classifier[1].in_features)  # type: ignore[union-attr]
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model.to(DEVICE)


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def _train(model: nn.Module, loader: DataLoader[Any]) -> None:
    """Standard supervised training loop — AMP-aware, AdamW optimiser."""
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Use the updated non-deprecated AMP API
    scaler = torch.amp.GradScaler("cuda") if AMP_ENABLED else None

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss   = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            correct    += int((logits.argmax(1) == labels).sum().item())
            total      += labels.size(0)

        acc = 100.0 * correct / max(total, 1)
        print(
            f"  [Epoch {epoch + 1:02d}/{NUM_EPOCHS}]  "
            f"Loss: {total_loss / max(len(loader), 1):.4f}   "
            f"Accuracy: {acc:.2f}%"
        )


# ─── ORCHESTRATOR ─────────────────────────────────────────────────────────────

def train_gatekeeper() -> None:
    """
    Full Gatekeeper pipeline.  Idempotent — exits if weights already exist.
    """
    print("=" * 70)
    print("  HYDRA — Gatekeeper Training (EfficientNet-B0 Binary Classifier)")
    print("=" * 70)

    if gatekeeper_already_trained():
        return

    if not os.path.isdir("dataset_ensemble"):
        print("[CRITICAL] 'dataset_ensemble' not found.  Run 01_neuroimaging_data_acquisition.py first.")
        return

    print("[INFO] Building balanced staging directory …")
    _build_staging_directory()

    print("[INFO] Constructing DataLoader …")
    loader, num_classes = _build_data_loader()

    print(f"[INFO] Initialising EfficientNet-B0 on {DEVICE} …")
    model = _build_model(num_classes)

    print(f"[INFO] Training — {NUM_EPOCHS} epochs, batch size {BATCH_SIZE} …\n")
    _train(model, loader)

    torch.save(model.state_dict(), WEIGHT_OUTPUT_PATH)
    print(f"\n[SUCCESS] Gatekeeper weights → '{WEIGHT_OUTPUT_PATH}'")

    shutil.rmtree(STAGING_DIR, ignore_errors=True)
    print("[INFO] Staging directory cleaned.")
    print("=" * 70)


if __name__ == "__main__":
    train_gatekeeper()
