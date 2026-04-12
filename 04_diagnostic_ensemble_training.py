"""
04_diagnostic_ensemble_training.py — HYDRA Council (3-Model Ensemble) Training
===============================================================================
Trains the three-branch weighted voting ensemble for 5-class brain tumour
classification.

Council Architecture
--------------------
Branch 1 — SwinV2-Tiny      (40% vote)  — global texture via window attention
Branch 2 — ConvNeXtV2-Nano  (30% vote)  — spatial stability and rotation invariance
Branch 3 — MONAI Swin-UNETR (30% vote)  — medical-domain anatomical intuition

Consensus Formula
-----------------
P_final = 0.4 × P_swin + 0.3 × P_conv + 0.3 × P_monai

Skip Logic
----------
If ALL three .pth weight files exist and each is > 5 MB, training is skipped.
This protects 13+ hours of prior training from accidental overwrite.

Fixes Applied
-------------
• torch.cuda.amp.GradScaler → torch.amp.GradScaler('cuda')
• torch.cuda.amp.autocast  → torch.amp.autocast('cuda')
• Explicit Any / tuple typing for Pylance strict mode
• classification_report cast to str
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms  # type: ignore[import-untyped]
from tqdm import tqdm

from hydra_core import MedicalSwinAdapter


# ─── CONFIGURATION ────────────────────────────────────────────────────────────

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"

WEIGHT_SWIN  = "HYDRA_Swin_Council.pth"
WEIGHT_CONV  = "HYDRA_ConvNext_Council.pth"
WEIGHT_MONAI = "HYDRA_MONAI_Council.pth"

DATASET_DIR  = "dataset_ensemble"

INPUT_SIZE   = 256
BATCH_SIZE   = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS   = 12
MIN_WEIGHT_MB = 5.0


# ─── GUARD ────────────────────────────────────────────────────────────────────

def ensemble_already_trained() -> bool:
    """Return True when all three weight files exist and each is > MIN_WEIGHT_MB."""
    for path in [WEIGHT_SWIN, WEIGHT_CONV, WEIGHT_MONAI]:
        if not os.path.exists(path):
            return False
        if os.path.getsize(path) / (1024 * 1024) < MIN_WEIGHT_MB:
            return False

    print("[SKIP] All Council weights are present and valid:")
    for path in [WEIGHT_SWIN, WEIGHT_CONV, WEIGHT_MONAI]:
        size = os.path.getsize(path) / (1024 * 1024)
        print(f"       {path:<32}  {size:.1f} MB")
    print("       Training skipped.  Proceed to: python 05_volumetric_brain_finetune.py")
    return True


# ─── DATA PIPELINE ────────────────────────────────────────────────────────────

def _build_data_loaders() -> tuple[DataLoader[Any], DataLoader[Any], torch.Tensor, list[str]]:
    """
    Build stratified train / validation DataLoaders with class-weighted loss tensor.
    """
    train_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    full_dataset: Any = datasets.ImageFolder(DATASET_DIR, transform=train_transform)

    train_idx, val_idx = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        stratify=full_dataset.targets,
        random_state=42,
    )

    train_loader: DataLoader[Any] = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=AMP_ENABLED,
    )

    val_dataset: Any = datasets.ImageFolder(DATASET_DIR, transform=val_transform)
    val_loader: DataLoader[Any] = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=AMP_ENABLED,
    )

    # Compute inverse-frequency class weights to handle imbalance
    targets_arr = np.array(full_dataset.targets)
    weights = compute_class_weight(
        "balanced",
        classes=np.unique(targets_arr),
        y=targets_arr,
    )
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print(f"[INFO] Class weights: {class_weights.cpu().numpy().round(4)}")

    return train_loader, val_loader, class_weights, full_dataset.classes


# ─── TRAINING LOOP ────────────────────────────────────────────────────────────

def _train_branch(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    class_weights: torch.Tensor,
    branch_name: str,
    weight_path: str,
) -> None:
    """Train a single council branch with AMP, class-weighted CE loss, and val F1."""
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler    = torch.amp.GradScaler("cuda") if AMP_ENABLED else None
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_f1 = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(
            train_loader, desc=f"  [{branch_name}] Ep {epoch + 1}/{NUM_EPOCHS}", leave=False
        ):
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

            running_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                if AMP_ENABLED:
                    with torch.amp.autocast("cuda"):
                        logits = model(images)
                else:
                    logits = model(images)
                all_preds.extend(logits.argmax(1).cpu().tolist())
                all_labels.extend(labels.tolist())

        val_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
        avg_loss = running_loss / max(len(train_loader), 1)
        print(
            f"  [{branch_name}] Epoch {epoch + 1:02d}/{NUM_EPOCHS}  "
            f"Loss: {avg_loss:.4f}   Val Macro F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), weight_path)
            print(f"    ↑ Best F1 — checkpoint saved → '{weight_path}'")

    print(f"  [{branch_name}] Best Macro F1: {best_f1:.4f}")


# ─── MODEL BUILDERS ───────────────────────────────────────────────────────────

def _build_swin(num_classes: int) -> nn.Module:
    return timm.create_model(
        "swinv2_tiny_window8_256", pretrained=True, num_classes=num_classes
    ).to(DEVICE)


def _build_convnext(num_classes: int) -> nn.Module:
    return timm.create_model(
        "convnextv2_nano", pretrained=True, num_classes=num_classes
    ).to(DEVICE)


def _build_monai(num_classes: int) -> nn.Module:
    # MedicalSwinAdapter uses DIAGNOSTIC_LABELS length — num_classes should match
    assert num_classes == 5, "Council expects exactly 5 classes"
    return MedicalSwinAdapter().to(DEVICE)


# ─── ORCHESTRATOR ─────────────────────────────────────────────────────────────

def train_council() -> None:
    """Full Council training pipeline.  Idempotent."""
    print("=" * 70)
    print("  HYDRA — Diagnostic Council Training (SwinV2 + ConvNeXt + MONAI)")
    print("=" * 70)

    if ensemble_already_trained():
        return

    if not os.path.isdir(DATASET_DIR):
        print(f"[CRITICAL] Dataset '{DATASET_DIR}' not found.  Run 01_neuroimaging_data_acquisition.py first.")
        return

    print("[INFO] Building data loaders …")
    train_loader, val_loader, class_weights, class_names = _build_data_loaders()
    num_classes = len(class_names)
    print(f"[INFO] Classes ({num_classes}): {class_names}")
    print(f"[INFO] Training samples: {len(train_loader.dataset)}")
    print(f"[INFO] Validation samples: {len(val_loader.dataset)}\n")

    # Branch 1 — SwinV2-Tiny
    if not os.path.exists(WEIGHT_SWIN) or os.path.getsize(WEIGHT_SWIN) / (1024**2) < MIN_WEIGHT_MB:
        print("[BRANCH 1] SwinV2-Tiny …")
        swin = _build_swin(num_classes)
        _train_branch(swin, train_loader, val_loader, class_weights, "SwinV2", WEIGHT_SWIN)
    else:
        print(f"[SKIP] SwinV2 weights already exist: '{WEIGHT_SWIN}'")

    # Branch 2 — ConvNeXtV2-Nano
    if not os.path.exists(WEIGHT_CONV) or os.path.getsize(WEIGHT_CONV) / (1024**2) < MIN_WEIGHT_MB:
        print("\n[BRANCH 2] ConvNeXtV2-Nano …")
        conv = _build_convnext(num_classes)
        _train_branch(conv, train_loader, val_loader, class_weights, "ConvNeXt", WEIGHT_CONV)
    else:
        print(f"[SKIP] ConvNeXt weights already exist: '{WEIGHT_CONV}'")

    # Branch 3 — MONAI Swin-UNETR
    if not os.path.exists(WEIGHT_MONAI) or os.path.getsize(WEIGHT_MONAI) / (1024**2) < MIN_WEIGHT_MB:
        print("\n[BRANCH 3] MONAI Swin-UNETR …")
        monai = _build_monai(num_classes)
        _train_branch(monai, train_loader, val_loader, class_weights, "MONAI", WEIGHT_MONAI)
    else:
        print(f"[SKIP] MONAI weights already exist: '{WEIGHT_MONAI}'")

    # Final report
    print("\n" + "=" * 70)
    print("  Final Evaluation (Ensemble Weighted Vote)")
    print("=" * 70)

    swin_m = _build_swin(num_classes)
    swin_m.load_state_dict(torch.load(WEIGHT_SWIN, map_location=DEVICE, weights_only=True))
    conv_m = _build_convnext(num_classes)
    conv_m.load_state_dict(torch.load(WEIGHT_CONV, map_location=DEVICE, weights_only=True))
    from hydra_core import BRANCH_WEIGHTS, load_monai_adapter_checkpoint
    monai_m = _build_monai(num_classes)
    load_monai_adapter_checkpoint(monai_m, WEIGHT_MONAI, DEVICE, strict=False)
    for m in [swin_m, conv_m, monai_m]:
        m.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            p_s = torch.softmax(swin_m(images), dim=1)
            p_c = torch.softmax(conv_m(images), dim=1)
            p_m = torch.softmax(monai_m(images), dim=1)
            consensus = (
                BRANCH_WEIGHTS[0] * p_s +
                BRANCH_WEIGHTS[1] * p_c +
                BRANCH_WEIGHTS[2] * p_m
            )
            all_preds.extend(consensus.argmax(1).cpu().tolist())
            all_labels.extend(labels.tolist())

    report: str = str(classification_report(
        all_labels, all_preds, target_names=class_names, digits=4
    ))
    print(report)
    print("=" * 70)
    print("[SUCCESS] Council training complete.")


if __name__ == "__main__":
    train_council()
