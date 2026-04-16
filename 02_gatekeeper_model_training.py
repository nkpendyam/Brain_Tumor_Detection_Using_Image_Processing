"""
02_gatekeeper_model_training.py
=================================
Trains the Safety Gatekeeper — EfficientNet-B0 binary classifier.
Saves weights to: Gatekeeper_v1.pth

Skip  Gatekeeper_v1.pth > 1 MB → skipped.
"""

from __future__ import annotations

import glob
import json
import os
import random
import shutil
from typing import Any, Optional, cast

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms  # type: ignore[import-untyped]


# ── Config ────────────────────────────────────────────────────────────────────

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP         = DEVICE.type == "cuda"

WEIGHT_PATH    = "Gatekeeper_v1.pth"
CLASS_MAP_PATH = "gatekeeper_class_map.json"
STAGING_DIR    = "staging_gatekeeper"

BRAIN_CAP    = 3_000
NEG_CAP      = 1_500
INPUT_RES    = 224
BATCH        = 32
LR           = 1e-4
EPOCHS       = 5


# ── Skip guard ────────────────────────────────────────────────────────────────

def _already_trained() -> bool:
    if os.path.exists(WEIGHT_PATH) and os.path.getsize(WEIGHT_PATH) / 1e6 > 1.0:
        mb = os.path.getsize(WEIGHT_PATH) / 1e6
        print(f"[SKIP] Gatekeeper — {WEIGHT_PATH} ({mb:.1f} MB). Training skipped.")
        return True
    return False


# ── Staging ───────────────────────────────────────────────────────────────────

def _build_staging() -> None:
    if os.path.exists(STAGING_DIR):
        shutil.rmtree(STAGING_DIR)
    os.makedirs(f"{STAGING_DIR}/Brain",    exist_ok=True)
    os.makedirs(f"{STAGING_DIR}/NotBrain", exist_ok=True)

    all_brain: list[str] = (
        glob.glob("dataset_ensemble/*/*.jpg")  +
        glob.glob("dataset_ensemble/*/*.jpeg") +
        glob.glob("dataset_ensemble/*/*.png")
    )
    random.shuffle(all_brain)
    for p in all_brain[:BRAIN_CAP]:
        shutil.copy(p, f"{STAGING_DIR}/Brain/")
    print(f"[INFO] Brain samples: {min(len(all_brain), BRAIN_CAP)}")

    xray = glob.glob("dataset_negatives/*.jpg")[:NEG_CAP]
    face = glob.glob("dataset_faces/*.jpg")[:NEG_CAP]
    for p in xray + face:
        shutil.copy(p, f"{STAGING_DIR}/NotBrain/")
    print(f"[INFO] NotBrain samples: {len(xray) + len(face)}")


def _make_loader() -> tuple[DataLoader[Any], int]:
    tfm = transforms.Compose([
        transforms.Resize((INPUT_RES, INPUT_RES)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds: Any = datasets.ImageFolder(STAGING_DIR, transform=tfm)
    with open(CLASS_MAP_PATH, "w") as fh:
        json.dump(ds.class_to_idx, fh, indent=2)
    loader: DataLoader[Any] = DataLoader(
        ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=AMP,
    )
    return loader, len(ds.classes)


# ── Model ─────────────────────────────────────────────────────────────────────

def _build_model(num_classes: int) -> nn.Module:
    m = models.efficientnet_b0(weights="DEFAULT")
    old_head: nn.Linear = cast(nn.Linear, m.classifier[1])  # type: ignore[index]
    in_feat: int = int(old_head.in_features)
    m.classifier[1] = nn.Linear(in_feat, num_classes)        # type: ignore[index]
    return m.to(DEVICE)


# ── Training ──────────────────────────────────────────────────────────────────

def _train(model: nn.Module, loader: DataLoader[Any]) -> None:
    opt  = optim.AdamW(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    scaler: Optional[torch.amp.GradScaler] = (  # type: ignore[name-defined]
        torch.amp.GradScaler("cuda") if AMP else None  # type: ignore[attr-defined]
    )

    for ep in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()

            if scaler is not None:
                with torch.amp.autocast("cuda"):  # type: ignore[attr-defined]
                    logits = model(imgs)
                    loss   = crit(logits, labels)
                scaler.scale(loss).backward()   # type: ignore[union-attr]
                scaler.step(opt)                # type: ignore[union-attr]
                scaler.update()                 # type: ignore[union-attr]
            else:
                logits = model(imgs)
                loss   = crit(logits, labels)
                loss.backward()
                opt.step()

            total_loss += float(loss.item())
            correct    += int((logits.argmax(1) == labels).sum().item())
            total      += int(labels.size(0))

        acc = 100.0 * correct / max(total, 1)
        print(
            f"  [Epoch {ep+1:02d}/{EPOCHS}]  "
            f"Loss: {total_loss / max(len(loader), 1):.4f}   Accuracy: {acc:.2f}%"
        )


# ── Orchestrator ──────────────────────────────────────────────────────────────

def train_gatekeeper() -> None:
    print("=" * 70)
    print("  Brain Tumor Detection — Gatekeeper Training (EfficientNet-B0)")
    print("=" * 70)

    if _already_trained():
        return

    if not os.path.isdir("dataset_ensemble"):
        print("[CRITICAL] dataset_ensemble/ not found. Run 01 first.")
        return

    _build_staging()
    loader, num_classes = _make_loader()
    print(f"[INFO] {num_classes} classes | Device: {DEVICE}")
    model = _build_model(num_classes)
    _train(model, loader)
    torch.save(model.state_dict(), WEIGHT_PATH)
    print(f"[OK]  Weights → {WEIGHT_PATH}")
    shutil.rmtree(STAGING_DIR, ignore_errors=True)
    print("=" * 70)


if __name__ == "__main__":
    train_gatekeeper()
