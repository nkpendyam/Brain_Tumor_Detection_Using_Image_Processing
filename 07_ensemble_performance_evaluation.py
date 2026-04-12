"""
Module  : 07_ensemble_performance_evaluation.py
Project : HYDRA — Clinical Brain Tumor Analysis System
Purpose : Standalone zero-training evaluation of the trained Council ensemble.

Why a Separate Evaluation Script?
-----------------------------------
Decoupling evaluation from the training loop allows rapid A/B testing of
voting strategies (e.g., equal vs weighted consensus) without triggering
the multi-hour training pipeline.  It also provides a clean, reproducible
audit trail of model performance for seminar / publication use.

Reproducibility Guarantee
--------------------------
By fixing random_state=42 (identical to Script 04), this script evaluates
on the same 20 % of images that were never shown to the model during training,
ensuring a valid out-of-sample performance estimate.

Metrics Reported
----------------
• Per-class Precision, Recall, F1-Score (4 decimal places)
• Overall Accuracy
• Macro-average F1-Score (primary clinical metric)
• Confusion matrix saved to 'HYDRA_Confusion_Matrix.csv'
"""

import csv
import os
from typing import List

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from tqdm import tqdm

from monai.networks.nets import SwinUNETR


# ─── CONFIGURATION ────────────────────────────────────────────────────────────

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHT_SWIN    = "HYDRA_Swin_Council.pth"
WEIGHT_CONV    = "HYDRA_ConvNext_Council.pth"
WEIGHT_MONAI   = "HYDRA_MONAI_Council.pth"

DATASET_DIR    = "dataset_ensemble"
OUTPUT_CSV     = "HYDRA_Confusion_Matrix.csv"

INPUT_SIZE     = 256
BATCH_SIZE     = 32

# Voting weights — must match 06_clinical_diagnostic_interface.py
WEIGHTS        = [0.4, 0.3, 0.3]


# ─── MONAI ADAPTER (identical to training scripts) ────────────────────────────

class MedicalSwinAdapter(nn.Module):
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


# ─── DATA PIPELINE ────────────────────────────────────────────────────────────

def _build_val_loader():
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)

    _, val_idx = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        stratify=dataset.targets,
        random_state=42,      # Must match Script 04
    )

    loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return loader, dataset.classes


# ─── MODEL LOADING ────────────────────────────────────────────────────────────

def _load_council():
    swin = timm.create_model(
        "swinv2_tiny_window8_256", pretrained=False, num_classes=5
    ).to(DEVICE)
    swin.load_state_dict(
        torch.load(WEIGHT_SWIN, map_location=DEVICE, weights_only=True)
    )

    conv = timm.create_model(
        "convnextv2_nano", pretrained=False, num_classes=5
    ).to(DEVICE)
    conv.load_state_dict(
        torch.load(WEIGHT_CONV, map_location=DEVICE, weights_only=True)
    )

    monai = MedicalSwinAdapter().to(DEVICE)
    monai.load_state_dict(
        torch.load(WEIGHT_MONAI, map_location=DEVICE, weights_only=True),
        strict=True,
    )

    if DEVICE.type == "cuda":
        swin.half().eval()
        conv.half().eval()
    else:
        swin.eval()
        conv.eval()
    monai.eval()

    print(f"  [OK] SwinV2   ← {WEIGHT_SWIN}")
    print(f"  [OK] ConvNeXt ← {WEIGHT_CONV}")
    print(f"  [OK] MONAI    ← {WEIGHT_MONAI}")
    return swin, conv, monai


# ─── INFERENCE LOOP ──────────────────────────────────────────────────────────

def _run_inference(swin, conv, monai, val_loader) -> tuple:
    predictions, ground_truth = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Running inference"):
            images = images.to(DEVICE)

            fp16 = images.half()  if DEVICE.type == "cuda" else images
            fp32 = images.float() if DEVICE.type == "cuda" else images

            if DEVICE.type == "cuda":
                with torch.amp.autocast("cuda"):
                    p_s = torch.softmax(swin(fp16), dim=1)
                    p_c = torch.softmax(conv(fp16), dim=1)
            else:
                p_s = torch.softmax(swin(fp32), dim=1)
                p_c = torch.softmax(conv(fp32), dim=1)

            p_m = torch.softmax(monai(fp32), dim=1)

            consensus = (WEIGHTS[0] * p_s) + (WEIGHTS[1] * p_c) + (WEIGHTS[2] * p_m)
            predictions.extend(torch.argmax(consensus, dim=1).cpu().numpy())
            ground_truth.extend(labels.numpy())

    return ground_truth, predictions


# ─── CONFUSION MATRIX EXPORT ─────────────────────────────────────────────────

def _save_confusion_matrix(gt: List, preds: List, class_names: List) -> None:
    cm = confusion_matrix(gt, preds)
    with open(OUTPUT_CSV, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([""] + class_names)
        for row_name, row in zip(class_names, cm):
            writer.writerow([row_name] + list(row))
    print(f"  Confusion matrix saved → '{OUTPUT_CSV}'")


# ─── ORCHESTRATOR ────────────────────────────────────────────────────────────

def evaluate_ensemble() -> None:
    print("=" * 70)
    print("  HYDRA — Ensemble Performance Evaluation")
    print("=" * 70)

    # Validate weight files
    missing = [p for p in [WEIGHT_SWIN, WEIGHT_CONV, WEIGHT_MONAI]
               if not os.path.exists(p)]
    if missing:
        print(f"[CRITICAL] Missing weight files: {missing}")
        print("  Run 04_diagnostic_ensemble_training.py first.")
        return

    if not os.path.isdir(DATASET_DIR):
        print(f"[CRITICAL] Dataset directory '{DATASET_DIR}' not found.")
        return

    print(f"[INFO] Hardware: {DEVICE}")
    print("[INFO] Loading Council weights …")
    swin, conv, monai = _load_council()

    print("[INFO] Building validation data loader …")
    val_loader, class_names = _build_val_loader()
    print(f"  Validation set size: {len(val_loader.dataset)} samples")

    print("\n[INFO] Running weighted-vote inference …")
    ground_truth, predictions = _run_inference(swin, conv, monai, val_loader)

    acc  = accuracy_score(ground_truth, predictions)
    f1   = f1_score(ground_truth, predictions, average="macro")
    rpt  = classification_report(
        ground_truth, predictions,
        target_names=class_names, digits=4,
    )

    print("\n" + "=" * 70)
    print("  COUNCIL CLINICAL PERFORMANCE CARD")
    print("=" * 70)
    print(rpt)
    print(f"  Overall Accuracy       : {acc * 100:.2f}%")
    print(f"  Overall Macro F1-Score : {f1:.4f}")
    print("=" * 70)

    _save_confusion_matrix(ground_truth, predictions, class_names)

    print("\n[SUCCESS] Evaluation complete.")


if __name__ == "__main__":
    evaluate_ensemble()
