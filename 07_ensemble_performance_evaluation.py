"""
07_ensemble_performance_evaluation.py — HYDRA Council Evaluation
================================================================
Standalone zero-training evaluation of the trained Council ensemble.
Produces per-class and aggregate metrics plus a confusion matrix CSV.

Uses identical random_state=42 split as Script 04 — evaluates on the
same 20% holdout that was never seen during training.

Fixes Applied
-------------
• torch.cuda.amp.autocast → torch.amp.autocast('cuda')
• classification_report cast to str for type safety
"""

from __future__ import annotations

import csv
import os
from typing import List

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
from torchvision import datasets, transforms  # type: ignore[import-untyped]
from tqdm import tqdm

from hydra_core import BRANCH_WEIGHTS, MedicalSwinAdapter, load_monai_adapter_checkpoint


# ─── CONFIGURATION ────────────────────────────────────────────────────────────

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"

WEIGHT_SWIN  = "HYDRA_Swin_Council.pth"
WEIGHT_CONV  = "HYDRA_ConvNext_Council.pth"
WEIGHT_MONAI = "HYDRA_MONAI_Council.pth"

DATASET_DIR = "dataset_ensemble"
OUTPUT_CSV  = "HYDRA_Confusion_Matrix.csv"

INPUT_SIZE  = 256
BATCH_SIZE  = 32


# ─── DATA PIPELINE ────────────────────────────────────────────────────────────

def _build_val_loader() -> tuple[DataLoader, List[str]]:
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
        random_state=42,  # Must match Script 04 for valid out-of-sample estimate
    )

    loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=AMP_ENABLED,
    )
    return loader, dataset.classes


# ─── MODEL LOADING ────────────────────────────────────────────────────────────

def _load_council() -> tuple[nn.Module, nn.Module, nn.Module]:
    swin: nn.Module = timm.create_model(
        "swinv2_tiny_window8_256", pretrained=False, num_classes=5
    ).to(DEVICE)
    swin.load_state_dict(
        torch.load(WEIGHT_SWIN, map_location=DEVICE, weights_only=True)
    )

    conv: nn.Module = timm.create_model(
        "convnextv2_nano", pretrained=False, num_classes=5
    ).to(DEVICE)
    conv.load_state_dict(
        torch.load(WEIGHT_CONV, map_location=DEVICE, weights_only=True)
    )

    monai: nn.Module = MedicalSwinAdapter().to(DEVICE)
    load_monai_adapter_checkpoint(monai, WEIGHT_MONAI, DEVICE, strict=False)

    if AMP_ENABLED:
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


# ─── INFERENCE ────────────────────────────────────────────────────────────────

def _run_inference(
    swin: nn.Module,
    conv: nn.Module,
    monai: nn.Module,
    val_loader: DataLoader,
) -> tuple[List[int], List[int]]:
    preds:  List[int] = []
    truths: List[int] = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Running inference"):
            images = images.to(DEVICE)

            # Use updated non-deprecated AMP API
            with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                p_s = torch.softmax(swin(images.half() if AMP_ENABLED else images), dim=1)
                p_c = torch.softmax(conv(images.half() if AMP_ENABLED else images), dim=1)

            p_m = torch.softmax(monai(images.float()), dim=1)

            consensus = (
                BRANCH_WEIGHTS[0] * p_s.float() +
                BRANCH_WEIGHTS[1] * p_c.float() +
                BRANCH_WEIGHTS[2] * p_m.float()
            )
            preds.extend(torch.argmax(consensus, dim=1).cpu().tolist())
            truths.extend(labels.tolist())

    return truths, preds


# ─── CONFUSION MATRIX ─────────────────────────────────────────────────────────

def _save_confusion_matrix(gt: List[int], preds: List[int], classes: List[str]) -> None:
    cm = confusion_matrix(gt, preds)
    with open(OUTPUT_CSV, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([""] + classes)
        for row_name, row in zip(classes, cm):
            writer.writerow([row_name] + list(row))
    print(f"  Confusion matrix → '{OUTPUT_CSV}'")


# ─── ORCHESTRATOR ─────────────────────────────────────────────────────────────

def evaluate_ensemble() -> None:
    print("=" * 70)
    print("  HYDRA — Ensemble Performance Evaluation")
    print("=" * 70)

    missing = [p for p in [WEIGHT_SWIN, WEIGHT_CONV, WEIGHT_MONAI] if not os.path.exists(p)]
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
    print(f"  Validation samples: {len(val_loader.dataset)}")

    print("\n[INFO] Running weighted-vote inference …")
    ground_truth, predictions = _run_inference(swin, conv, monai, val_loader)

    acc = accuracy_score(ground_truth, predictions)
    f1  = f1_score(ground_truth, predictions, average="macro")
    rpt = str(classification_report(
        ground_truth, predictions, target_names=class_names, digits=4
    ))

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
