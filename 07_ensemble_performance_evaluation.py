"""
07_ensemble_performance_evaluation.py
========================================
Zero-training evaluation of the diagnostic council on the 20% holdout.
Weights: Swin_5C.pth | ConvNext_5C.pth | MONAI_5C.pth
Output:  BrainTumor_Confusion_Matrix.csv
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, cast

import timm
import torch
import torch.nn as nn
from monai.networks.nets.swin_unetr import SwinUNETR
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms  # type: ignore[import-untyped]
from tqdm import tqdm


# ── Constants ─────────────────────────────────────────────────────────────────

DIAGNOSTIC_LABELS: list[str] = [
    "Glioma", "Meningioma", "No Tumor",
    "Pituitary", "Tumor (Generic / CT)",
]
BRANCH_WEIGHTS: list[float] = [0.4, 0.3, 0.3]
NUM_CLASSES: int            = len(DIAGNOSTIC_LABELS)


# ── MONAI adapter ─────────────────────────────────────────────────────────────

class MedicalSwinAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = SwinUNETR(
            spatial_dims=2, in_channels=3, out_channels=14, feature_size=24,
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(384, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        feats: torch.Tensor = self.backbone.swinViT(x, normalize=True)[-1]
        return self.classifier(feats)


def _remap(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("swin.swinViT."):
            out[k.replace("swin.swinViT.", "backbone.swinViT.")] = v
        elif k.startswith("swin."):
            out[k.replace("swin.", "backbone.")] = v
        else:
            out[k] = v
    return out


def _load_monai(model: nn.Module, path: str, device: torch.device) -> None:
    sd: Dict[str, torch.Tensor] = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(_remap(sd), strict=False)


# ── Config ────────────────────────────────────────────────────────────────────

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP         = DEVICE.type == "cuda"

WEIGHT_SWIN  = "Swin_5C.pth"
WEIGHT_CONV  = "ConvNext_5C.pth"
WEIGHT_MONAI = "MONAI_5C.pth"

DATASET_DIR  = "dataset_ensemble"
OUTPUT_CSV   = "BrainTumor_Confusion_Matrix.csv"
INPUT_SIZE   = 256
BATCH_SIZE   = 32


# ── Data ──────────────────────────────────────────────────────────────────────

def _val_loader() -> tuple[DataLoader[Any], List[str]]:
    tfm = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    ds: Any = datasets.ImageFolder(DATASET_DIR, transform=tfm)
    targets: List[int] = cast(List[int], ds.targets)
    split = train_test_split(
        list(range(len(ds))), test_size=0.2, stratify=targets, random_state=42,
    )
    va_idx: List[int] = cast(List[int], split[1])
    va_sub: Dataset[Any] = Subset(ds, va_idx)
    ld: DataLoader[Any] = DataLoader(
        va_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=AMP,
    )
    return ld, cast(List[str], ds.classes)


# ── Models ────────────────────────────────────────────────────────────────────

def _load_council() -> tuple[nn.Module, nn.Module, nn.Module]:
    swin: nn.Module = timm.create_model(
        "swinv2_tiny_window8_256", pretrained=False, num_classes=5,
    ).to(DEVICE)
    swin.load_state_dict(torch.load(WEIGHT_SWIN, map_location=DEVICE, weights_only=True))

    conv: nn.Module = timm.create_model(
        "convnextv2_nano", pretrained=False, num_classes=5,
    ).to(DEVICE)
    conv.load_state_dict(torch.load(WEIGHT_CONV, map_location=DEVICE, weights_only=True))

    monai: nn.Module = MedicalSwinAdapter().to(DEVICE)
    _load_monai(monai, WEIGHT_MONAI, DEVICE)

    if AMP: swin.half().eval(); conv.half().eval()
    else:   swin.eval();        conv.eval()
    monai.eval()

    print(f"  [OK] SwinV2   ← {WEIGHT_SWIN}")
    print(f"  [OK] ConvNeXt ← {WEIGHT_CONV}")
    print(f"  [OK] MONAI    ← {WEIGHT_MONAI}")
    return swin, conv, monai


# ── Inference ─────────────────────────────────────────────────────────────────

def _infer(
    swin: nn.Module, conv: nn.Module, monai: nn.Module,
    loader: DataLoader[Any],
) -> tuple[List[int], List[int]]:
    preds: List[int] = []; truths: List[int] = []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Inference"):
            imgs = imgs.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=AMP):  # type: ignore[attr-defined]
                imgs_h = imgs.half() if AMP else imgs
                ps = torch.softmax(swin(imgs_h), 1)
                pc = torch.softmax(conv(imgs_h), 1)
            pm = torch.softmax(monai(imgs.float()), 1)
            cons = BRANCH_WEIGHTS[0]*ps.float() + BRANCH_WEIGHTS[1]*pc.float() + BRANCH_WEIGHTS[2]*pm.float()
            preds.extend(cast(List[int], cons.argmax(1).cpu().tolist()))
            truths.extend(cast(List[int], lbls.tolist()))
    return truths, preds


# ── Orchestrator ──────────────────────────────────────────────────────────────

def evaluate_ensemble() -> None:
    print("=" * 70)
    print("  Brain Tumor Detection — Ensemble Performance Evaluation")
    print("=" * 70)

    for p in [WEIGHT_SWIN, WEIGHT_CONV, WEIGHT_MONAI]:
        if not os.path.exists(p):
            print(f"[CRITICAL] Missing: {p}. Run 04 first."); return
    if not os.path.isdir(DATASET_DIR):
        print(f"[CRITICAL] '{DATASET_DIR}' not found."); return

    print(f"[INFO] Device: {DEVICE}")
    swin, conv, monai = _load_council()
    loader, classes   = _val_loader()
    print(f"  Validation samples: {len(loader.dataset)}\n")

    gt, pred = _infer(swin, conv, monai, loader)
    acc = accuracy_score(cast(Any, gt), cast(Any, pred))
    f1  = f1_score(cast(Any, gt), cast(Any, pred), average="macro")
    rpt: str = str(classification_report(
        cast(Any, gt), cast(Any, pred), target_names=classes, digits=4,
    ))

    print("=" * 70)
    print("  DIAGNOSTIC COUNCIL — PERFORMANCE REPORT")
    print("=" * 70)
    print(rpt)
    print(f"  Overall Accuracy : {acc*100:.2f}%")
    print(f"  Macro F1-Score   : {f1:.4f}")
    print("=" * 70)

    cm = confusion_matrix(cast(Any, gt), cast(Any, pred))
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + classes)
        for name, row in zip(classes, cm):
            w.writerow([name] + cast(List[Any], list(row)))
    print(f"\n  Confusion matrix → {OUTPUT_CSV}")
    print("[OK]  Evaluation complete.")


if __name__ == "__main__":
    evaluate_ensemble()
