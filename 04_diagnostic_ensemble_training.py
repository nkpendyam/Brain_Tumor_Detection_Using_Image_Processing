"""
04_diagnostic_ensemble_training.py
=====================================
Trains the 5-class diagnostic council:
  SwinV2-Tiny (40%) + ConvNeXtV2-Nano (30%) + MONAI Swin-UNETR (30%)

Output weights: Swin_5C.pth  |  ConvNext_5C.pth  |  MONAI_5C.pth
Skip  All three > 5 MB → skipped.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, cast

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from monai.networks.nets.swin_unetr import SwinUNETR
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms  # type: ignore[import-untyped]
from tqdm import tqdm


# ── Constants ─────────────────────────────────────────────────────────────────

DIAGNOSTIC_LABELS: list[str] = [
    "Glioma", "Meningioma", "No Tumor",
    "Pituitary", "Tumor (Generic / CT)",
]
BRANCH_NAMES: list[str]    = ["SwinV2", "ConvNeXt", "MONAI"]
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
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(384, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        feats: torch.Tensor = self.backbone.swinViT(x, normalize=True)[-1]
        return self.classifier(feats)


def _remap_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
    model.load_state_dict(_remap_keys(sd), strict=False)


# ── Config ────────────────────────────────────────────────────────────────────

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP          = DEVICE.type == "cuda"

WEIGHT_SWIN  = "Swin_5C.pth"
WEIGHT_CONV  = "ConvNext_5C.pth"
WEIGHT_MONAI = "MONAI_5C.pth"

DATASET_DIR  = "dataset_ensemble"
INPUT_SIZE   = 256
BATCH_SIZE   = 16
LR           = 1e-4
NUM_EPOCHS   = 12
MIN_MB       = 5.0


# ── Skip guard ────────────────────────────────────────────────────────────────

def _already_trained() -> bool:
    for p in [WEIGHT_SWIN, WEIGHT_CONV, WEIGHT_MONAI]:
        if not os.path.exists(p) or os.path.getsize(p) / 1e6 < MIN_MB:
            return False
    print("[SKIP] All council weights present:")
    for p in [WEIGHT_SWIN, WEIGHT_CONV, WEIGHT_MONAI]:
        print(f"       {p}  ({os.path.getsize(p)/1e6:.1f} MB)")
    return True


# ── Data ──────────────────────────────────────────────────────────────────────

def _loaders() -> tuple[DataLoader[Any], DataLoader[Any], torch.Tensor, list[str]]:
    tfm_tr = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    tfm_va = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    full_ds: Any = datasets.ImageFolder(DATASET_DIR, transform=tfm_tr)
    val_ds: Any  = datasets.ImageFolder(DATASET_DIR, transform=tfm_va)
    targets: list[int] = cast(list[int], full_ds.targets)

    split = train_test_split(
        list(range(len(full_ds))), test_size=0.2,
        stratify=targets, random_state=42,
    )
    tr_idx: list[int] = cast(list[int], split[0])
    va_idx: list[int] = cast(list[int], split[1])

    tr_sub: Dataset[Any] = Subset(full_ds, tr_idx)
    va_sub: Dataset[Any] = Subset(val_ds,  va_idx)

    tr_ld: DataLoader[Any] = DataLoader(
        tr_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=AMP,
    )
    va_ld: DataLoader[Any] = DataLoader(
        va_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=AMP,
    )

    tgt_arr = np.array(targets)
    w = compute_class_weight("balanced", classes=np.unique(tgt_arr), y=tgt_arr)
    cw = torch.tensor(w, dtype=torch.float32).to(DEVICE)
    print(f"[INFO] Class weights: {cw.cpu().numpy().round(3)}")
    return tr_ld, va_ld, cw, cast(list[str], full_ds.classes)


# ── Branch training ───────────────────────────────────────────────────────────

def _train_branch(
    model: nn.Module,
    tr: DataLoader[Any], va: DataLoader[Any],
    cw: torch.Tensor, name: str, out: str,
) -> None:
    opt    = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    crit   = nn.CrossEntropyLoss(weight=cw)
    sch    = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS)
    scaler = torch.amp.GradScaler("cuda") if AMP else None  # type: ignore[attr-defined]
    best   = 0.0

    for ep in range(NUM_EPOCHS):
        model.train()
        rl = 0.0
        for imgs, lbls in tqdm(tr, desc=f"  [{name}] ep{ep+1}/{NUM_EPOCHS}", leave=False):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            opt.zero_grad()
            if scaler is not None:
                with torch.amp.autocast("cuda"):  # type: ignore[attr-defined]
                    loss = crit(model(imgs), lbls)
                scaler.scale(loss).backward()   # type: ignore[union-attr]
                scaler.step(opt)                # type: ignore[union-attr]
                scaler.update()                 # type: ignore[union-attr]
            else:
                loss = crit(model(imgs), lbls)
                loss.backward(); opt.step()
            rl += float(loss.item())
        sch.step()  # type: ignore[union-attr]

        model.eval()
        preds: List[int] = []; gts: List[int] = []
        with torch.no_grad():
            for imgs, lbls in va:
                imgs = imgs.to(DEVICE)
                with torch.amp.autocast("cuda", enabled=AMP):  # type: ignore[attr-defined]
                    logits = model(imgs)
                preds.extend(cast(List[int], logits.argmax(1).cpu().tolist()))
                gts.extend(cast(List[int], lbls.tolist()))

        vf1 = float(f1_score(gts, preds, average="macro", zero_division=0))
        print(f"  [{name}] ep{ep+1:02d}/{NUM_EPOCHS}  loss={rl/max(len(tr),1):.4f}  val_f1={vf1:.4f}")
        if vf1 > best:
            best = vf1
            torch.save(model.state_dict(), out)
            print(f"    ↑ best → {out}")
    print(f"  [{name}] best F1 = {best:.4f}")


# ── Model builders ────────────────────────────────────────────────────────────

def _swin(nc: int) -> nn.Module:
    return timm.create_model("swinv2_tiny_window8_256", pretrained=True, num_classes=nc).to(DEVICE)

def _conv(nc: int) -> nn.Module:
    return timm.create_model("convnextv2_nano", pretrained=True, num_classes=nc).to(DEVICE)

def _monai_model(nc: int) -> nn.Module:
    assert nc == 5
    return MedicalSwinAdapter().to(DEVICE)


# ── Orchestrator ──────────────────────────────────────────────────────────────

def train_council() -> None:
    print("=" * 70)
    print("  Brain Tumor Detection — Council Training (SwinV2 + ConvNeXt + MONAI)")
    print("=" * 70)

    if _already_trained(): return
    if not os.path.isdir(DATASET_DIR):
        print(f"[CRITICAL] '{DATASET_DIR}' not found. Run 01 first."); return

    tr_ld, va_ld, cw, classes = _loaders()
    nc = len(classes)
    print(f"[INFO] Classes: {classes}  |  Train: {len(tr_ld.dataset)}  |  Val: {len(va_ld.dataset)}\n")

    if not os.path.exists(WEIGHT_SWIN) or os.path.getsize(WEIGHT_SWIN)/1e6 < MIN_MB:
        print("[BRANCH 1] SwinV2-Tiny …")
        _train_branch(_swin(nc), tr_ld, va_ld, cw, "SwinV2", WEIGHT_SWIN)
    else:
        print(f"[SKIP] {WEIGHT_SWIN} valid.")

    if not os.path.exists(WEIGHT_CONV) or os.path.getsize(WEIGHT_CONV)/1e6 < MIN_MB:
        print("\n[BRANCH 2] ConvNeXtV2-Nano …")
        _train_branch(_conv(nc), tr_ld, va_ld, cw, "ConvNeXt", WEIGHT_CONV)
    else:
        print(f"[SKIP] {WEIGHT_CONV} valid.")

    if not os.path.exists(WEIGHT_MONAI) or os.path.getsize(WEIGHT_MONAI)/1e6 < MIN_MB:
        print("\n[BRANCH 3] MONAI Swin-UNETR …")
        _train_branch(_monai_model(nc), tr_ld, va_ld, cw, "MONAI", WEIGHT_MONAI)
    else:
        print(f"[SKIP] {WEIGHT_MONAI} valid.")

    # Ensemble eval
    print("\n" + "=" * 70 + "\n  Ensemble Evaluation")
    ms = _swin(nc); ms.load_state_dict(torch.load(WEIGHT_SWIN, map_location=DEVICE, weights_only=True)); ms.eval()
    mc = _conv(nc); mc.load_state_dict(torch.load(WEIGHT_CONV, map_location=DEVICE, weights_only=True)); mc.eval()
    mm = _monai_model(nc); _load_monai(mm, WEIGHT_MONAI, DEVICE); mm.eval()

    preds: List[int] = []; gts: List[int] = []
    with torch.no_grad():
        for imgs, lbls in va_ld:
            imgs = imgs.to(DEVICE)
            p = (BRANCH_WEIGHTS[0]*torch.softmax(ms(imgs), 1) +
                 BRANCH_WEIGHTS[1]*torch.softmax(mc(imgs), 1) +
                 BRANCH_WEIGHTS[2]*torch.softmax(mm(imgs), 1))
            preds.extend(cast(List[int], p.argmax(1).cpu().tolist()))
            gts.extend(cast(List[int], lbls.tolist()))

    print(str(classification_report(gts, preds, target_names=classes, digits=4)))
    print("[OK]  Council training complete.")


if __name__ == "__main__":
    train_council()
