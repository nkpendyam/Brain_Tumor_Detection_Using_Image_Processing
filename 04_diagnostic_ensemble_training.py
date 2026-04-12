"""
Module  : 04_diagnostic_ensemble_training.py
Project : HYDRA — Clinical Brain Tumor Analysis System
Purpose : Train the three-branch weighted voting ensemble (Council) for
          5-class brain tumour classification.

Council Architecture
--------------------
Branch 1 — SwinV2-Tiny (40 % vote weight)
  Hierarchical window-attention transformer.  Best for capturing global
  tumour texture patterns across the full 256 × 256 feature map.

Branch 2 — ConvNeXtV2-Nano (30 % vote weight)
  Modernised convolutional backbone.  Provides spatial stability and
  corrects for rotational variance that transformers occasionally miss.

Branch 3 — MONAI Swin-UNETR (30 % vote weight)
  Medical-domain pretrained encoder.  Anatomical intuition from diverse
  volumetric medical imaging datasets gives it unique clinical texture
  awareness.

Soft Voting Formula (inference)
--------------------------------
P_consensus = (0.4 × P_swin) + (0.3 × P_conv) + (0.3 × P_monai)

Skip Behaviour
--------------
If ALL three weight files exist and each is > 5 MB, training is skipped.
This protects the 13+ hours of prior training from accidental erasure.

Output Artefacts
----------------
HYDRA_Swin_Council.pth    — SwinV2-Tiny state dictionary
HYDRA_ConvNext_Council.pth — ConvNeXtV2-Nano state dictionary
HYDRA_MONAI_Council.pth   — MONAI Swin-UNETR state dictionary
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from monai.networks.nets import SwinUNETR
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


# ─── CONFIGURATION ────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHT_SWIN    = "HYDRA_Swin_Council.pth"
WEIGHT_CONV    = "HYDRA_ConvNext_Council.pth"
WEIGHT_MONAI   = "HYDRA_MONAI_Council.pth"

DATASET_DIR    = "dataset_ensemble"

INPUT_SIZE     = 256
BATCH_SIZE     = 16
LEARNING_RATE  = 1e-4
NUM_EPOCHS     = 12
MIN_WEIGHT_MB  = 5.0     # Minimum file size to consider weights valid


# ─── GUARD: SKIP IF ALREADY TRAINED ──────────────────────────────────────────

def ensemble_already_trained() -> bool:
    """
    Return True when all three weight files exist and each exceeds MIN_WEIGHT_MB.
    """
    paths = [WEIGHT_SWIN, WEIGHT_CONV, WEIGHT_MONAI]
    for path in paths:
        if not os.path.exists(path):
            return False
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb < MIN_WEIGHT_MB:
            return False

    sizes = {p: os.path.getsize(p) / (1024 * 1024) for p in paths}
    print("[SKIP] All Council weights are present and valid:")
    for path, size in sizes.items():
        print(f"       {path:<30}  {size:.1f} MB")
    print("       Training is skipped.  Run 05_volumetric_brain_finetune.py "
          "to adapt the Council on whole-brain slice data.")
    return True


# ─── MONAI ADAPTER ───────────────────────────────────────────────────────────

class MedicalSwinAdapter(nn.Module):
    """
    Wraps the MONAI Swin-UNETR encoder and attaches a 5-class linear head.

    Naming Convention (must match 06_clinical_diagnostic_interface.py)
    -------------------------------------------------------------------
    self.backbone     → SwinUNETR medical encoder
    self.classifier   → AdaptivePool → Flatten → Linear(384, 5)
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


# ─── DATA PIPELINE ───────────────────────────────────────────────────────────

def _build_data_loaders():
    """
    Build stratified 80 / 20 train / validation loaders with class-weight
    computation for handling imbalanced tumour classes.
    """
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)

    train_idx, val_idx = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        stratify=dataset.targets,
        random_state=42,
    )

    # Class weights — penalise model more for missing rare classes (e.g. Pituitary)
    train_labels = [dataset.targets[i] for i in train_idx]
    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(train_labels),
        y=train_labels,
    )
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    loader_kwargs = dict(num_workers=2, pin_memory=True)
    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=BATCH_SIZE,
        shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=BATCH_SIZE,
        shuffle=False, **loader_kwargs
    )

    return train_loader, val_loader, weight_tensor, dataset.classes


# ─── MODEL INITIALISATION ────────────────────────────────────────────────────

def _build_ensemble():
    """
    Construct all three Council branches and restore any existing
    checkpoint weights to resume from prior training progress.
    """
    swin = timm.create_model(
        "swinv2_tiny_window8_256", pretrained=True, num_classes=5
    ).to(DEVICE)

    conv = timm.create_model(
        "convnextv2_nano", pretrained=True, num_classes=5
    ).to(DEVICE)

    monai = MedicalSwinAdapter().to(DEVICE)

    checkpoints = {
        "SwinV2":   (swin,  WEIGHT_SWIN),
        "ConvNeXt": (conv,  WEIGHT_CONV),
        "MONAI":    (monai, WEIGHT_MONAI),
    }

    for name, (model, path) in checkpoints.items():
        if os.path.exists(path):
            try:
                model.load_state_dict(
                    torch.load(path, map_location=DEVICE, weights_only=True),
                    strict=True,
                )
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"  [RESUME] {name:<10} — Loaded '{path}' ({size_mb:.1f} MB)")
            except Exception as err:
                print(f"  [WARNING] {name} weight load failed: {err}")
        else:
            print(f"  [NEW]    {name:<10} — Starting from pretrained ImageNet.")

    return swin, conv, monai


# ─── TRAINING LOOP ───────────────────────────────────────────────────────────

def _train_council(
    swin, conv, monai,
    train_loader, val_loader,
    class_weights, class_names,
) -> None:
    """
    Joint optimisation of all three branches.
    Losses are summed so the shared optimiser finds weights that improve
    all three architectures simultaneously.
    """
    optimizer = optim.AdamW(
        list(swin.parameters()) +
        list(conv.parameters()) +
        list(monai.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Label smoothing (0.1) prevents overconfident predictions.
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1,
    )

    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    for epoch in range(NUM_EPOCHS):
        swin.train(); conv.train(); monai.train()
        progress = tqdm(train_loader, desc=f"[Epoch {epoch + 1:02d}/{NUM_EPOCHS}]")

        for images, labels in progress:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast("cuda"):
                    loss = (
                        criterion(swin(images),  labels) +
                        criterion(conv(images),  labels) +
                        criterion(monai(images.float()), labels)
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = (
                    criterion(swin(images),  labels) +
                    criterion(conv(images),  labels) +
                    criterion(monai(images), labels)
                )
                loss.backward()
                optimizer.step()

            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

    # ── Validation ──────────────────────────────────────────────────────────
    print("\n[INFO] Running weighted-voting validation …")
    swin.eval(); conv.eval(); monai.eval()
    predictions, ground_truth = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)

            p_s = torch.softmax(swin(images),         dim=1)
            p_c = torch.softmax(conv(images),         dim=1)
            p_m = torch.softmax(monai(images.float()), dim=1)

            # Soft-vote with agreed weights
            consensus = (0.4 * p_s) + (0.3 * p_c) + (0.3 * p_m)
            predictions.extend(torch.argmax(consensus, dim=1).cpu().numpy())
            ground_truth.extend(labels.numpy())

    f1 = f1_score(ground_truth, predictions, average="macro")
    print("\n" + "=" * 70)
    print("  COUNCIL DIAGNOSTIC PERFORMANCE CARD")
    print("=" * 70)
    print(classification_report(
        ground_truth, predictions,
        target_names=class_names, digits=4,
    ))
    print(f"  Overall Macro F1-Score : {f1:.4f}")
    print("=" * 70)


# ─── ORCHESTRATOR ────────────────────────────────────────────────────────────

def train_diagnostic_ensemble() -> None:
    """
    Full Council training pipeline.
    Idempotent — exits immediately if all valid weights already exist.
    """
    print("=" * 70)
    print("  HYDRA — Diagnostic Ensemble Training (Council)")
    print("=" * 70)

    if ensemble_already_trained():
        return

    if not os.path.isdir(DATASET_DIR):
        print(f"[CRITICAL] Dataset directory '{DATASET_DIR}' not found. "
              "Run 01_neuroimaging_data_acquisition.py first.")
        return

    print("[INFO] Building stratified data loaders …")
    train_loader, val_loader, class_weights, class_names = _build_data_loaders()

    print("[INFO] Initialising Council branches …")
    swin, conv, monai = _build_ensemble()

    print(f"\n[INFO] Starting joint training — {NUM_EPOCHS} epochs on {DEVICE} …\n")
    _train_council(
        swin, conv, monai,
        train_loader, val_loader,
        class_weights, class_names,
    )

    torch.save(swin.state_dict(),  WEIGHT_SWIN)
    torch.save(conv.state_dict(),  WEIGHT_CONV)
    torch.save(monai.state_dict(), WEIGHT_MONAI)

    print(f"\n[SUCCESS] Council weights saved:")
    print(f"  → {WEIGHT_SWIN}")
    print(f"  → {WEIGHT_CONV}")
    print(f"  → {WEIGHT_MONAI}")
    print("=" * 70)


if __name__ == "__main__":
    train_diagnostic_ensemble()
