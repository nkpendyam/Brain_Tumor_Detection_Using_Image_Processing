"""
Module  : 02_gatekeeper_model_training.py
Project : HYDRA — Clinical Brain Tumor Analysis System
Purpose : Train the Safety Gatekeeper — a binary EfficientNet-B0 classifier
          that rejects out-of-distribution (OOD) inputs before they reach the
          diagnostic ensemble.

Architecture Decision
---------------------
Model   : EfficientNet-B0  (ImageNet pretrained, fine-tuned)
Task    : Binary classification — Brain vs. NotBrain
Rationale:
  • Diagnostic ensemble models are over-specialised. If fed a chest X-ray
    or a photograph, they will still attempt to find a tumour, producing
    dangerous false positives.
  • EfficientNet-B0 offers >99 % binary accuracy at 10× lower inference
    latency than a transformer, making it ideal for a lightweight Gatekeeper.

Skip Behaviour
--------------
If the trained weight file 'HYDRA_Gatekeeper_v1.pth' already exists on disk
(size > 1 MB), training is skipped entirely.  Re-training is triggered only
when the file is absent or corrupted.

Output Artefacts
----------------
HYDRA_Gatekeeper_v1.pth   — Trained EfficientNet-B0 state dictionary
gatekeeper_class_map.json — Class-name → index mapping for the web interface
"""

import os
import glob
import json
import random
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ─── CONFIGURATION ────────────────────────────────────────────────────────────

DEVICE                = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHT_OUTPUT_PATH    = "HYDRA_Gatekeeper_v1.pth"
CLASS_MAP_OUTPUT_PATH = "gatekeeper_class_map.json"
STAGING_DIR           = "staging_gatekeeper"

BRAIN_SAMPLE_CAP      = 3000
NEGATIVE_SAMPLE_CAP   = 1500   # Each: X-rays and faces

INPUT_RESOLUTION      = 224
BATCH_SIZE            = 32
LEARNING_RATE         = 1e-4
NUM_EPOCHS            = 5       # Binary task converges rapidly on EfficientNet


# ─── GUARD: SKIP IF ALREADY TRAINED ──────────────────────────────────────────

def gatekeeper_already_trained() -> bool:
    """
    Return True when a valid weight file exists (size > 1 MB).
    This prevents accidental re-training after the 13+ hour initial session.
    """
    if os.path.exists(WEIGHT_OUTPUT_PATH):
        size_mb = os.path.getsize(WEIGHT_OUTPUT_PATH) / (1024 * 1024)
        if size_mb > 1.0:
            print(f"[SKIP] Gatekeeper weights found at '{WEIGHT_OUTPUT_PATH}' "
                  f"({size_mb:.1f} MB).  Skipping re-training.")
            return True
    return False


# ─── DATASET PREPARATION ──────────────────────────────────────────────────────

def _build_staging_directory() -> None:
    """
    Assemble a balanced Brain / NotBrain staging folder from existing
    ensemble and distractor datasets.
    """
    if os.path.exists(STAGING_DIR):
        shutil.rmtree(STAGING_DIR)

    os.makedirs(f"{STAGING_DIR}/Brain",    exist_ok=True)
    os.makedirs(f"{STAGING_DIR}/NotBrain", exist_ok=True)

    # Positive class — randomly sampled brain scans
    all_brain = (
        glob.glob("dataset_ensemble/*/*.jpg") +
        glob.glob("dataset_ensemble/*/*.jpeg") +
        glob.glob("dataset_ensemble/*/*.png")
    )
    random.shuffle(all_brain)
    selected_brain = all_brain[:BRAIN_SAMPLE_CAP]
    print(f"[INFO] Gatekeeper — Positive class: {len(selected_brain)} brain scans.")
    for path in selected_brain:
        shutil.copy(path, f"{STAGING_DIR}/Brain/")

    # Negative class — X-ray + facial distractors
    xray_distractor  = glob.glob("dataset_negatives/*.jpg")[:NEGATIVE_SAMPLE_CAP]
    face_distractor  = glob.glob("dataset_faces/*.jpg")[:NEGATIVE_SAMPLE_CAP]
    negative_samples = xray_distractor + face_distractor
    print(f"[INFO] Gatekeeper — Negative class: {len(negative_samples)} distractor images.")
    for path in negative_samples:
        shutil.copy(path, f"{STAGING_DIR}/NotBrain/")


def _build_data_loader() -> tuple:
    """
    Construct augmented DataLoader and save the class-index mapping.
    """
    transform = transforms.Compose([
        transforms.Resize((INPUT_RESOLUTION, INPUT_RESOLUTION)),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(STAGING_DIR, transform=transform)

    with open(CLASS_MAP_OUTPUT_PATH, "w") as fh:
        json.dump(dataset.class_to_idx, fh, indent=2)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, pin_memory=True)
    return loader, len(dataset.classes)


# ─── MODEL CONSTRUCTION ───────────────────────────────────────────────────────

def _build_gatekeeper_model(num_classes: int) -> nn.Module:
    """
    Load the ImageNet-pretrained EfficientNet-B0 and replace the
    1 000-class head with a num_classes-node linear layer.
    """
    model = models.efficientnet_b0(weights="DEFAULT")
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model.to(DEVICE)


# ─── TRAINING LOOP ────────────────────────────────────────────────────────────

def _execute_training(model: nn.Module, loader: DataLoader) -> None:
    """Standard supervised training loop with AdamW and CrossEntropyLoss."""
    optimizer  = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion  = nn.CrossEntropyLoss()
    scaler     = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss  = 0.0
        epoch_correct = 0
        epoch_total   = 0

        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            if scaler:
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

            epoch_loss    += loss.item()
            epoch_correct += (logits.argmax(1) == labels).sum().item()
            epoch_total   += labels.size(0)

        acc = 100 * epoch_correct / epoch_total
        print(f"  [Epoch {epoch + 1:02d}/{NUM_EPOCHS}]  "
              f"Loss: {epoch_loss:.4f}   Accuracy: {acc:.2f}%")


# ─── ORCHESTRATOR ─────────────────────────────────────────────────────────────

def train_gatekeeper() -> None:
    """
    Full Gatekeeper training pipeline.
    Idempotent — exits immediately if valid weights already exist.
    """
    print("=" * 70)
    print("  HYDRA — Gatekeeper Model Training (EfficientNet-B0)")
    print("=" * 70)

    if gatekeeper_already_trained():
        return

    # Validate upstream data
    if not os.path.isdir("dataset_ensemble"):
        print("[CRITICAL] 'dataset_ensemble' not found. Run "
              "01_neuroimaging_data_acquisition.py first.")
        return

    print("[INFO] Building balanced staging directory …")
    _build_staging_directory()

    print("[INFO] Constructing DataLoader …")
    loader, num_classes = _build_data_loader()

    print(f"[INFO] Initialising EfficientNet-B0 on {DEVICE} …")
    model = _build_gatekeeper_model(num_classes)

    print(f"[INFO] Starting training — {NUM_EPOCHS} epochs, "
          f"batch size {BATCH_SIZE} …\n")
    _execute_training(model, loader)

    torch.save(model.state_dict(), WEIGHT_OUTPUT_PATH)
    print(f"\n[SUCCESS] Gatekeeper weights saved → '{WEIGHT_OUTPUT_PATH}'")

    shutil.rmtree(STAGING_DIR, ignore_errors=True)
    print("[INFO] Staging directory cleaned up.")
    print("=" * 70)


if __name__ == "__main__":
    train_gatekeeper()
