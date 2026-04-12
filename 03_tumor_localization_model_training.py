"""
03_tumor_localization_model_training.py — HYDRA Tumour Localizer Training
=========================================================================
Trains the YOLOv11n spatial localizer (Hunter) that produces bounding-box
predictions on brain scans — telling clinicians WHERE the tumour is.

Model: YOLOv11n (Nano) via Ultralytics
  • Single-pass detection + bounding-box regression
  • Feature Pyramid Network for multi-scale tumour detection
  • 35 epochs with Mosaic augmentation

Skip Logic
----------
If runs/detect/hydra_tumor_localizer/weights/best.pt exists and is > 1 MB,
training is skipped.  TorchScript export is always re-validated.

YOLO Dataset
------------
Requires a labeled detection dataset in YOLO format.
Default config file: brain-tumor.yaml

If you have the dataset from a previous run at:
  ~/backup/datasets/brain-tumor/
Update brain-tumor.yaml to point to that path.
"""

from __future__ import annotations

import os

import torch
from ultralytics import YOLO  # type: ignore[import-untyped]


# ─── CONFIGURATION ────────────────────────────────────────────────────────────

YOLO_DATA_CONFIG = "brain-tumor.yaml"
CHECKPOINT_DIR   = "runs/detect/hydra_tumor_localizer"
BEST_WEIGHT_PATH = f"{CHECKPOINT_DIR}/weights/best.pt"

PRETRAINED_BASE  = "yolo11n.pt"

EPOCHS      = 35
IMAGE_SIZE  = 640
BATCH_SIZE  = 8
NUM_WORKERS = 2
DEVICE_ID   = 0  # Primary NVIDIA GPU (auto-selects CUDA:0)

MOSAIC_PROB = 0.5
MIXUP_PROB  = 0.0  # Disabled — anatomical blending is clinically invalid


# ─── GUARD ────────────────────────────────────────────────────────────────────

def localizer_already_trained() -> bool:
    """Return True when a valid YOLO checkpoint exists (> 1 MB)."""
    if os.path.exists(BEST_WEIGHT_PATH):
        size_mb = os.path.getsize(BEST_WEIGHT_PATH) / (1024 * 1024)
        if size_mb > 1.0:
            print(
                f"[SKIP] Localizer checkpoint found → '{BEST_WEIGHT_PATH}' "
                f"({size_mb:.1f} MB).  Training skipped."
            )
            return True
    return False


# ─── MODEL ────────────────────────────────────────────────────────────────────

def _load_model() -> YOLO:
    """Load existing checkpoint for fine-tuning, or pretrained nano base."""
    if os.path.exists(BEST_WEIGHT_PATH):
        print(f"[INFO] Resuming from checkpoint: {BEST_WEIGHT_PATH}")
        return YOLO(BEST_WEIGHT_PATH)
    print(f"[INFO] No checkpoint — initialising from {PRETRAINED_BASE}")
    return YOLO(PRETRAINED_BASE)


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def _run_training(model: YOLO) -> None:
    """Execute YOLOv11 training with clinical configuration."""
    print(f"\n[INFO] Launching YOLOv11 training — {EPOCHS} epochs …\n")
    model.train(
        data      = YOLO_DATA_CONFIG,
        epochs    = EPOCHS,
        imgsz     = IMAGE_SIZE,
        batch     = BATCH_SIZE,
        workers   = NUM_WORKERS,
        cache     = "disk",
        device    = DEVICE_ID,
        amp       = True,
        augment   = True,
        mosaic    = MOSAIC_PROB,
        mixup     = MIXUP_PROB,
        optimizer = "AdamW",
        lr0       = 0.001,
        name      = "hydra_tumor_localizer",
        exist_ok  = True,
    )


# ─── TORCHSCRIPT EXPORT ───────────────────────────────────────────────────────

def _export_torchscript() -> None:
    """Export best.pt to TorchScript for deployment."""
    if not os.path.exists(BEST_WEIGHT_PATH):
        print("[WARN] best.pt not found — export skipped.")
        return

    ts_path = BEST_WEIGHT_PATH.replace(".pt", ".torchscript")
    if os.path.exists(ts_path):
        print(f"[SKIP] TorchScript already exists: '{ts_path}'")
        return

    print("[INFO] Exporting best checkpoint → TorchScript …")
    export_model = YOLO(BEST_WEIGHT_PATH)
    exported = export_model.export(format="torchscript")
    print(f"[SUCCESS] TorchScript exported → {exported}")


# ─── ORCHESTRATOR ─────────────────────────────────────────────────────────────

def train_tumor_localizer() -> None:
    """Full localizer pipeline.  Idempotent."""
    print("=" * 70)
    print("  HYDRA — Tumour Localizer Training (YOLOv11n)")
    print("=" * 70)

    if not os.path.exists(YOLO_DATA_CONFIG):
        print(
            f"[CRITICAL] YOLO data config '{YOLO_DATA_CONFIG}' not found.\n"
            "  Ensure brain-tumor.yaml exists and 'path:' points to your\n"
            "  labeled detection dataset folder."
        )
        return

    torch.cuda.empty_cache()

    if not localizer_already_trained():
        model = _load_model()
        _run_training(model)

    _export_torchscript()

    print("\n[SUCCESS] Tumour Localizer pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    train_tumor_localizer()
