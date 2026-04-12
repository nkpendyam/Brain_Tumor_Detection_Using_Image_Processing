"""
Module  : 03_tumor_localization_model_training.py
Project : HYDRA — Clinical Brain Tumor Analysis System
Purpose : Train / fine-tune the YOLOv11 tumour-localization model (Hunter).

What Hunter Does
----------------
The diagnostic ensemble (Script 04) produces a class label, but clinicians
also need to know *where* the tumour is in the scan — for biopsy planning,
radiotherapy targeting, and resection guidance.  Hunter provides spatial
bounding-box predictions in a single forward pass.

Model Choice — YOLOv11n (Nano)
-------------------------------
• Single-pass: object detection and bounding-box regression in one step.
• Feature Pyramid Network (FPN): detects tumours at small, medium, and
  large scales simultaneously.
• Nano backbone: minimal latency, suitable for edge GPU deployment.
• 35 epochs with Mosaic augmentation: teaches focus on local tumour
  objects rather than global head morphology.

Skip Behaviour
--------------
If 'runs/detect/hydra_tumor_localizer/weights/best.pt' already exists and is
> 1 MB, the training phase is skipped and only the TorchScript export is
re-run (if needed).

Output Artefacts
----------------
runs/detect/hydra_tumor_localizer/weights/best.pt — Best YOLO checkpoint
runs/detect/hydra_tumor_localizer/weights/best.torchscript — Serialized model
"""

import os
import torch
from ultralytics import YOLO


# ─── CONFIGURATION ────────────────────────────────────────────────────────────

YOLO_DATA_CONFIG   = "brain-tumor.yaml"
CHECKPOINT_DIR     = "runs/detect/hydra_tumor_localizer"
BEST_WEIGHT_PATH   = f"{CHECKPOINT_DIR}/weights/best.pt"
LAST_WEIGHT_PATH   = f"{CHECKPOINT_DIR}/weights/last.pt"

PRETRAINED_BASE    = "yolo11n.pt"   # Nano backbone for low-latency deployment

# Training hyper-parameters
EPOCHS             = 35
IMAGE_SIZE         = 640
BATCH_SIZE         = 8
NUM_WORKERS        = 2
DEVICE_TARGET      = 0             # Primary NVIDIA GPU (falls back to CPU)

# Advanced augmentation
MOSAIC_PROBABILITY = 0.5           # Combines 4 scans → forces local focus
MIXUP_PROBABILITY  = 0.0           # Disabled — blended anatomy is clinically invalid


# ─── GUARD: SKIP IF ALREADY TRAINED ──────────────────────────────────────────

def localizer_already_trained() -> bool:
    """
    Return True if a valid best.pt checkpoint exists (size > 1 MB).
    Prevents accidental full re-training.
    """
    if os.path.exists(BEST_WEIGHT_PATH):
        size_mb = os.path.getsize(BEST_WEIGHT_PATH) / (1024 * 1024)
        if size_mb > 1.0:
            print(f"[SKIP] Localizer checkpoint found at '{BEST_WEIGHT_PATH}' "
                  f"({size_mb:.1f} MB).  Training will be skipped.")
            return True
    return False


# ─── MODEL INITIALISATION ─────────────────────────────────────────────────────

def _load_model() -> YOLO:
    """
    Load from the best existing checkpoint for fine-tuning, or initialise
    from the pretrained nano backbone if no checkpoint exists.
    """
    if os.path.exists(BEST_WEIGHT_PATH):
        print(f"[INFO] Resuming from existing checkpoint: {BEST_WEIGHT_PATH}")
        return YOLO(BEST_WEIGHT_PATH)

    print(f"[INFO] No checkpoint found — initialising from {PRETRAINED_BASE}")
    return YOLO(PRETRAINED_BASE)


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def _run_training(model: YOLO) -> None:
    """
    Execute the YOLO training loop with clinical configuration.
    """
    print(f"\n[INFO] Launching YOLOv11 training — {EPOCHS} epochs …\n")

    model.train(
        data      = YOLO_DATA_CONFIG,
        epochs    = EPOCHS,
        imgsz     = IMAGE_SIZE,
        batch     = BATCH_SIZE,
        workers   = NUM_WORKERS,
        cache     = "disk",         # Pre-cache MRI frames to disk to reduce IO wait
        device    = DEVICE_TARGET,

        amp       = True,           # Mixed precision — maximises GPU throughput
        augment   = True,           # Enable built-in geometric/colour augmentations
        mosaic    = MOSAIC_PROBABILITY,
        mixup     = MIXUP_PROBABILITY,

        optimizer = "AdamW",
        lr0       = 0.001,

        name      = "hydra_tumor_localizer",
        exist_ok  = True,           # Append to existing run folder
    )


# ─── TORCHSCRIPT EXPORT ───────────────────────────────────────────────────────

def _export_torchscript() -> None:
    """
    Serialise the best checkpoint into TorchScript format for deployment.
    TorchScript runs independently of the Python interpreter, reducing
    CPU overhead during intensive batch-inference sessions.
    """
    if not os.path.exists(BEST_WEIGHT_PATH):
        print("[WARNING] best.pt not found — export skipped.")
        return

    ts_path = BEST_WEIGHT_PATH.replace(".pt", ".torchscript")
    if os.path.exists(ts_path):
        print(f"[SKIP] TorchScript already exported at '{ts_path}'.")
        return

    print("[INFO] Exporting best checkpoint to TorchScript …")
    final_model = YOLO(BEST_WEIGHT_PATH)
    exported = final_model.export(format="torchscript")
    print(f"[SUCCESS] TorchScript model exported → {exported}")


# ─── ORCHESTRATOR ─────────────────────────────────────────────────────────────

def train_tumor_localizer() -> None:
    """
    Full tumour-localizer training pipeline.
    Idempotent — skips the training phase if valid weights already exist,
    then re-evaluates the TorchScript export step.
    """
    print("=" * 70)
    print("  HYDRA — Tumour Localizer Training (YOLOv11n)")
    print("=" * 70)

    # Validate data configuration
    if not os.path.exists(YOLO_DATA_CONFIG):
        print(f"[CRITICAL] YOLO data config '{YOLO_DATA_CONFIG}' not found. "
              "Please create this file before running training.")
        return

    torch.cuda.empty_cache()

    already_done = localizer_already_trained()

    if not already_done:
        model = _load_model()
        _run_training(model)
    else:
        print("[INFO] Proceeding directly to export validation …")

    _export_torchscript()

    print("\n[SUCCESS] Tumour Localizer pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    train_tumor_localizer()
