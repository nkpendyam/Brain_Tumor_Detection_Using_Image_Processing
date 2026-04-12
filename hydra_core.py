"""
hydra_core.py — HYDRA Shared Runtime Components
================================================
Shared model adapters, label metadata, checkpoint helpers, and directory
fingerprinting used across all training, evaluation, and UI modules.

Keeping these in one place guarantees that training, fine-tuning, and the
dashboard all use identical checkpoint formats and label definitions.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable

import torch
import torch.nn as nn
from monai.networks.nets.swin_unetr import SwinUNETR


# ─── LABEL REGISTRY ───────────────────────────────────────────────────────────

DIAGNOSTIC_LABELS: list[str] = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary",
    "Tumor (Generic / CT)",
]

BRANCH_NAMES: list[str]   = ["SwinV2", "ConvNeXt", "MONAI"]
BRANCH_WEIGHTS: list[float] = [0.4, 0.3, 0.3]

# Index of the "no tumour" class — used in volumetric skip logic
NO_TUMOR_CLASS_INDEX: int = 2


# ─── MONAI ADAPTER ────────────────────────────────────────────────────────────

class MedicalSwinAdapter(nn.Module):
    """
    Classification wrapper around the MONAI Swin-UNETR backbone.

    The SwinUNETR backbone produces a hierarchy of feature maps.  We take
    the final (deepest) feature map, apply global average pooling, then pass
    the flattened vector through a linear classification head.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = SwinUNETR(
            spatial_dims=2,
            in_channels=3,
            out_channels=14,
            feature_size=24,
        )
        # Input to classifier is 384 channels (Swin-UNETR smallest variant)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(384, len(DIAGNOSTIC_LABELS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features: torch.Tensor = self.backbone.swinViT(x, normalize=True)[-1]
        return self.classifier(features)


# ─── CHECKPOINT HELPERS ───────────────────────────────────────────────────────

def remap_monai_checkpoint_keys(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Align historic MONAI checkpoint key names with the current adapter layout.

    Old checkpoints may use 'swin.swinViT.*' or 'swin.*' prefixes.
    This remaps them to 'backbone.*' expected by MedicalSwinAdapter.
    """
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("swin.swinViT."):
            remapped[key.replace("swin.swinViT.", "backbone.swinViT.")] = value
        elif key.startswith("swin."):
            remapped[key.replace("swin.", "backbone.")] = value
        else:
            remapped[key] = value
    return remapped


def load_monai_adapter_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
    strict: bool = False,
) -> None:
    """Load a MONAI council checkpoint into the shared adapter."""
    raw: Dict[str, torch.Tensor] = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(remap_monai_checkpoint_keys(raw), strict=strict)


# ─── DIRECTORY FINGERPRINTING ─────────────────────────────────────────────────

def fingerprint_directory(
    root: str | Path,
    extensions: Iterable[str] | None = None,
) -> str:
    """
    Build a stable directory fingerprint from relative paths, file sizes,
    and modified times.

    Used by 05_volumetric_brain_finetune.py to decide whether to skip
    retraining when the dataset has not changed since the last run.

    Returns an empty string if the directory does not exist.
    """
    root_path = Path(root)
    digest = hashlib.sha256()
    allowed: set[str] | None = (
        None if extensions is None else {ext.lower() for ext in extensions}
    )

    if not root_path.exists():
        return ""

    for path in sorted(p for p in root_path.rglob("*") if p.is_file()):
        if allowed is not None:
            if not any(str(path).lower().endswith(ext) for ext in allowed):
                continue
        stat = path.stat()
        digest.update(str(path.relative_to(root_path)).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(int(stat.st_mtime)).encode("utf-8"))

    return digest.hexdigest()
