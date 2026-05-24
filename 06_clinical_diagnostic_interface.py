"""
06_clinical_diagnostic_interface.py
======================================
Brain Tumor Detection — Clinical Dashboard (Dark Theme)

Pipeline:
  1. Ingest DICOM / NIfTI / MRI / CT slices
  2. Gatekeeper OOD screening (EfficientNet-B0)
  3. Diagnostic council vote per slice (SwinV2 + ConvNeXt + MONAI)
  4. Aggregate across slices → verdict
  5. Grad-CAM saliency map on peak-signal slice
  6. YOLO Hunter bounding-box overlay on ORIGINAL scan (not heatmap)
  7. PDF report with combined overlay
"""

from __future__ import annotations

import json
import os
import tempfile
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import cv2
import gradio as gr
import nibabel as nib       # type: ignore[import-untyped]
import numpy as np
import pydicom
import timm
import torch
import torch.nn as nn
from fpdf import FPDF, XPos, YPos
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.visualize.class_activation_maps import GradCAM
from PIL import Image, ImageOps, ImageDraw
from torchvision import models, transforms  # type: ignore[import-untyped]
from ultralytics import YOLO               # type: ignore[import-untyped]


# ── Constants ─────────────────────────────────────────────────────────────────

DIAGNOSTIC_LABELS: list[str] = [
    "Glioma", "Meningioma", "No Tumor",
    "Pituitary", "Tumor (Generic / CT)",
]
BRANCH_NAMES: list[str]    = ["SwinV2", "ConvNeXt", "MONAI"]
BRANCH_WEIGHTS: list[float] = [0.4, 0.3, 0.3]
NO_TUMOR_IDX: int           = 2
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


def _load_monai_ckpt(model: nn.Module, path: Path, device: torch.device) -> None:
    sd: Dict[str, torch.Tensor] = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(_remap(sd), strict=False)


# ── Runtime config ────────────────────────────────────────────────────────────

torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP         = DEVICE.type == "cuda"

MAX_SLICES  = 300
TRIM_FRAC   = 0.15
OOD_THRESH  = 0.70

MAX_FILES              = int(os.getenv("BTD_MAX_FILES", "512"))
MAX_TOTAL_UPLOAD_BYTES = int(os.getenv("BTD_MAX_TOTAL_MB", "2048")) * 1024 * 1024
MAX_SINGLE_FILE_BYTES  = int(os.getenv("BTD_MAX_FILE_MB", "512")) * 1024 * 1024
MAX_IMAGE_PIXELS       = int(os.getenv("BTD_MAX_IMAGE_PIXELS", "40000000"))
MAX_NIFTI_VOXELS       = int(os.getenv("BTD_MAX_NIFTI_VOXELS", "100000000"))

REPORT_DIR = Path(os.getenv("BTD_REPORT_DIR", "reports"))

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

W_GATE = (
    Path("Gatekeeper_Clinical.pth") if Path("Gatekeeper_Clinical.pth").exists()
    else Path("Gatekeeper_v1.pth")
)
W_SWIN  = Path("Swin_5C.pth")
W_CONV  = Path("ConvNext_5C.pth")
W_MONAI = Path("MONAI_5C.pth")
W_CLS   = Path("gatekeeper_class_map.json")
W_YOLO  = Path("runs/detect/tumor_localizer/weights/best.pt")

PROJECT_NAME = "Brain Tumor Detection"


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_gatekeeper() -> Tuple[Optional[nn.Module], Optional[Dict[str, int]]]:
    try:
        m = models.efficientnet_b0(weights=None)
        old_head: nn.Linear = cast(nn.Linear, m.classifier[1])  # type: ignore[index]
        in_f: int = int(old_head.in_features)
        m.classifier[1] = nn.Linear(in_f, 2)                     # type: ignore[index]
        m.load_state_dict(torch.load(W_GATE, map_location=DEVICE, weights_only=True))
        m = m.to(DEVICE).eval()
        if AMP: m = m.half()
        cls: Dict[str, int] = json.loads(W_CLS.read_text())
        print(f"  [OK] Gatekeeper  -> {W_GATE}"); return m, cls
    except Exception as e:
        print(f"  [WARN] Gatekeeper: {e}"); return None, None


def _load_council() -> Tuple[Optional[List[nn.Module]], Optional[nn.Module]]:
    try:
        swin: nn.Module = timm.create_model("swinv2_tiny_window8_256", pretrained=False, num_classes=5).to(DEVICE)
        swin.load_state_dict(torch.load(W_SWIN, map_location=DEVICE, weights_only=True))
        conv: nn.Module = timm.create_model("convnextv2_nano", pretrained=False, num_classes=5).to(DEVICE)
        conv.load_state_dict(torch.load(W_CONV, map_location=DEVICE, weights_only=True))
        monai: nn.Module = MedicalSwinAdapter().to(DEVICE)
        _load_monai_ckpt(monai, W_MONAI, DEVICE)
        if AMP:
            swin = swin.half()  # type: ignore[assignment]
            conv = conv.half()  # type: ignore[assignment]
        # MONAI stays float32 — SwinUNETR does not support FP16 reliably
        for m in (swin, conv, monai): m.eval()
        print(f"  [OK] SwinV2   -> {W_SWIN}")
        print(f"  [OK] ConvNeXt -> {W_CONV}")
        print(f"  [OK] MONAI    -> {W_MONAI}")
        return [swin, conv, monai], monai
    except Exception as e:
        print(f"  [ERROR] Council: {e}"); return None, None


def _load_hunter() -> Optional[Any]:
    if not W_YOLO.exists():
        print("  [WARN] Hunter not found - localisation disabled."); return None
    try:
        h = YOLO(str(W_YOLO)); print(f"  [OK] Hunter   -> {W_YOLO}"); return h
    except Exception as e:
        print(f"  [WARN] Hunter: {e}"); return None


print(f"\n[{PROJECT_NAME}] Starting on {DEVICE} ...")
SYS: Dict[str, Any] = {"gk": None, "gk_cls": None, "council": None, "explainer": None, "hunter": None}
SYS["gk"], SYS["gk_cls"]         = _load_gatekeeper()
SYS["council"], SYS["explainer"] = _load_council()
SYS["hunter"]                     = _load_hunter()
print(f"[{PROJECT_NAME}] Ready.\n")


# ── Transforms ────────────────────────────────────────────────────────────────

T_GATE = transforms.Compose([
    transforms.Resize((256, 256)), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
])
T_DIAG = transforms.Compose([
    transforms.Resize((256, 256)), transforms.CenterCrop(256),
    transforms.ToTensor(), transforms.Normalize([.5,.5,.5],[.5,.5,.5]),
])


# ── Image utilities ───────────────────────────────────────────────────────────

def _norm(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32); lo, hi = float(a.min()), float(a.max())
    return np.uint8(np.clip((a - lo) / (hi - lo + 1e-8) * 255, 0, 255))


def _skull_strip(rgb: np.ndarray) -> np.ndarray:
    try:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            largest = max(cnts, key=cv2.contourArea)
            clean = np.zeros_like(gray)
            cv2.drawContours(clean, [largest], -1, 255, -1)
            return cv2.bitwise_and(rgb, rgb, mask=clean)
    except Exception:
        pass
    return rgb


def _uni(length: int, cap: int) -> np.ndarray:
    return np.arange(length) if length <= cap else np.linspace(0, length-1, cap, dtype=int)


# ── Ingest ────────────────────────────────────────────────────────────────────

def _upload_path(f: Any) -> str:
    return str(f.name) if hasattr(f, "name") else str(f)


def _mb(n: int) -> float:
    return n / (1024 * 1024)


def _validate_uploads(file_list: List[Any]) -> Tuple[Optional[str], List[str]]:
    paths = [_upload_path(f) for f in file_list]
    if len(paths) > MAX_FILES:
        return f"Upload rejected: maximum {MAX_FILES} files per study.", paths

    total = 0
    allowed = (".dcm", ".nii", ".nii.gz", ".jpg", ".jpeg", ".png")
    for p in paths:
        low = p.lower()
        if not low.endswith(allowed):
            return f"Upload rejected: unsupported file type for `{Path(p).name}`.", paths
        try:
            size = os.path.getsize(p)
        except OSError:
            return f"Upload rejected: could not read `{Path(p).name}`.", paths
        if size > MAX_SINGLE_FILE_BYTES:
            return (
                f"Upload rejected: `{Path(p).name}` is {_mb(size):.1f} MB; "
                f"limit is {_mb(MAX_SINGLE_FILE_BYTES):.0f} MB."
            ), paths
        total += size

    if total > MAX_TOTAL_UPLOAD_BYTES:
        return (
            f"Upload rejected: total study size is {_mb(total):.1f} MB; "
            f"limit is {_mb(MAX_TOTAL_UPLOAD_BYTES):.0f} MB."
        ), paths
    return None, paths


def _load_nifti(path: str) -> List[Image.Image]:
    io = nib.load(path)         # type: ignore[reportPrivateImportUsage]
    shape = tuple(int(v) for v in io.shape[:3])  # type: ignore[attr-defined]
    if len(shape) < 3:
        return []
    voxels = int(np.prod(shape))
    if voxels > MAX_NIFTI_VOXELS:
        raise ValueError(
            f"NIfTI volume has {voxels:,} voxels; limit is {MAX_NIFTI_VOXELS:,}."
        )
    data = io.dataobj  # type: ignore[attr-defined]
    depth = int(shape[2])
    s, e  = int(depth * TRIM_FRAC), int(depth * (1 - TRIM_FRAC))
    idxs  = (_uni(e - s, MAX_SLICES) + s).astype(int)
    out: List[Image.Image] = []
    for i in idxs:
        arr = np.asarray(data[:, :, i])
        if arr.ndim > 2:
            arr = arr[..., 0]
        raw = _norm(arr)
        rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        out.append(Image.fromarray(_skull_strip(rgb)))
    return out


def _load_dicom(files: List[str]) -> List[Image.Image]:
    recs: list[tuple[float, str]] = []
    for fp in files:
        try:
            ds = pydicom.dcmread(fp, force=True, stop_before_pixels=True)  # type: ignore[union-attr]
            rows = int(getattr(ds, "Rows", 0) or 0)
            cols = int(getattr(ds, "Columns", 0) or 0)
            if rows and cols and rows * cols > MAX_IMAGE_PIXELS:
                continue
            key: float = (
                float(ds.ImagePositionPatient[2]) if hasattr(ds, "ImagePositionPatient")  # type: ignore[union-attr]
                else float(ds.InstanceNumber) if hasattr(ds, "InstanceNumber")             # type: ignore[union-attr]
                else float(len(recs))
            )
            recs.append((key, fp))
        except Exception:
            continue
    recs.sort(key=lambda x: x[0])
    dcms = [r for _, r in recs]
    idxs = _uni(len(dcms), MAX_SLICES).astype(int)
    out: List[Image.Image] = []
    for i in idxs:
        try:
            ds = pydicom.dcmread(dcms[i], force=True)  # type: ignore[union-attr]
            raw = _norm(ds.pixel_array.astype(np.float32))  # type: ignore[union-attr]
            rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB) if raw.ndim == 2 else raw
            out.append(Image.fromarray(_skull_strip(rgb)))
        except Exception:
            continue
    return out


def ingest(file_list: List[Any]) -> List[Image.Image]:
    if not file_list: return []
    paths: List[str] = [_upload_path(f) for f in file_list]
    dcm = [p for p in paths if p.lower().endswith(".dcm")]
    if dcm: return _load_dicom(dcm)
    for p in sorted(paths):
        if p.lower().endswith((".nii.gz", ".nii")): return _load_nifti(p)
    slices: List[Image.Image] = []
    for p in sorted(paths):
        if p.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                with Image.open(p) as img:
                    w, h = img.size
                    if w * h > MAX_IMAGE_PIXELS:
                        continue
                    rgb = img.convert("RGB")
                    slices.append(Image.fromarray(_skull_strip(np.array(rgb))))
            except Exception:
                continue
    idxs = _uni(len(slices), MAX_SLICES).astype(int)
    return [slices[i] for i in idxs]


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

def _gradcam(model: nn.Module, t: torch.Tensor, orig: np.ndarray) -> np.ndarray:
    with torch.enable_grad():
        try:
            cam = GradCAM(nn_module=model, target_layers="backbone.swinViT.layers.3")
            sal: Any = cam(x=t.float())
        except Exception:
            return orig
    h: np.ndarray = sal[0][0].detach().cpu().numpy()
    h = cv2.resize(h, (orig.shape[1], orig.shape[0]))
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)
    col: np.ndarray = cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(orig, 0.58, col, 0.42, 0)


def _yolo_overlay(img_rgb: np.ndarray, hunter: Any) -> Tuple[np.ndarray, int]:
    """Run YOLO Hunter on the ORIGINAL scan image and draw bounding boxes.

    Returns:
        Tuple of (annotated_image, detection_count)
    """
    det_count = 0
    try:
        # Resize to 640 for optimal YOLO detection, then map boxes back
        h_orig, w_orig = img_rgb.shape[:2]
        scale = 640 / max(h_orig, w_orig)
        new_w, new_h = int(w_orig * scale), int(h_orig * scale)
        resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        results = hunter(resized, verbose=False, conf=0.25)
        out = img_rgb.copy()

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                # Map coordinates back to original resolution
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1 = int(x1 / scale); y1 = int(y1 / scale)
                x2 = int(x2 / scale); y2 = int(y2 / scale)
                conf_val = float(box.conf[0].item())
                det_count += 1

                # Confidence-based color: high=cyan, medium=yellow, low=orange
                if conf_val >= 0.7:
                    color = (0, 255, 255)     # bright cyan
                elif conf_val >= 0.4:
                    color = (0, 220, 255)     # amber-yellow
                else:
                    color = (0, 165, 255)     # orange

                thickness = max(2, min(4, int(min(h_orig, w_orig) / 150)))

                # Draw main bounding box
                cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

                # Draw corner accents (modern look)
                corner_len = max(12, int(min(x2-x1, y2-y1) * 0.15))
                cv2.line(out, (x1, y1), (x1 + corner_len, y1), color, thickness + 1)
                cv2.line(out, (x1, y1), (x1, y1 + corner_len), color, thickness + 1)
                cv2.line(out, (x2, y1), (x2 - corner_len, y1), color, thickness + 1)
                cv2.line(out, (x2, y1), (x2, y1 + corner_len), color, thickness + 1)
                cv2.line(out, (x1, y2), (x1 + corner_len, y2), color, thickness + 1)
                cv2.line(out, (x1, y2), (x1, y2 - corner_len), color, thickness + 1)
                cv2.line(out, (x2, y2), (x2 - corner_len, y2), color, thickness + 1)
                cv2.line(out, (x2, y2), (x2, y2 - corner_len), color, thickness + 1)

                # Label with background
                label = f"Tumour {conf_val*100:.0f}%"
                font_scale = max(0.45, min(0.7, min(h_orig, w_orig) / 600))
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
                label_y = max(y1 - 8, th + 8)
                # Semi-transparent label background
                overlay = out.copy()
                cv2.rectangle(overlay, (x1, label_y - th - 6), (x1 + tw + 10, label_y + 4), color, -1)
                cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)
                cv2.putText(out, label, (x1 + 5, label_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)

        if det_count > 0:
            print(f"  [Hunter] {det_count} tumour region(s) detected")
        return out, det_count

    except Exception as e:
        print(f"  [WARN] YOLO overlay failed: {e}")
        traceback.print_exc()
        return img_rgb, 0


# ── PDF report ────────────────────────────────────────────────────────────────

def _pdf_report(
    verdict: str, confidence: float, entropy: float,
    primary_diagnosis: str, primary_prob: float,
    total_slices: int, analysed_slices: int,
    branch_notes: Dict[str, str],
    heatmap: Optional[np.ndarray],
    detection_img: Optional[np.ndarray],
    prob_dict: Dict[str, float],
    det_count: int,
) -> str:
    doc = FPDF()
    doc.add_page()

    # Header bar
    doc.set_fill_color(15, 40, 80)
    doc.rect(0, 0, 210, 30, "F")
    doc.set_text_color(255, 255, 255)
    doc.set_font("Helvetica", "B", 17)
    doc.set_y(9)
    doc.cell(0, 11, "Brain Tumor Detection - Clinical Report",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    doc.set_text_color(0, 0, 0)
    doc.set_y(36)
    doc.set_font("Helvetica", "", 9)
    doc.cell(0, 5,
             f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}  |  "
             f"Slices uploaded: {total_slices}  |  Slices analysed: {analysed_slices}  |  Device: {DEVICE}",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    doc.ln(3)

    # ── VERDICT BANNER ──
    is_tumor = "Tumor" in verdict and "No" not in verdict
    r, g, b = (196, 60, 60) if is_tumor else (40, 130, 80)
    doc.set_fill_color(r, g, b)
    doc.set_text_color(255, 255, 255)
    doc.set_font("Helvetica", "B", 14)
    doc.cell(0, 10,
             f"  RESULT: {verdict}",
             fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    doc.set_text_color(0, 0, 0)
    doc.ln(2)

    # ── PRIMARY DIAGNOSIS BOX ──
    doc.set_fill_color(245, 247, 250)
    doc.set_font("Helvetica", "B", 12)
    doc.cell(0, 8, f"  Primary Diagnosis:  {primary_diagnosis}  ({primary_prob*100:.1f}%)",
             fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    doc.set_font("Helvetica", "", 9)
    doc.cell(0, 6,
             f"  Overall Confidence: {confidence*100:.1f}%   |   "
             f"Uncertainty (Entropy): {entropy:.4f}   |   "
             f"YOLO Detections: {det_count}",
             fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    doc.ln(2)

    # ── CLINICAL INTERPRETATION ──
    doc.set_font("Helvetica", "B", 11)
    doc.cell(0, 7, "Clinical Interpretation",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    doc.set_font("Helvetica", "", 9)
    if primary_diagnosis == "No Tumor":
        interp = (
            f"The AI ensemble analysed {analysed_slices} slice(s) and found NO evidence of brain tumour. "
            f"The 'No Tumor' class scored {primary_prob*100:.1f}%, indicating a normal brain scan. "
            "Clinical correlation is recommended."
        )
    else:
        interp = (
            f"The AI ensemble analysed {analysed_slices} slice(s) and detected a possible {primary_diagnosis} "
            f"with {primary_prob*100:.1f}% probability. "
            f"The combined tumour probability across all classes is {confidence*100:.1f}%. "
            "Further imaging and biopsy recommended for confirmation."
        )
    doc.multi_cell(0, 4.5, interp)
    doc.ln(3)

    # ── PROBABILITY TABLE ──
    doc.set_font("Helvetica", "B", 11)
    doc.cell(0, 7, "Differential Probability Distribution",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    doc.set_font("Helvetica", "", 10)
    for lbl, val in prob_dict.items():
        # Highlight the PRIMARY DIAGNOSIS row
        if lbl == primary_diagnosis:
            if is_tumor:
                doc.set_fill_color(255, 235, 235)  # light red highlight for tumor
            else:
                doc.set_fill_color(230, 255, 235)  # light green highlight for no tumor
            doc.set_font("Helvetica", "B", 10)
            doc.cell(90, 6, f"  >> {lbl}", new_x=XPos.RIGHT, new_y=YPos.TOP, fill=True)
            doc.cell(0,  6, f"{val*100:.2f}%  << PRIMARY", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
            doc.set_fill_color(255, 255, 255)
            doc.set_font("Helvetica", "", 10)
        else:
            doc.cell(90, 6, f"  {lbl}", new_x=XPos.RIGHT, new_y=YPos.TOP)
            doc.cell(0,  6, f"{val*100:.2f}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    doc.ln(3)

    # ── BRANCH SUMMARY ──
    doc.set_font("Helvetica", "B", 11)
    doc.cell(0, 7, "AI Model Branch Summary",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    doc.set_font("Helvetica", "", 10)
    for bname, note in branch_notes.items():
        doc.cell(0, 6, f"  {bname}: {note}",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    with tempfile.TemporaryDirectory(prefix="btd_report_") as tmp_dir:
        tmp_root = Path(tmp_dir)

        # ── YOLO DETECTION IMAGE ──
        if detection_img is not None and det_count > 0:
            tmp_det = tmp_root / "detection.jpg"
            Image.fromarray(detection_img).save(tmp_det, quality=95)
            doc.ln(3)
            doc.set_font("Helvetica", "B", 11)
            doc.cell(0, 7, f"Tumour Localisation - {det_count} Region(s) Detected",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            doc.image(str(tmp_det), x=45, w=120)

        # ── SALIENCY MAP ──
        if heatmap is not None:
            tmp_heat = tmp_root / "heatmap.jpg"
            Image.fromarray(heatmap).save(tmp_heat, quality=95)
            doc.ln(3)
            doc.set_font("Helvetica", "B", 11)
            doc.cell(0, 7, "AI Saliency Map (Grad-CAM - Peak Signal Slice)",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            doc.image(str(tmp_heat), x=45, w=120)

        doc.ln(6)
        doc.set_font("Helvetica", "I", 8)
        doc.set_text_color(100, 100, 100)
        doc.multi_cell(0, 4,
            "For research and educational use only. This AI output is not a "
            "standalone medical diagnosis. Consult a qualified radiologist.")

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORT_DIR / f"BrainTumor_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.pdf"
        doc.output(str(out))
        return str(out)


# ── Markdown report ───────────────────────────────────────────────────────────

def _md_report(
    total_slices: int, analysed_slices: int,
    verdict: str, conf: float, entropy: float,
    probs: np.ndarray, history: Dict[str, List[np.ndarray]],
    det_count: int,
    primary_diagnosis: str = "",
) -> str:
    if total_slices == 1:
        slice_info = "**1 slice** uploaded and analysed"
    elif analysed_slices == total_slices:
        slice_info = f"**{total_slices} slices** uploaded and all analysed"
    else:
        slice_info = f"**{total_slices} slices** uploaded - **{analysed_slices} analysed** (uniform sampling)"

    top_idx = int(np.argmax(probs))
    if not primary_diagnosis:
        primary_diagnosis = DIAGNOSTIC_LABELS[top_idx]
    primary_prob = float(probs[top_idx])

    prob_rows = "\n".join(
        f"| **{l}** | **{v*100:.2f}%** | {'<< PRIMARY' if l == primary_diagnosis else ''} |"
        if l == primary_diagnosis else
        f"| {l} | {v*100:.2f}% | |"
        for l, v in zip(DIAGNOSTIC_LABELS, probs)
    )
    branch_rows: List[str] = []
    for name in BRANCH_NAMES:
        h = history[name]
        if not h: continue
        vals = np.array(h)
        means = np.mean(vals, 0)
        stab  = 1.0 - float(np.mean(np.std(vals, 0)))
        top   = int(np.argmax(means))
        burden = float(np.mean(np.argmax(vals, 1) != NO_TUMOR_IDX)) * 100
        branch_rows.append(
            f"| {name} | {DIAGNOSTIC_LABELS[top]} | {means[top]*100:.1f}% | "
            f"{stab*100:.1f}% | {burden:.1f}% |"
        )
    icon = "🔴" if "Tumor" in verdict and "No" not in verdict else "🟢"
    yolo_line = f"**YOLO Hunter:** {det_count} tumour region(s) localised" if det_count > 0 else "**YOLO Hunter:** No regions detected"

    # Clinical interpretation
    if primary_diagnosis == "No Tumor":
        interp = (
            f"The AI ensemble analysed **{analysed_slices} slice(s)** and found "
            f"**NO evidence of brain tumour**. The 'No Tumor' class scored "
            f"**{primary_prob*100:.1f}%**. Clinical correlation is recommended."
        )
    else:
        tumor_pct = (1.0 - float(probs[NO_TUMOR_IDX])) * 100
        interp = (
            f"The AI ensemble analysed **{analysed_slices} slice(s)** and detected "
            f"a possible **{primary_diagnosis}** with **{primary_prob*100:.1f}%** probability. "
            f"Combined tumour probability across all classes: **{tumor_pct:.1f}%**. "
            f"Further imaging and biopsy recommended."
        )

    return f"""# {icon} {verdict}

> **Primary Diagnosis: {primary_diagnosis} ({primary_prob*100:.1f}%)**

**Confidence:** {conf*100:.2f}% &nbsp;&nbsp;|&nbsp;&nbsp; **Uncertainty (Entropy):** {entropy:.4f}

**Input:** {slice_info} &nbsp;&nbsp;|&nbsp;&nbsp; **Device:** {DEVICE}

{yolo_line}

### Clinical Interpretation
{interp}

---

## Differential Probability Distribution
| Diagnosis | Probability | Note |
|-----------|-------------|------|
{prob_rows}

---

## AI Council Branch Analysis
| Branch | Primary Signal | Confidence | Stability | Tumour Burden |
|--------|----------------|------------|-----------|---------------|
{chr(10).join(branch_rows) or '| - | - | - | - | - |'}

---

*For research and educational use only. Always consult a qualified radiologist.*
"""


# ── Main inference pipeline ───────────────────────────────────────────────────

def run_diagnostic(
    file_list: Any,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, float]], Optional[str]]:
    """Returns: (markdown, heatmap_img, detection_img, prob_dict, pdf_path)"""
    if AMP: torch.cuda.empty_cache()

    if not file_list:
        return "## Upload a brain scan to begin analysis.", None, None, None, None

    files: List[Any] = list(file_list)
    upload_error, _ = _validate_uploads(files)
    if upload_error:
        return f"## {upload_error}", None, None, None, None

    try:
        slices = ingest(files)
    except Exception as e:
        return f"## Could not process upload.\n{e}", None, None, None, None
    total_slices = len(slices)

    if total_slices == 0:
        return "## No readable slices found.\nUpload DICOM / NIfTI / MRI / CT files.", None, None, None, None
    if SYS["council"] is None:
        return "## Models not loaded. Check weight files exist.", None, None, None, None

    running   = np.zeros(NUM_CLASSES, dtype=np.float64)
    valid_n   = 0
    rejected_n = 0
    best_heatmap: Optional[np.ndarray] = None
    best_original: Optional[np.ndarray] = None  # Store original for YOLO
    peak      = 0.0
    history: Dict[str, List[np.ndarray]] = {n: [] for n in BRANCH_NAMES}

    for sl in progress.tqdm(slices, desc=f"Analysing {total_slices} slices"):
        pil_sl: Image.Image = cast(Image.Image, sl)
        pil_sl.thumbnail((1024, 1024))

        # Gatekeeper OOD screening
        if SYS["gk"] is not None and SYS["gk_cls"] is not None:
            ood_i: int = cast(int, SYS["gk_cls"].get("NotBrain", 1))
            gt: torch.Tensor = T_GATE(pil_sl).unsqueeze(0).to(DEVICE)
            if AMP: gt = gt.half()
            gp = torch.softmax(SYS["gk"](gt), 1)
            if float(gp[0][ood_i].item()) > OOD_THRESH:
                rejected_n += 1
                continue

        t0: torch.Tensor = cast(torch.Tensor, T_DIAG(pil_sl))
        t1: torch.Tensor = cast(torch.Tensor, T_DIAG(ImageOps.mirror(pil_sl)))
        bat = torch.stack([t0, t1]).to(DEVICE)

        branch_p: List[np.ndarray] = []
        for bi, voter in enumerate(cast(List[nn.Module], SYS["council"])):
            with torch.no_grad():
                is_monai = voter is SYS["explainer"]
                if is_monai:
                    logits = voter(bat.float())
                else:
                    with torch.amp.autocast("cuda", enabled=AMP):  # type: ignore[attr-defined]
                        logits = voter(bat.half() if AMP else bat)
            p = torch.softmax(logits / 1.5, 1).mean(0).float().cpu().numpy()
            branch_p.append(p); history[BRANCH_NAMES[bi]].append(p)

        w: np.ndarray = np.sum([bw * bp for bw, bp in zip(BRANCH_WEIGHTS, branch_p)], axis=0)
        running += w; valid_n += 1

        ts = float(1.0 - w[NO_TUMOR_IDX])
        if ts > peak:
            peak = ts
            # Store the ORIGINAL image for YOLO (critical fix)
            best_original = np.array(pil_sl)
            # Generate Grad-CAM heatmap separately
            explainer: Optional[nn.Module] = cast(Optional[nn.Module], SYS["explainer"])
            if explainer is not None:
                best_heatmap = _gradcam(explainer, t0.unsqueeze(0).to(DEVICE), np.array(pil_sl))
            else:
                best_heatmap = np.array(pil_sl)

    if valid_n == 0:
        msg = (
            f"## All {total_slices} slice{'s' if total_slices != 1 else ''} rejected by Safety Gatekeeper.\n\n"
            f"**{rejected_n} of {total_slices}** slices did not pass the brain-scan check "
            f"(confidence > {OOD_THRESH*100:.0f}% NotBrain).\n\n"
            "Please upload MRI or CT brain scans only."
        )
        return msg, None, None, None, None

    avg       = running / valid_n
    top_idx   = int(np.argmax(avg))
    primary_diagnosis = DIAGNOSTIC_LABELS[top_idx]
    primary_prob      = float(avg[top_idx])
    tumor     = 1.0 - float(avg[NO_TUMOR_IDX])

    # Determine verdict: use the actual top class name
    if top_idx == NO_TUMOR_IDX:
        verdict = "No Tumor Detected"
        conf    = float(avg[NO_TUMOR_IDX])
    else:
        # Top class is a tumor type — name it explicitly
        verdict = f"Tumor Detected - {primary_diagnosis}"
        conf    = tumor

    entropy   = float(-np.sum(avg * np.log(avg + 1e-8)))
    prob_dict = {l: float(round(v, 4)) for l, v in zip(DIAGNOSTIC_LABELS, avg)}
    bnotes: Dict[str, str] = {}
    for name in BRANCH_NAMES:
        h = history[name]
        if h:
            m = np.mean(np.array(h), 0); top = int(np.argmax(m))
            bnotes[name] = f"{DIAGNOSTIC_LABELS[top]} | {m[top]*100:.1f}%"

    # Run YOLO on the ORIGINAL scan (not heatmap)
    detection_img: Optional[np.ndarray] = None
    det_count = 0
    if best_original is not None and SYS["hunter"] is not None:
        detection_img, det_count = _yolo_overlay(best_original, SYS["hunter"])
    elif best_original is not None:
        detection_img = best_original

    md  = _md_report(total_slices, valid_n, verdict, conf, entropy, avg, history, det_count, primary_diagnosis)
    pdf = _pdf_report(verdict, conf, entropy, primary_diagnosis, primary_prob,
                      total_slices, valid_n, bnotes,
                      best_heatmap, detection_img, prob_dict, det_count)
    return md, best_heatmap, detection_img, prob_dict, pdf


# ── Status HTML ───────────────────────────────────────────────────────────────

def _chip(label: str, ok: bool) -> str:
    cls = "chip-ok" if ok else "chip-off"
    ico = "✓" if ok else "✗"
    return f'<div class="chip {cls}"><span class="chip-ico">{ico}</span>{label}</div>'


def _status_html(n: int = 0) -> str:
    chips = "".join([
        _chip("Gatekeeper",  SYS["gk"]       is not None),
        _chip("Hunter/YOLO", SYS["hunter"]    is not None),
        _chip("SwinV2",      SYS["council"]   is not None),
        _chip("ConvNeXt",    SYS["council"]   is not None),
        _chip("MONAI",       SYS["explainer"] is not None),
    ])
    return (
        f'<div class="status-box">'
        f'<div class="stat-row">'
        f'<div class="stat"><span class="stat-n">{n}</span><span class="stat-l">Files uploaded</span></div>'
        f'<div class="stat"><span class="stat-n">{MAX_SLICES}</span><span class="stat-l">Max slices</span></div>'
        f'<div class="stat"><span class="stat-n">{str(DEVICE).upper()}</span><span class="stat-l">Device</span></div>'
        f'</div>'
        f'<div class="chip-row">{chips}</div>'
        f'</div>'
    )


def _on_upload(f: Any) -> str:
    files = list(f) if f else []
    upload_error, _ = _validate_uploads(files) if files else (None, [])
    if upload_error:
        return _status_html(len(files)) + f'<div class="warn-box">{upload_error}</div>'
    return _status_html(len(files))


# ── CSS (Dark Theme — Premium Glassmorphism) ──────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:       #080c14;
  --s1:       #0f1520;
  --s2:       #141c2b;
  --s3:       #1a2435;
  --s4:       #212d42;
  --border:   rgba(255,255,255,.06);
  --border2:  rgba(255,255,255,.1);
  --border3:  rgba(6,182,212,.2);
  --accent:   #06b6d4;
  --accent2:  #0891b2;
  --accent3:  #818cf8;
  --green:    #34d399;
  --green-bg: rgba(52,211,153,.1);
  --red:      #f87171;
  --red-bg:   rgba(248,113,113,.1);
  --amber:    #fbbf24;
  --text:     #e2e8f0;
  --text2:    #94a3b8;
  --muted:    #64748b;
  --r:        14px;
  --font:     'Inter', system-ui, sans-serif;
  --mono:     'JetBrains Mono', monospace;
  --shadow:   0 2px 8px rgba(0,0,0,.3), 0 8px 32px rgba(0,0,0,.2);
  --glow:     0 0 20px rgba(6,182,212,.15);
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font) !important;
}

.gradio-container {
  max-width: 98% !important;
  width: 98% !important;
  margin: 0 auto !important;
  padding: 20px 0 !important;
}
footer { display: none !important; }

/* ── Hero ── */
#hero {
  background: linear-gradient(145deg, #0c1220 0%, #111b2e 40%, #0e1628 70%, #0a1018 100%);
  border-bottom: 1px solid var(--border2);
  padding: 40px 60px 36px;
  position: relative; overflow: hidden;
  border-radius: 20px;
  margin-bottom: 20px;
}
#hero::before {
  content: '';
  position: absolute; inset: 0;
  background:
    radial-gradient(ellipse 60% 80% at 5% 50%, rgba(6,182,212,.08) 0%, transparent 60%),
    radial-gradient(ellipse 40% 60% at 85% 20%, rgba(129,140,248,.06) 0%, transparent 55%),
    radial-gradient(ellipse 30% 40% at 50% 80%, rgba(6,182,212,.04) 0%, transparent 50%);
  pointer-events: none;
}
#hero::after {
  content: '';
  position: absolute; top: 0; right: 0;
  width: 400px; height: 400px;
  background: radial-gradient(circle, rgba(6,182,212,.06) 0%, transparent 70%);
  border-radius: 50%;
  transform: translate(30%, -30%);
  pointer-events: none;
}
#hero > * { position: relative; z-index: 1; }

.btd-tag {
  display: inline-flex; align-items: center; gap: 8px;
  background: rgba(6,182,212,.1); border: 1px solid rgba(6,182,212,.25);
  color: var(--accent); padding: 6px 14px; border-radius: 999px;
  font-size: .72rem; font-weight: 700; letter-spacing: .1em;
  text-transform: uppercase; margin-bottom: 16px;
  backdrop-filter: blur(8px);
}
.dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--accent); box-shadow: 0 0 8px var(--accent);
  animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
  0%,100% { opacity: 1; box-shadow: 0 0 8px var(--accent); }
  50% { opacity: .4; box-shadow: 0 0 16px var(--accent); }
}

#hero h1 {
  font-size: clamp(1.7rem, 3.5vw, 2.5rem);
  font-weight: 800; letter-spacing: -.04em;
  color: #f1f5f9; margin: 0 0 12px;
  line-height: 1.2;
}
#hero h1 .hl {
  color: var(--accent) !important;
}
#hero p {
  font-size: .92rem; color: var(--text2);
  max-width: 680px; line-height: 1.75; margin: 0 0 22px;
}

.hero-pills { display: flex; gap: 9px; flex-wrap: wrap; margin-bottom: 24px; }
.hpill {
  background: rgba(6,182,212,.08); border: 1px solid rgba(6,182,212,.18);
  padding: 6px 14px; border-radius: 999px; font-size: .78rem;
  color: var(--accent); font-weight: 600;
  transition: all .2s ease;
}
.hpill:hover {
  background: rgba(6,182,212,.15);
  border-color: rgba(6,182,212,.35);
}

.hero-stats { display: flex; gap: 28px; flex-wrap: wrap; margin-top: 4px; }
.hm .v {
  font-size: 1.9rem; font-weight: 800;
  color: var(--accent) !important;
  font-family: var(--mono); line-height: 1;
}
.hm .l {
  font-size: .66rem; color: var(--text2);
  text-transform: uppercase; letter-spacing: .1em; margin-top: 5px;
}

/* ── Panels (glassmorphism cards) ── */
.panel {
  background: var(--s2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--r) !important; padding: 20px !important;
  box-shadow: var(--shadow) !important;
  margin-bottom: 10px !important;
  transition: border-color .3s ease !important;
}
.panel:hover {
  border-color: var(--border3) !important;
}
.ptitle {
  font-size: .74rem; font-weight: 700; letter-spacing: .1em;
  text-transform: uppercase; color: var(--accent); margin-bottom: 14px;
  display: flex; align-items: center; gap: 6px;
}

/* ── Run button ── */
#run-btn {
  background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
  border: none !important; color: #0a0e14 !important;
  font-weight: 800 !important; font-size: .95rem !important;
  height: 50px !important; border-radius: var(--r) !important;
  box-shadow: 0 4px 20px rgba(6,182,212,.35), var(--glow) !important;
  transition: all .25s ease !important;
  letter-spacing: .02em !important;
}
#run-btn:hover {
  filter: brightness(1.1) !important;
  box-shadow: 0 6px 28px rgba(6,182,212,.45), 0 0 30px rgba(6,182,212,.2) !important;
  transform: translateY(-1px) !important;
}

/* ── Status ── */
.status-box { }
.stat-row { display: flex; gap: 10px; margin-bottom: 12px; }
.stat {
  flex: 1; background: var(--s3); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px 10px; text-align: center;
}
.stat-n {
  display: block; font-size: 1.35rem; font-weight: 700;
  color: var(--accent); font-family: var(--mono);
}
.stat-l {
  display: block; font-size: .68rem; color: var(--text2);
  text-transform: uppercase; letter-spacing: .08em; margin-top: 3px;
}
.chip-row { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 2px; }
.chip {
  display: flex; align-items: center; gap: 5px;
  padding: 5px 11px; border-radius: 8px; font-size: .76rem;
  font-weight: 600; border: 1px solid;
  transition: all .2s ease;
}
.chip-ok  {
  background: var(--green-bg); border-color: rgba(52,211,153,.3);
  color: var(--green);
}
.chip-off {
  background: var(--red-bg); border-color: rgba(248,113,113,.25);
  color: var(--red);
}
.chip-ico { font-size: .7rem; }

.warn-box {
  margin-top: 12px; padding: 10px 12px; border-radius: 8px;
  background: rgba(248,113,113,.1); border: 1px solid rgba(248,113,113,.25);
  color: #fecdd3; font-size: .82rem; line-height: 1.45;
}

/* ── Report markdown ── */
.report-out .prose { color: var(--text) !important; }
.report-out .prose h1 {
  color: var(--text) !important; font-size: 1.5rem !important;
  border-bottom: 2px solid var(--accent) !important;
  padding-bottom: 12px !important;
  margin-bottom: 20px !important;
}
.report-out .prose h2 {
  color: var(--accent) !important; font-size: 1rem !important;
  text-transform: uppercase !important; letter-spacing: .08em !important;
  margin-top: 24px !important;
  margin-bottom: 12px !important;
}
.report-out .prose table {
  border-collapse: collapse !important; width: 100% !important;
  background: var(--s1) !important; border-radius: 8px !important;
  overflow: hidden !important;
}
.report-out .prose th {
  background: var(--s3) !important; color: var(--accent) !important;
  font-size: .78rem !important; text-transform: uppercase !important;
  letter-spacing: .06em !important;
  border-bottom: 1px solid var(--border2) !important;
  padding: 8px 12px !important;
}
.report-out .prose td {
  border-top: 1px solid var(--border) !important;
  padding: 7px 12px !important; font-size: .87rem !important;
  color: var(--text2) !important;
}
.report-out .prose hr {
  border-color: var(--border) !important;
  margin: 16px 0 !important;
}
.report-out .prose strong { color: var(--text) !important; }
.report-out .prose em { color: var(--muted) !important; }

/* ── Tabs ── */
.tabs .tab-nav {
  background: var(--s1) !important;
  border-bottom: 1px solid var(--border2) !important;
  border-radius: 12px 12px 0 0 !important;
  padding: 4px 8px 0 !important;
}
.tabs .tab-nav button {
  color: var(--muted) !important;
  font-weight: 600 !important; font-size: .82rem !important;
  padding: 10px 16px !important;
  border: none !important; border-radius: 10px 10px 0 0 !important;
  transition: all .2s ease !important;
  background: transparent !important;
}
.tabs .tab-nav button:hover {
  color: var(--text2) !important;
  background: rgba(6,182,212,.05) !important;
}
.tabs .tab-nav button.selected {
  color: var(--accent) !important;
  background: var(--s2) !important;
  border-bottom: 2px solid var(--accent) !important;
  box-shadow: 0 0 12px rgba(6,182,212,.1) !important;
}
.tabs .tabitem {
  background: var(--s2) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 12px 12px !important;
  padding: 32px 48px !important;
  min-height: 500px !important;
}

/* ── Mobile Responsiveness ── */
@media (max-width: 768px) {
  .gradio-container {
    width: 100% !important;
    max-width: 100% !important;
    padding: 0 !important;
  }
  #hero {
    padding: 24px 20px;
    border-radius: 0;
    margin-bottom: 12px;
  }
  #hero h1 { font-size: 1.8rem; }
  .hero-stats { gap: 16px; }
  .hm .v { font-size: 1.4rem; }
  
  .panel {
    padding: 16px !important;
    margin-bottom: 8px !important;
    border-radius: 0 !important;
  }
  
  .tabs .tab-nav button {
    padding: 8px 12px !important;
    font-size: .75rem !important;
  }
  .tabs .tabitem {
    padding: 16px !important;
    min-height: 400px !important;
    border-radius: 0 !important;
    border-left: none !important;
    border-right: none !important;
  }
  
  .stat-row {
    flex-direction: column;
  }
  
  .stat-n { font-size: 1.2rem; }
}

/* ── Override Gradio dark defaults ── */
.gradio-container label,
.gradio-container .label-wrap,
.gradio-container .svelte-1gfkn6j {
  color: var(--text2) !important;
}
.gradio-container .block {
  background: var(--s2) !important;
  border-color: var(--border) !important;
}

/* Ensure upload zone is dark-themed */
.gradio-container input[type="file"],
.gradio-container .upload-container,
.gradio-container .file-preview {
  background: var(--s3) !important;
  color: var(--text) !important;
  border-color: var(--border2) !important;
}
.gradio-container .upload-container:hover {
  border-color: var(--accent) !important;
  box-shadow: 0 0 16px rgba(6,182,212,.1) !important;
}

/* Image containers */
.gradio-container .image-container {
  background: var(--s1) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
}

/* Label component */
.gradio-container .label-container {
  background: var(--s2) !important;
}

/* ── Pipeline text ── */
.pipe-step {
  display: flex; align-items: flex-start; gap: 12px;
  font-size: .85rem; color: var(--text); padding: 8px 0;
  border-bottom: 1px solid var(--border);
  transition: color .2s ease;
  line-height: 1.5;
}
.pipe-step:hover { color: var(--accent); }
.pipe-step:last-child { border-bottom: none; }
.pipe-num {
  background: rgba(6,182,212,.15); color: var(--accent);
  font-weight: 700; font-size: .75rem;
  width: 24px; height: 24px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0; margin-top: 1px;
  border: 1px solid rgba(6,182,212,.2);
}

/* ── Footer ── */
.app-footer {
  text-align: center; padding: 16px;
  color: var(--muted); font-size: .74rem;
  border-top: 1px solid var(--border);
  background: var(--s1);
  margin-top: 16px;
  border-radius: 0 0 12px 12px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--s1); }
::-webkit-scrollbar-thumb { background: var(--s4); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent2); }
"""


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Brain Tumor Detection") as app:

    gr.HTML(f"""
    <div id="hero">
      <div class="btd-tag"><span class="dot"></span>Clinical AI System</div>
      <h1>Brain <span class="hl">Tumor</span> Detection</h1>
      <p>AI-powered brain tumour analysis from MRI, CT, DICOM series, and NIfTI volumes.
         Handles single images up to 300-slice full-brain studies with automatic aggregation.</p>
      <div class="hero-pills">
        <span class="hpill">EfficientNet-B0 Gatekeeper</span>
        <span class="hpill">YOLOv11n Localizer</span>
        <span class="hpill">SwinV2 + ConvNeXt + MONAI Ensemble</span>
        <span class="hpill">Grad-CAM Saliency</span>
        <span class="hpill">PDF Report</span>
      </div>
      <div class="hero-stats">
        <div class="hm"><div class="v">98.25%</div><div class="l">Accuracy</div></div>
        <div class="hm"><div class="v">0.9783</div><div class="l">Macro F1</div></div>
        <div class="hm"><div class="v">300</div><div class="l">Max Slices</div></div>
        <div class="hm"><div class="v">5</div><div class="l">Classes</div></div>
      </div>
    </div>
    """)

    with gr.Row(equal_height=False):
        # ── Left column: controls ──
        with gr.Column(scale=4, min_width=320):
            with gr.Group(elem_classes=["panel"]):
                gr.HTML('<div class="ptitle">📂 Upload Brain Scan</div>')
                upload = gr.File(
                    file_count="multiple",
                    label="DICOM (.dcm) / NIfTI (.nii/.nii.gz) / MRI/CT images",
                    height=200,
                )
                run_btn = gr.Button(
                    "🔬  Analyse Scan", elem_id="run-btn", variant="primary",
                )

            with gr.Group(elem_classes=["panel"]):
                gr.HTML('<div class="ptitle">⚙ Model Status</div>')
                status_out = gr.HTML(_status_html())

            with gr.Group(elem_classes=["panel"]):
                gr.HTML("""
                <div class="ptitle">📋 Pipeline</div>
                <div class="pipe-step"><div class="pipe-num">1</div><div>Safety Gatekeeper screens non-brain inputs</div></div>
                <div class="pipe-step"><div class="pipe-num">2</div><div>Council votes per slice (SwinV2 + ConvNeXt + MONAI)</div></div>
                <div class="pipe-step"><div class="pipe-num">3</div><div>Results aggregated across all slices</div></div>
                <div class="pipe-step"><div class="pipe-num">4</div><div>YOLO Hunter localises tumour regions on original scan</div></div>
                <div class="pipe-step"><div class="pipe-num">5</div><div>Grad-CAM saliency + bounding boxes + PDF generated</div></div>
                """)

        # ── Right column: results ──
        with gr.Column(scale=8):
            with gr.Tabs():
                with gr.Tab("📊  Diagnostic Report"):
                    report_md = gr.Markdown(
                        "### Upload a scan and press **Analyse Scan** to begin.\n\n"
                        "Supported: DICOM `.dcm` · NIfTI `.nii` / `.nii.gz` · "
                        "MRI/CT images `.jpg` / `.png`\n\n"
                        "Upload a single image **or** a full multi-slice study.",
                        elem_classes=["report-out"],
                    )
                with gr.Tab("🔬  AI Saliency"):
                    heatmap_img = gr.Image(label="Grad-CAM Saliency Map (Peak Signal Slice)")
                with gr.Tab("🎯  Tumor Localization"):
                    detection_img = gr.Image(label="YOLO Hunter - Bounding Box Detection (Original Scan)")
                with gr.Tab("📈  Probabilities"):
                    prob_lbl = gr.Label(num_top_classes=5, label="Class Probability Distribution")
                with gr.Tab("📄  PDF Export"):
                    pdf_out = gr.File(label="Download clinical report PDF", interactive=False)

    gr.HTML(
        '<div class="app-footer">'
        'Brain Tumor Detection - For research and educational use only &middot; Not a medical device'
        '</div>'
    )

    upload.change(fn=_on_upload, inputs=upload, outputs=status_out)
    run_btn.click(
        fn=run_diagnostic,
        inputs=upload,
        outputs=[report_md, heatmap_img, detection_img, prob_lbl, pdf_out],
    )


if __name__ == "__main__":
    server_name = os.getenv("BTD_SERVER_NAME", os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"))
    server_port = int(os.getenv("BTD_SERVER_PORT", "7860"))
    share = _env_bool("BTD_SHARE", False)
    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        theme=gr.themes.Base(
            primary_hue="cyan",
            secondary_hue="indigo",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
            font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
        ),
        css=CSS,
    )
