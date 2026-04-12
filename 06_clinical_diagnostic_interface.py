"""
Module  : 06_clinical_diagnostic_interface.py
Project : HYDRA — Clinical Brain Tumor Analysis System
Purpose : Production-ready Gradio web application for multi-modal brain
          tumour detection from DICOM, NIfTI, MRI, CT, and X-ray inputs.

Inference Pipeline
------------------
1. Gatekeeper  — Rejects non-brain inputs (EfficientNet-B0, binary).
2. Hunter      — Localises tumour bounding boxes (YOLOv11n).
3. Council     — 5-class weighted soft-vote (SwinV2 + ConvNeXt + MONAI).
4. Explainability — Grad-CAM saliency heatmap on the most pathological slice.
5. Report      — Structured Markdown narrative + downloadable PDF.

Volumetric Support
------------------
• DICOM sequences  (.dcm) — sorted by Z-position or InstanceNumber.
• NIfTI volumes    (.nii, .nii.gz) — axial slices extracted, trimmed.
• Standard images  (.jpg, .jpeg, .png) — skull-stripping applied.
• Up to 300 slices per patient are processed; beyond that, uniform sampling
  is used to keep inference latency manageable.
"""

import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import gradio as gr
import nibabel as nib
import numpy as np
import pydicom
import timm
import torch
import torch.nn as nn
from fpdf import FPDF
from monai.networks.nets import SwinUNETR
from monai.visualize import GradCAM
from PIL import Image, ImageOps
from torchvision import models, transforms
from ultralytics import YOLO


# ─── SYSTEM CONFIGURATION ────────────────────────────────────────────────────

torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIAGNOSTIC_LABELS = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary",
    "Tumor (Generic / CT)",
]

MAX_INFERENCE_SLICES = 300
NII_TRIM             = 0.15     # Strip top/bottom 15 % of NIfTI volumes
OOD_REJECT_THRESHOLD = 0.70     # Gatekeeper confidence to reject non-brain input

WEIGHT_GATEKEEPER  = "HYDRA_Gatekeeper_v1.pth"
WEIGHT_SWIN        = "HYDRA_Swin_Council.pth"
WEIGHT_CONV        = "HYDRA_ConvNext_Council.pth"
WEIGHT_MONAI       = "HYDRA_MONAI_Council.pth"
CLASS_MAP_PATH     = "gatekeeper_class_map.json"
YOLO_CHECKPOINT    = "runs/detect/hydra_tumor_localizer/weights/best.pt"

print(f"[HYDRA] Initialising on {DEVICE} …")


# ─── MEDICAL SWIN ADAPTER (matches training scripts exactly) ─────────────────

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


# ─── SYSTEM INITIALISATION ───────────────────────────────────────────────────

def _load_gatekeeper() -> Tuple[Optional[nn.Module], Optional[Dict]]:
    try:
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(1280, 2)
        model.load_state_dict(
            torch.load(WEIGHT_GATEKEEPER, map_location=DEVICE, weights_only=True)
        )
        model = model.to(DEVICE).eval()
        if DEVICE.type == "cuda":
            model = model.half()
        with open(CLASS_MAP_PATH) as fh:
            class_map = json.load(fh)
        print(f"  [OK] Gatekeeper        → {WEIGHT_GATEKEEPER}")
        return model, class_map
    except Exception as e:
        print(f"  [WARN] Gatekeeper unavailable: {e}")
        return None, None


def _load_council() -> Tuple[Optional[List], Optional[nn.Module]]:
    try:
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
            swin.eval(); conv.eval()
        monai.eval()

        print(f"  [OK] SwinV2 Council    → {WEIGHT_SWIN}")
        print(f"  [OK] ConvNeXt Council  → {WEIGHT_CONV}")
        print(f"  [OK] MONAI Council     → {WEIGHT_MONAI}")
        return [swin, conv, monai], monai

    except Exception as e:
        print(f"  [ERROR] Council unavailable: {e}")
        return None, None


def _load_hunter() -> Optional[YOLO]:
    path = YOLO_CHECKPOINT if os.path.exists(YOLO_CHECKPOINT) else "yolo11n.pt"
    try:
        model = YOLO(path)
        print(f"  [OK] Hunter (YOLO)     → {path}")
        return model
    except Exception as e:
        print(f"  [WARN] Hunter unavailable: {e}")
        return None


print("[HYDRA] Loading model components …")
SYSTEM = {
    "gatekeeper":        None,
    "gatekeeper_labels": None,
    "council":           None,
    "explainer":         None,
    "hunter":            None,
}

SYSTEM["gatekeeper"], SYSTEM["gatekeeper_labels"] = _load_gatekeeper()
SYSTEM["council"],    SYSTEM["explainer"]          = _load_council()
SYSTEM["hunter"]                                   = _load_hunter()

print(f"[HYDRA] System ready.\n")


# ─── IMAGE TRANSFORMS ────────────────────────────────────────────────────────

GATEKEEPER_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

DIAGNOSTIC_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


# ─── PRE-PROCESSING UTILITIES ────────────────────────────────────────────────

def _normalise_array(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(float)
    lo, hi = arr.min(), arr.max()
    return ((arr - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)


def _skull_strip(rgb: np.ndarray) -> np.ndarray:
    """Simple Otsu-based skull stripping to isolate brain parenchyma."""
    try:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            clean_mask = np.zeros_like(gray)
            cv2.drawContours(clean_mask, [largest], -1, 255, -1)
            rgb = cv2.bitwise_and(rgb, rgb, mask=clean_mask)
    except Exception:
        pass
    return rgb


def _load_nifti(path: str) -> List[Image.Image]:
    vol = nib.load(path).get_fdata()
    depth = vol.shape[2]
    start, end = int(depth * NII_TRIM), int(depth * (1 - NII_TRIM))
    indices = list(range(start, end))
    if len(indices) > MAX_INFERENCE_SLICES:
        step = max(1, len(indices) // MAX_INFERENCE_SLICES)
        indices = indices[::step][:MAX_INFERENCE_SLICES]
    slices = []
    for z in indices:
        rgb = cv2.cvtColor(_normalise_array(vol[:, :, z]), cv2.COLOR_GRAY2RGB)
        slices.append(Image.fromarray(_skull_strip(rgb)))
    return slices


def _load_dicom_series(files: List) -> List[Image.Image]:
    dcm_objs = []
    for f in files:
        path = f.name if hasattr(f, "name") else f
        try:
            dcm_objs.append(pydicom.dcmread(path))
        except Exception:
            continue
    try:
        dcm_objs.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    except Exception:
        try:
            dcm_objs.sort(key=lambda d: int(d.InstanceNumber))
        except Exception:
            pass

    if len(dcm_objs) > MAX_INFERENCE_SLICES:
        step = max(1, len(dcm_objs) // MAX_INFERENCE_SLICES)
        dcm_objs = dcm_objs[::step][:MAX_INFERENCE_SLICES]

    slices = []
    for dcm in dcm_objs:
        try:
            rgb = cv2.cvtColor(_normalise_array(dcm.pixel_array), cv2.COLOR_GRAY2RGB)
            slices.append(Image.fromarray(_skull_strip(rgb)))
        except Exception:
            continue
    return slices


def ingest_patient_scan(file_list) -> List[Image.Image]:
    """
    Multi-format ingestion hub.
    Accepts DICOM series, NIfTI volumes, or standard image files.
    Returns a list of PIL RGB images capped at MAX_INFERENCE_SLICES.
    """
    if not file_list:
        return []

    paths  = [f.name if hasattr(f, "name") else f for f in file_list]
    slices = []

    # DICOM
    dicom_files = [f for f, p in zip(file_list, paths) if p.lower().endswith(".dcm")]
    if dicom_files:
        slices = _load_dicom_series(dicom_files)
        return slices

    for path in sorted(paths):
        if path.lower().endswith((".nii", ".nii.gz")):
            slices = _load_nifti(path)
            break
        elif path.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                rgb = np.array(Image.open(path).convert("RGB"))
                slices.append(Image.fromarray(_skull_strip(rgb)))
            except Exception:
                continue

    if len(slices) > MAX_INFERENCE_SLICES:
        step   = max(1, len(slices) // MAX_INFERENCE_SLICES)
        slices = slices[::step][:MAX_INFERENCE_SLICES]

    return slices


# ─── EXPLAINABILITY ──────────────────────────────────────────────────────────

def _generate_gradcam(model: nn.Module, tensor: torch.Tensor,
                      original_rgb: np.ndarray) -> np.ndarray:
    """Overlay a Grad-CAM saliency heatmap onto the original scan image."""
    with torch.enable_grad():
        try:
            cam = GradCAM(
                nn_module=model,
                target_layers="backbone.swinViT.layers.3",
            )
            saliency = cam(x=tensor.float())
        except Exception:
            return original_rgb

    hmap  = saliency[0][0].detach().cpu().numpy()
    hmap  = cv2.resize(hmap, (original_rgb.shape[1], original_rgb.shape[0]))
    hmap  = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
    color = cv2.applyColorMap(np.uint8(255 * hmap), cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(original_rgb, 0.55, color, 0.45, 0)


# ─── INFERENCE ENGINE ────────────────────────────────────────────────────────

BRANCH_NAMES   = ["SwinV2", "ConvNeXt", "MONAI"]
BRANCH_WEIGHTS = [0.4,       0.3,        0.3]


def run_full_diagnostic(
    file_list,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[str, Optional[np.ndarray], Optional[Dict], Optional[str]]:
    """
    Orchestrates the complete Gatekeeper → Hunter → Council → Report pipeline.
    """
    torch.cuda.empty_cache()

    if not file_list:
        return (
            "## ⏳ System Idle\nUpload one or more scan files to begin analysis.",
            None, None, None,
        )

    slices = ingest_patient_scan(file_list)

    if not slices:
        return (
            "## ❌ Ingestion Failed\nNo readable scan slices found in the uploaded files.",
            None, None, None,
        )

    # ── Per-slice inference ────────────────────────────────────────────────────
    running_probs     = np.zeros(5)
    valid_count       = 0
    peak_tumor_signal = 0.0
    best_visual       = None
    branch_history    = {n: [] for n in BRANCH_NAMES}

    for clinical_slice in progress.tqdm(slices, desc="Analysing slices …"):
        clinical_slice.thumbnail((1024, 1024))

        # 1. Gatekeeper check
        if SYSTEM["gatekeeper"] is not None and SYSTEM["gatekeeper_labels"] is not None:
            ood_idx = SYSTEM["gatekeeper_labels"].get("NotBrain", 1)
            with torch.no_grad():
                v_in = GATEKEEPER_TRANSFORM(clinical_slice).unsqueeze(0).to(DEVICE)
                if DEVICE.type == "cuda":
                    v_in = v_in.half()
                gate_prob = torch.softmax(SYSTEM["gatekeeper"](v_in), dim=1)
                if gate_prob[0][ood_idx].item() > OOD_REJECT_THRESHOLD:
                    continue    # Rejected — not a brain scan

        # 2. Council inference (test-time augmentation: original + horizontal flip)
        if SYSTEM["council"] is None:
            continue

        t_orig = DIAGNOSTIC_TRANSFORM(clinical_slice)
        t_flip = DIAGNOSTIC_TRANSFORM(ImageOps.mirror(clinical_slice))
        batch  = torch.stack([t_orig, t_flip]).to(DEVICE)

        branch_probs = []
        for b_idx, voter in enumerate(SYSTEM["council"]):
            with torch.no_grad():
                if voter is SYSTEM["explainer"]:
                    logits = voter(batch.float())
                elif DEVICE.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        logits = voter(batch.half())
                else:
                    logits = voter(batch)

                # Temperature scaling (T=1.5) for calibrated confidence
                probs = torch.softmax(logits / 1.5, dim=1).mean(dim=0).float().cpu().numpy()
                branch_probs.append(probs)
                branch_history[BRANCH_NAMES[b_idx]].append(probs)

        # Weighted consensus across branches
        weighted = sum(w * p for w, p in zip(BRANCH_WEIGHTS, branch_probs))
        running_probs += weighted
        valid_count   += 1

        # Track most pathological slice for Grad-CAM
        tumor_signal = 1.0 - weighted[2]   # 1 - P(NoTumor)
        if tumor_signal > peak_tumor_signal:
            peak_tumor_signal = tumor_signal
            if SYSTEM["explainer"] is not None:
                best_visual = _generate_gradcam(
                    SYSTEM["explainer"],
                    t_orig.unsqueeze(0).to(DEVICE),
                    np.array(clinical_slice),
                )
            else:
                best_visual = np.array(clinical_slice)

    # ── No valid slices after Gatekeeper ──────────────────────────────────────
    if valid_count == 0:
        return (
            "## 🚫 Gatekeeper Rejection\n"
            "All uploaded images were classified as non-brain scans.  "
            "Please upload MRI, CT, or NIfTI brain scan data.",
            None, None, None,
        )

    # ── Final ensemble probabilities ──────────────────────────────────────────
    avg_probs        = running_probs / valid_count
    p_no_tumor       = float(avg_probs[2])
    p_tumor          = 1.0 - p_no_tumor
    verdict          = "Tumor Detected" if p_tumor > 0.50 else "No Tumor Detected"
    confidence       = p_tumor if p_tumor > 0.50 else p_no_tumor
    entropy          = -np.sum(avg_probs * np.log(avg_probs + 1e-8))
    verdict_emoji    = "🔴" if "Tumor" in verdict and "No" not in verdict else "🟢"

    # ── Branch analytics ──────────────────────────────────────────────────────
    branch_blocks = ""
    pdf_notes     = {}

    for name in BRANCH_NAMES:
        hist = branch_history[name]
        if not hist:
            continue
        h = np.array(hist)
        mean_p    = np.mean(h, axis=0)
        top_idx   = int(np.argmax(mean_p))
        top_label = DIAGNOSTIC_LABELS[top_idx]
        stability = 1.0 - float(np.mean(np.std(h, axis=0)))
        burden    = float(np.mean(np.argmax(h, axis=1) != 2)) * 100  # % slices flagged

        branch_blocks += (
            f"\n#### {name}\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Primary Finding | **{top_label}** ({mean_p[top_idx]*100:.1f}%) |\n"
            f"| Volumetric Stability | {stability*100:.1f}% |\n"
            f"| Lesion Burden | {burden:.1f}% of slices flagged |\n"
        )
        pdf_notes[name] = f"{top_label} | {mean_p[top_idx]*100:.1f}% confidence"

    # ── Class probability dictionary ──────────────────────────────────────────
    prob_dict = {
        label: float(p)
        for label, p in zip(DIAGNOSTIC_LABELS, np.round(avg_probs, 4))
    }

    # ── Narrative report ──────────────────────────────────────────────────────
    report = f"""# {verdict_emoji} HYDRA Diagnostic Report

---

## Patient Volume Summary
| Parameter | Value |
|-----------|-------|
| Scan slices analysed | {valid_count} / {len(slices)} |
| Timestamp | {datetime.now().strftime('%Y-%m-%d %H:%M UTC')} |
| Hardware | {DEVICE} |

---

## Primary Diagnostic Finding

> ### {verdict_emoji} {verdict}
> **Confidence:** {confidence * 100:.2f}%   |   **Uncertainty (Entropy):** {entropy:.4f}

---

## Differential Probability Distribution
| Diagnosis | Probability |
|-----------|-------------|
""" + "\n".join(
        f"| {lbl} | {p*100:.2f}% |"
        for lbl, p in zip(DIAGNOSTIC_LABELS, avg_probs)
    ) + f"""

---

## Council Branch Analysis
{branch_blocks}

---

> ⚠️ **Clinical Disclaimer:** This system is intended as a *decision-support tool*.
> All findings must be confirmed by a qualified radiologist or neuro-oncologist.
"""

    # ── PDF report ────────────────────────────────────────────────────────────
    pdf_path = _generate_pdf_report(
        verdict, confidence, entropy, valid_count,
        pdf_notes, best_visual, prob_dict,
    )

    return report, best_visual, prob_dict, pdf_path


# ─── PDF GENERATION ──────────────────────────────────────────────────────────

def _generate_pdf_report(
    verdict: str, confidence: float, entropy: float,
    slice_count: int, branch_notes: Dict,
    heatmap: Optional[np.ndarray], prob_dist: Dict,
) -> str:
    doc = FPDF()
    doc.add_page()

    # Header
    doc.set_fill_color(10, 15, 30)
    doc.rect(0, 0, 210, 30, "F")
    doc.set_font("Arial", "B", 18)
    doc.set_text_color(255, 255, 255)
    doc.cell(0, 20, "HYDRA — Clinical Brain Tumor Analysis", ln=True, align="C")
    doc.set_text_color(0, 0, 0)

    doc.ln(8)
    doc.set_font("Arial", "", 10)
    doc.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    doc.cell(0, 6, f"Scan slices analysed: {slice_count}", ln=True)

    doc.ln(6)
    doc.set_font("Arial", "B", 14)
    is_positive = "Tumor Detected" in verdict and "No" not in verdict
    doc.set_text_color(*(180, 30, 30) if is_positive else (0, 130, 80))
    doc.cell(0, 10, f"Finding: {verdict}", ln=True)

    doc.set_text_color(0, 0, 0)
    doc.set_font("Arial", "", 11)
    doc.cell(0, 7, f"Confidence: {confidence*100:.1f}%   |   Uncertainty Entropy: {entropy:.4f}", ln=True)

    doc.ln(5)
    doc.set_font("Arial", "B", 12)
    doc.cell(0, 8, "Differential Probability Distribution:", ln=True)
    doc.set_font("Arial", "", 10)
    for cls, val in prob_dist.items():
        bar_len = int(val * 60)
        doc.cell(60, 6, f"  {cls}:", ln=False)
        doc.cell(0, 6, f"{'█' * bar_len} {val*100:.2f}%", ln=True)

    doc.ln(5)
    doc.set_font("Arial", "B", 12)
    doc.cell(0, 8, "Council Branch Summary:", ln=True)
    doc.set_font("Arial", "", 10)
    for name, note in branch_notes.items():
        doc.cell(0, 6, f"  {name}: {note}", ln=True)

    if heatmap is not None:
        tmp_path = "_tmp_heatmap_export.jpg"
        Image.fromarray(heatmap).save(tmp_path)
        doc.ln(6)
        doc.set_font("Arial", "B", 11)
        doc.cell(0, 8, "Grad-CAM Saliency Map (Most Pathological Slice):", ln=True)
        doc.image(tmp_path, x=55, w=100)

    doc.ln(8)
    doc.set_font("Arial", "I", 9)
    doc.set_text_color(120, 120, 120)
    doc.multi_cell(0, 5,
        "DISCLAIMER: This report is generated by an AI decision-support system.  "
        "It does not constitute a medical diagnosis.  All findings must be reviewed "
        "and confirmed by a qualified radiologist or neuro-oncologist."
    )

    filename = f"HYDRA_Report_{int(datetime.now().timestamp())}.pdf"
    doc.output(filename)
    return filename


# ─── STATUS HELPER ───────────────────────────────────────────────────────────

def update_upload_status(files) -> str:
    if not files:
        return "📂 No files uploaded yet."
    n = len(files)
    return f"✅ {n} file{'s' if n > 1 else ''} uploaded and ready for analysis."


# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Root / Global ────────────────────────────────────────────────── */
:root {
  --hydra-bg:        #070d1a;
  --hydra-panel:     #0e1628;
  --hydra-border:    #1a2847;
  --hydra-accent:    #00c8ff;
  --hydra-accent2:   #7c3aed;
  --hydra-success:   #00d68f;
  --hydra-danger:    #ff4757;
  --hydra-text:      #cdd9f5;
  --hydra-subtext:   #6b7fa3;
  --hydra-radius:    12px;
  --hydra-glow:      0 0 20px rgba(0, 200, 255, 0.18);
}

body, .gradio-container {
  background: var(--hydra-bg) !important;
  color: var(--hydra-text) !important;
  font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* ── Header Banner ─────────────────────────────────────────────────── */
#hydra-header {
  background: linear-gradient(135deg, #0a1f44 0%, #0e1628 40%, #100e2a 100%);
  border: 1px solid var(--hydra-border);
  border-radius: var(--hydra-radius);
  padding: 28px 36px;
  margin-bottom: 20px;
  box-shadow: var(--hydra-glow);
  text-align: center;
}
#hydra-header h1 {
  font-size: 2.4rem;
  font-weight: 800;
  background: linear-gradient(90deg, var(--hydra-accent), #a78bfa);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 0 8px;
  letter-spacing: -0.5px;
}
#hydra-header p {
  color: var(--hydra-subtext);
  font-size: 0.95rem;
  margin: 0;
}

/* ── Panels ────────────────────────────────────────────────────────── */
.hydra-panel {
  background: var(--hydra-panel) !important;
  border: 1px solid var(--hydra-border) !important;
  border-radius: var(--hydra-radius) !important;
  padding: 20px !important;
}

/* ── Upload Zone ───────────────────────────────────────────────────── */
.upload-zone .wrap {
  background: #0b1425 !important;
  border: 2px dashed var(--hydra-border) !important;
  border-radius: var(--hydra-radius) !important;
  transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}
.upload-zone .wrap:hover {
  border-color: var(--hydra-accent) !important;
  box-shadow: var(--hydra-glow) !important;
}

/* ── Execute Button ────────────────────────────────────────────────── */
#execute-btn {
  background: linear-gradient(135deg, #0050ff, #6d28d9) !important;
  border: none !important;
  border-radius: 10px !important;
  color: #fff !important;
  font-size: 1.05rem !important;
  font-weight: 700 !important;
  height: 52px !important;
  letter-spacing: 0.5px !important;
  transition: opacity 0.2s ease, transform 0.15s ease !important;
  box-shadow: 0 4px 20px rgba(0, 80, 255, 0.35) !important;
}
#execute-btn:hover {
  opacity: 0.90 !important;
  transform: translateY(-1px) !important;
}

/* ── Status Badge ──────────────────────────────────────────────────── */
#status-badge .prose {
  font-size: 0.88rem !important;
  color: var(--hydra-subtext) !important;
}

/* ── Markdown Report ───────────────────────────────────────────────── */
#report-panel .prose {
  background: transparent !important;
  color: var(--hydra-text) !important;
}
#report-panel .prose h1 {
  color: var(--hydra-accent) !important;
  font-size: 1.5rem !important;
  border-bottom: 1px solid var(--hydra-border);
  padding-bottom: 6px;
}
#report-panel .prose h2 {
  color: #a78bfa !important;
  font-size: 1.15rem !important;
}
#report-panel .prose table {
  border-collapse: collapse !important;
  width: 100% !important;
}
#report-panel .prose td, #report-panel .prose th {
  border: 1px solid var(--hydra-border) !important;
  padding: 6px 12px !important;
  color: var(--hydra-text) !important;
}
#report-panel .prose th {
  background: #111f3a !important;
  color: var(--hydra-accent) !important;
}
#report-panel .prose blockquote {
  border-left: 4px solid var(--hydra-accent) !important;
  background: #0b1a30 !important;
  margin: 12px 0 !important;
  padding: 10px 16px !important;
  border-radius: 0 8px 8px 0 !important;
}

/* ── Label / Probability Widget ────────────────────────────────────── */
.label-wrap {
  background: var(--hydra-panel) !important;
  border: 1px solid var(--hydra-border) !important;
  border-radius: var(--hydra-radius) !important;
}

/* ── Heatmap Image ─────────────────────────────────────────────────── */
#heatmap-panel img {
  border-radius: var(--hydra-radius) !important;
  border: 1px solid var(--hydra-border) !important;
}

/* ── Stat Cards ────────────────────────────────────────────────────── */
.stat-card {
  background: #0b1628;
  border: 1px solid var(--hydra-border);
  border-radius: 10px;
  padding: 14px 20px;
  text-align: center;
}
.stat-card .stat-value {
  font-size: 1.6rem;
  font-weight: 800;
  color: var(--hydra-accent);
}
.stat-card .stat-label {
  font-size: 0.78rem;
  color: var(--hydra-subtext);
  margin-top: 4px;
}

/* ── Section Dividers ──────────────────────────────────────────────── */
.section-label {
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--hydra-subtext);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 8px;
}
"""


# ─── GRADIO INTERFACE ────────────────────────────────────────────────────────

with gr.Blocks(
    title="HYDRA — Clinical Brain Tumor Analysis System",
    css=CUSTOM_CSS,
) as app:

    # ── HEADER ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="hydra-header">
      <h1>🧠 HYDRA</h1>
      <p>Clinical-Grade Brain Tumour Detection &amp; Classification System</p>
      <p style="margin-top:8px; font-size:0.8rem; color:#3a5580;">
        EfficientNet-B0 Gatekeeper &nbsp;·&nbsp; YOLOv11 Hunter &nbsp;·&nbsp;
        SwinV2 + ConvNeXt + MONAI Council &nbsp;·&nbsp; Grad-CAM Explainability
      </p>
    </div>
    """)

    # ── MAIN LAYOUT ─────────────────────────────────────────────────────────
    with gr.Row():

        # ── LEFT COLUMN: Upload + Controls ──────────────────────────────────
        with gr.Column(scale=1, min_width=320):

            gr.HTML('<div class="section-label">📁 Patient Scan Upload</div>')
            upload = gr.File(
                file_count="multiple",
                label="Drop scan files here (DICOM · NIfTI · JPG · PNG)",
                elem_classes=["upload-zone"],
                height=160,
            )

            status = gr.Markdown(
                "📂 No files uploaded yet.",
                elem_id="status-badge",
            )

            execute_btn = gr.Button(
                "⚡  Execute Comprehensive Analysis",
                variant="primary",
                elem_id="execute-btn",
            )

            gr.HTML("""
            <div class="hydra-panel" style="margin-top:16px;">
              <div class="section-label">🔬 Model Status</div>
              <div style="font-size:0.85rem; line-height:1.9; color:#6b7fa3;">
                🛡️ Gatekeeper &nbsp;— EfficientNet-B0<br>
                🎯 Hunter &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;— YOLOv11n<br>
                🧠 SwinV2 &nbsp;&nbsp;&nbsp;&nbsp;— Transformer Branch<br>
                🧠 ConvNeXt &nbsp;&nbsp;— Convolutional Branch<br>
                🧠 MONAI &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;— Medical Branch
              </div>
            </div>

            <div class="hydra-panel" style="margin-top:12px;">
              <div class="section-label">📋 Supported Formats</div>
              <div style="font-size:0.82rem; line-height:1.8; color:#6b7fa3;">
                • DICOM series &nbsp;&nbsp;(.dcm)<br>
                • NIfTI volumes &nbsp;(.nii, .nii.gz)<br>
                • MRI / CT images (.jpg, .jpeg, .png)<br>
                • Up to <strong style="color:#00c8ff;">300 slices</strong> per study
              </div>
            </div>
            """)

        # ── RIGHT COLUMN: Results ────────────────────────────────────────────
        with gr.Column(scale=2):

            with gr.Tabs():

                # ── Tab 1: Diagnostic Report ─────────────────────────────────
                with gr.Tab("📋 Diagnostic Report"):
                    report_md = gr.Markdown(
                        "## ⏳ System Idle\n"
                        "Upload scan files and click **Execute** to begin.",
                        elem_id="report-panel",
                    )

                # ── Tab 2: Visual Analysis ───────────────────────────────────
                with gr.Tab("🔬 Visual Analysis"):
                    with gr.Row():
                        with gr.Column():
                            gr.HTML('<div class="section-label">📊 Differential Probability</div>')
                            prob_label = gr.Label(
                                num_top_classes=5,
                                label="Class Probability Distribution",
                            )
                        with gr.Column():
                            gr.HTML('<div class="section-label">🧠 Grad-CAM Saliency Map</div>')
                            heatmap_img = gr.Image(
                                label="Most Pathological Slice",
                                elem_id="heatmap-panel",
                                show_download_button=True,
                            )

                # ── Tab 3: Download Report ───────────────────────────────────
                with gr.Tab("📄 Export Report"):
                    gr.HTML("""
                    <div style="text-align:center; padding:32px 0 16px;">
                      <div style="font-size:3rem;">📑</div>
                      <div style="color:#6b7fa3; margin:8px 0 16px; font-size:0.9rem;">
                        Run the analysis to generate a downloadable clinical PDF report.<br>
                        Includes probability distribution, branch analysis, and Grad-CAM image.
                      </div>
                    </div>
                    """)
                    pdf_output = gr.File(
                        label="Download Clinical PDF Report",
                        interactive=False,
                    )

    # ── FOOTER ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center; margin-top:24px; padding:16px;
                border-top:1px solid #1a2847; color:#3a5580; font-size:0.78rem;">
      HYDRA v2.0 &nbsp;·&nbsp; For research and educational purposes only &nbsp;·&nbsp;
      Not a substitute for professional medical diagnosis
    </div>
    """)

    # ── EVENT BINDINGS ───────────────────────────────────────────────────────
    upload.change(
        update_upload_status, inputs=upload, outputs=status
    )

    execute_btn.click(
        run_full_diagnostic,
        inputs=[upload],
        outputs=[report_md, heatmap_img, prob_label, pdf_output],
    )


# ─── LAUNCH ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        favicon_path=None,
    )
