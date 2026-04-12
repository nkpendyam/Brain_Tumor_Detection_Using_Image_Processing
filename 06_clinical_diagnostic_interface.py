"""
06_clinical_diagnostic_interface.py — HYDRA Clinical Dashboard
==============================================================
Seminar-ready Gradio dashboard for multi-format brain scan analysis.

Supports: MRI, CT, DICOM series, NIfTI volumes, pre-sliced image folders.
Handles 200-300 slices per patient with uniform downsampling when needed.

Fixes Applied
-------------
• torch.cuda.amp.autocast → torch.amp.autocast('cuda')
• All deprecated AMP usage removed
• Gradio File type annotation corrected
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import nibabel as nib
import numpy as np
import pydicom
import timm
import torch
import torch.nn as nn
from fpdf import FPDF
from monai.visualize.class_activation_maps import GradCAM
from PIL import Image, ImageOps
from torchvision import models, transforms  # type: ignore[import-untyped]
from ultralytics import YOLO  # type: ignore[import-untyped]

from hydra_core import (
    BRANCH_NAMES,
    BRANCH_WEIGHTS,
    DIAGNOSTIC_LABELS,
    MedicalSwinAdapter,
    load_monai_adapter_checkpoint,
)


torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"

MAX_INFERENCE_SLICES = 300
NIFTI_TRIM_FRACTION  = 0.15
OOD_REJECT_THRESHOLD = 0.70

WEIGHT_GATEKEEPER = Path("HYDRA_Gatekeeper_v1.pth")
WEIGHT_SWIN       = Path("HYDRA_Swin_Council.pth")
WEIGHT_CONV       = Path("HYDRA_ConvNext_Council.pth")
WEIGHT_MONAI      = Path("HYDRA_MONAI_Council.pth")
CLASS_MAP_PATH    = Path("gatekeeper_class_map.json")
YOLO_CHECKPOINT   = Path("runs/detect/hydra_tumor_localizer/weights/best.pt")

print(f"[HYDRA] Initialising dashboard on {DEVICE} …")


# ─── MODEL LOADING ────────────────────────────────────────────────────────────

def _load_gatekeeper() -> Tuple[Optional[nn.Module], Optional[Dict[str, int]]]:
    try:
        model = models.efficientnet_b0(weights=None)
        in_features: int = int(model.classifier[1].in_features)  # type: ignore[union-attr]
        model.classifier[1] = nn.Linear(in_features, 2)
        model.load_state_dict(
            torch.load(WEIGHT_GATEKEEPER, map_location=DEVICE, weights_only=True)
        )
        model = model.to(DEVICE).eval()
        if AMP_ENABLED:
            model = model.half()
        class_map: Dict[str, int] = json.loads(CLASS_MAP_PATH.read_text())
        print(f"  [OK] Gatekeeper  → {WEIGHT_GATEKEEPER}")
        return model, class_map
    except Exception as e:
        print(f"  [WARN] Gatekeeper unavailable: {e}")
        return None, None


def _load_council() -> Tuple[Optional[List[nn.Module]], Optional[nn.Module]]:
    try:
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
            swin = swin.half()
            conv = conv.half()

        swin.eval(); conv.eval(); monai.eval()

        print(f"  [OK] SwinV2      → {WEIGHT_SWIN}")
        print(f"  [OK] ConvNeXt    → {WEIGHT_CONV}")
        print(f"  [OK] MONAI       → {WEIGHT_MONAI}")
        return [swin, conv, monai], monai
    except Exception as e:
        print(f"  [ERROR] Council unavailable: {e}")
        return None, None


def _load_hunter() -> Optional[Any]:
    checkpoint = YOLO_CHECKPOINT if YOLO_CHECKPOINT.exists() else None
    if checkpoint is None:
        print("  [WARN] Hunter checkpoint not found — localization disabled.")
        return None
    try:
        model = YOLO(str(checkpoint))
        print(f"  [OK] Hunter      → {checkpoint}")
        return model
    except Exception as e:
        print(f"  [WARN] Hunter unavailable: {e}")
        return None


SYSTEM: Dict[str, Any] = {
    "gatekeeper": None, "gatekeeper_labels": None,
    "council": None,    "explainer": None,
    "hunter": None,
}

print("[HYDRA] Loading model components …")
SYSTEM["gatekeeper"], SYSTEM["gatekeeper_labels"] = _load_gatekeeper()
SYSTEM["council"],    SYSTEM["explainer"]          = _load_council()
SYSTEM["hunter"]                                   = _load_hunter()
print("[HYDRA] Dashboard ready.\n")


# ─── TRANSFORMS ───────────────────────────────────────────────────────────────

GATE_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

DIAG_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


# ─── IMAGE UTILITIES ──────────────────────────────────────────────────────────

def _normalise(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    return np.uint8(np.clip(((arr - lo) / (hi - lo + 1e-8)) * 255, 0, 255))


def _skull_strip(rgb: np.ndarray) -> np.ndarray:
    """Remove background noise via Otsu thresholding (best-effort)."""
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


def _uniform_indices(length: int, cap: int) -> np.ndarray:
    if length <= cap:
        return np.arange(length)
    return np.linspace(0, length - 1, num=cap, dtype=int)


# ─── INGEST ───────────────────────────────────────────────────────────────────

def _load_nifti(path: str) -> List[Image.Image]:
    volume = nib.load(path).get_fdata()
    depth  = int(volume.shape[2])
    start  = int(depth * NIFTI_TRIM_FRACTION)
    end    = int(depth * (1 - NIFTI_TRIM_FRACTION))
    indices = _uniform_indices(end - start, MAX_INFERENCE_SLICES) + start

    out: List[Image.Image] = []
    for idx in indices:
        raw = _normalise(volume[:, :, int(idx)])
        rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        out.append(Image.fromarray(_skull_strip(rgb)))
    return out


def _load_dicom_series(files: List[str]) -> List[Image.Image]:
    records: list[tuple[float, Any]] = []
    for fp in files:
        try:
            ds = pydicom.dcmread(fp, force=True)
            if hasattr(ds, "ImagePositionPatient"):
                key = float(ds.ImagePositionPatient[2])
            elif hasattr(ds, "InstanceNumber"):
                key = float(ds.InstanceNumber)
            else:
                key = float(len(records))
            records.append((key, ds))
        except Exception:
            continue

    records.sort(key=lambda x: x[0])
    dcms = [r for _, r in records]
    indices = _uniform_indices(len(dcms), MAX_INFERENCE_SLICES)
    dcms = [dcms[i] for i in indices]

    out: List[Image.Image] = []
    for ds in dcms:
        try:
            raw = _normalise(ds.pixel_array.astype(np.float32))
            rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB) if raw.ndim == 2 else raw
            out.append(Image.fromarray(_skull_strip(rgb)))
        except Exception:
            continue
    return out


def ingest_patient_scan(file_list: list) -> List[Image.Image]:
    """
    Parse uploaded files into a list of PIL RGB slices.
    Handles: DICOM series, NIfTI, and pre-sliced JPG/PNG folders.
    """
    if not file_list:
        return []

    paths = [f.name if hasattr(f, "name") else str(f) for f in file_list]
    dcm_paths = [p for p in paths if p.lower().endswith(".dcm")]
    if dcm_paths:
        return _load_dicom_series(dcm_paths)

    for p in sorted(paths):
        if p.lower().endswith((".nii.gz", ".nii")):
            return _load_nifti(p)

    slices: List[Image.Image] = []
    for p in sorted(paths):
        if p.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                rgb = np.array(Image.open(p).convert("RGB"))
                slices.append(Image.fromarray(_skull_strip(rgb)))
            except Exception:
                continue

    indices = _uniform_indices(len(slices), MAX_INFERENCE_SLICES)
    return [slices[i] for i in indices]


# ─── GRAD-CAM ─────────────────────────────────────────────────────────────────

def _generate_gradcam(
    model: nn.Module,
    tensor: torch.Tensor,
    original_rgb: np.ndarray,
) -> np.ndarray:
    with torch.enable_grad():
        try:
            cam = GradCAM(nn_module=model, target_layers="backbone.swinViT.layers.3")
            saliency = cam(x=tensor.float())
        except Exception:
            return original_rgb

    heatmap = saliency[0][0].detach().cpu().numpy()
    heatmap  = cv2.resize(heatmap, (original_rgb.shape[1], original_rgb.shape[0]))
    heatmap  = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    colored  = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(original_rgb, 0.58, colored, 0.42, 0)


# ─── REPORT FORMATTING ────────────────────────────────────────────────────────

def _format_report(
    valid: int,
    total: int,
    verdict: str,
    confidence: float,
    entropy: float,
    probs: np.ndarray,
    branch_history: Dict[str, List[np.ndarray]],
) -> str:
    prob_rows = "\n".join(
        f"| {lbl} | {v * 100:.2f}% |"
        for lbl, v in zip(DIAGNOSTIC_LABELS, probs)
    )

    branch_rows: List[str] = []
    for name in BRANCH_NAMES:
        hist = branch_history[name]
        if not hist:
            continue
        vals = np.array(hist)
        means = np.mean(vals, axis=0)
        stability = 1.0 - float(np.mean(np.std(vals, axis=0)))
        top = int(np.argmax(means))
        burden = float(np.mean(np.argmax(vals, axis=1) != 2)) * 100
        branch_rows.append(
            f"| {name} | {DIAGNOSTIC_LABELS[top]} | "
            f"{means[top]*100:.1f}% | {stability*100:.1f}% | {burden:.1f}% |"
        )

    return f"""# HYDRA Clinical Diagnostic Report

## Executive Summary
| Parameter | Value |
|-----------|-------|
| Final finding | **{verdict}** |
| Confidence | **{confidence * 100:.2f}%** |
| Uncertainty entropy | {entropy:.4f} |
| Slices analysed | {valid} / {total} |
| Timestamp | {datetime.now().strftime('%Y-%m-%d %H:%M UTC')} |
| Runtime device | {DEVICE} |

## Differential Probability Distribution
| Diagnosis | Probability |
|-----------|-------------|
{prob_rows}

## Council Branch Stability
| Branch | Primary signal | Confidence | Stability | Slice burden |
|--------|----------------|------------|-----------|--------------|
{chr(10).join(branch_rows) or '| No branch data | - | - | - | - |'}

## Interpretation
Full HYDRA pipeline: gatekeeper screening → multi-branch classification → study-level aggregation.

## ⚠ Clinical Use Notice
**Research and demonstration only.** Clinical interpretation must be validated by a qualified radiologist.
"""


def _generate_pdf_report(
    verdict: str,
    confidence: float,
    entropy: float,
    slice_count: int,
    branch_notes: Dict[str, str],
    heatmap: Optional[np.ndarray],
    prob_dict: Dict[str, float],
) -> str:
    doc = FPDF()
    doc.add_page()
    doc.set_fill_color(16, 39, 61)
    doc.rect(0, 0, 210, 28, "F")
    doc.set_text_color(255, 255, 255)
    doc.set_font("Arial", "B", 18)
    doc.cell(0, 18, "HYDRA Clinical Analysis Report", ln=True, align="C")
    doc.set_text_color(0, 0, 0)
    doc.ln(6)
    doc.set_font("Arial", "", 10)
    doc.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}", ln=True)
    doc.cell(0, 6, f"Slices analysed: {slice_count}", ln=True)
    doc.ln(4)
    doc.set_font("Arial", "B", 13)
    doc.cell(0, 8, f"Finding: {verdict}", ln=True)
    doc.set_font("Arial", "", 11)
    doc.cell(0, 7, f"Confidence: {confidence*100:.1f}%   Entropy: {entropy:.4f}", ln=True)
    doc.ln(5)
    doc.set_font("Arial", "B", 12)
    doc.cell(0, 8, "Probability Distribution", ln=True)
    doc.set_font("Arial", "", 10)
    for lbl, val in prob_dict.items():
        doc.cell(90, 6, lbl, ln=False)
        doc.cell(0, 6, f"{val * 100:.2f}%", ln=True)
    doc.ln(4)
    doc.set_font("Arial", "B", 12)
    doc.cell(0, 8, "Branch Summary", ln=True)
    doc.set_font("Arial", "", 10)
    for name, note in branch_notes.items():
        doc.cell(0, 6, f"{name}: {note}", ln=True)

    tmp_img = None
    if heatmap is not None:
        tmp_img = "_hydra_heatmap_tmp.jpg"
        Image.fromarray(heatmap).save(tmp_img)
        doc.ln(5)
        doc.set_font("Arial", "B", 11)
        doc.cell(0, 8, "Representative Saliency Map", ln=True)
        doc.image(tmp_img, x=45, w=120)

    doc.ln(8)
    doc.set_font("Arial", "I", 9)
    doc.set_text_color(100, 100, 100)
    doc.multi_cell(0, 5, "Research and educational use only. Not a standalone medical diagnosis.")

    out_path = f"HYDRA_Report_{int(datetime.now().timestamp())}.pdf"
    doc.output(out_path)

    if tmp_img and os.path.exists(tmp_img):
        os.remove(tmp_img)

    return out_path


# ─── MAIN DIAGNOSTIC PIPELINE ─────────────────────────────────────────────────

def run_full_diagnostic(
    file_list: Any,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[str, Optional[np.ndarray], Optional[Dict[str, float]], Optional[str]]:
    """
    Full HYDRA pipeline:
      1. Ingest multi-format patient study
      2. Gatekeeper OOD screening
      3. Council weighted-vote inference (per slice)
      4. Patient-level aggregation
      5. Report + Grad-CAM + PDF generation
    """
    if AMP_ENABLED:
        torch.cuda.empty_cache()

    if not file_list:
        return "# System Idle\nUpload a study to begin.", None, None, None

    slices = ingest_patient_scan(file_list)
    if not slices:
        return "# Ingestion Failed\nNo readable slices detected.", None, None, None

    if SYSTEM["council"] is None:
        return "# Models Unavailable\nCouncil checkpoints not loaded.", None, None, None

    running_probs = np.zeros(len(DIAGNOSTIC_LABELS), dtype=np.float64)
    valid_count   = 0
    best_visual: Optional[np.ndarray] = None
    peak_tumor    = 0.0
    branch_history: Dict[str, List[np.ndarray]] = {n: [] for n in BRANCH_NAMES}

    for sl in progress.tqdm(slices, desc="Analysing study"):
        sl.thumbnail((1024, 1024))

        # Gatekeeper screening
        if SYSTEM["gatekeeper"] is not None and SYSTEM["gatekeeper_labels"] is not None:
            ood_idx = SYSTEM["gatekeeper_labels"].get("NotBrain", 1)
            gate_t  = GATE_TRANSFORM(sl).unsqueeze(0).to(DEVICE)
            if AMP_ENABLED:
                gate_t = gate_t.half()
            gate_p = torch.softmax(SYSTEM["gatekeeper"](gate_t), dim=1)
            if gate_p[0][ood_idx].item() > OOD_REJECT_THRESHOLD:
                continue

        orig_t = DIAG_TRANSFORM(sl)
        mir_t  = DIAG_TRANSFORM(ImageOps.mirror(sl))
        batch  = torch.stack([orig_t, mir_t]).to(DEVICE)

        branch_probs: List[np.ndarray] = []
        for bi, voter in enumerate(SYSTEM["council"]):
            with torch.no_grad():
                is_monai = voter is SYSTEM["explainer"]
                if is_monai:
                    logits = voter(batch.float())
                else:
                    with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                        logits = voter(batch.half() if AMP_ENABLED else batch)

            probs = (
                torch.softmax(logits / 1.5, dim=1)
                .mean(dim=0).float().cpu().numpy()
            )
            branch_probs.append(probs)
            branch_history[BRANCH_NAMES[bi]].append(probs)

        weighted = np.sum(
            [w * p for w, p in zip(BRANCH_WEIGHTS, branch_probs)], axis=0
        )
        running_probs += weighted
        valid_count   += 1

        tumor_signal = float(1.0 - weighted[2])
        if tumor_signal > peak_tumor:
            peak_tumor = tumor_signal
            if SYSTEM["explainer"] is not None:
                best_visual = _generate_gradcam(
                    SYSTEM["explainer"],
                    orig_t.unsqueeze(0).to(DEVICE),
                    np.array(sl),
                )
            else:
                best_visual = np.array(sl)

    if valid_count == 0:
        return (
            "# Gatekeeper Rejection\n"
            "All slices were rejected as non-brain inputs.\n"
            "Please upload MRI, CT, DICOM, or NIfTI brain studies.",
            None, None, None,
        )

    avg_probs      = running_probs / valid_count
    no_tumor_p     = float(avg_probs[2])
    tumor_p        = 1.0 - no_tumor_p
    verdict        = "Tumor Detected" if tumor_p > 0.5 else "No Tumor Detected"
    confidence     = tumor_p if tumor_p > 0.5 else no_tumor_p
    entropy        = float(-np.sum(avg_probs * np.log(avg_probs + 1e-8)))
    prob_dict      = {lbl: float(round(v, 4)) for lbl, v in zip(DIAGNOSTIC_LABELS, avg_probs)}
    branch_notes   = {}
    for name in BRANCH_NAMES:
        hist = branch_history[name]
        if hist:
            m = np.mean(np.array(hist), axis=0)
            top = int(np.argmax(m))
            branch_notes[name] = f"{DIAGNOSTIC_LABELS[top]} | {m[top]*100:.1f}%"

    report   = _format_report(valid_count, len(slices), verdict, confidence, entropy, avg_probs, branch_history)
    pdf_path = _generate_pdf_report(verdict, confidence, entropy, valid_count, branch_notes, best_visual, prob_dict)

    return report, best_visual, prob_dict, pdf_path


# ─── STATUS SIDEBAR ───────────────────────────────────────────────────────────

def _chip(label: str, ready: bool) -> str:
    cls  = "ready" if ready else "offline"
    txt  = "Ready" if ready else "Not loaded"
    return (
        f'<div class="model-chip {cls}">'
        f'<div class="model-chip__label">{label}</div>'
        f'<div class="model-chip__state">{txt}</div>'
        f'</div>'
    )


def build_status_html(n_files: int = 0) -> str:
    chips = "".join([
        _chip("Gatekeeper", SYSTEM["gatekeeper"] is not None),
        _chip("Hunter",     SYSTEM["hunter"]     is not None),
        _chip("SwinV2",     SYSTEM["council"]    is not None),
        _chip("ConvNeXt",   SYSTEM["council"]    is not None),
        _chip("MONAI",      SYSTEM["explainer"]  is not None),
    ])
    return (
        f'<div class="status-shell">'
        f'<div class="status-stat"><div class="status-stat__value">{n_files}</div>'
        f'<div class="status-stat__label">Uploaded files</div></div>'
        f'<div class="status-stat"><div class="status-stat__value">{MAX_INFERENCE_SLICES}</div>'
        f'<div class="status-stat__label">Slice capacity</div></div>'
        f'<div class="status-chip-grid">{chips}</div>'
        f'</div>'
    )


def on_upload(files: Any) -> str:
    return build_status_html(len(files) if files else 0)


# ─── CSS ──────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
:root {
  --ink: #10273d; --navy: #18344c; --accent: #1f7a8c;
  --panel: rgba(255,255,255,0.82); --border: rgba(16,39,61,0.12);
  --muted: #60758a; --alert: #c4533f; --radius: 22px;
  --shadow: 0 24px 70px rgba(17,38,56,0.08);
}
body,.gradio-container{background:linear-gradient(180deg,#faf6ee,#f3ecdf)!important;color:var(--ink)!important;font-family:"IBM Plex Sans","Segoe UI",sans-serif!important;}
.gradio-container{max-width:1380px!important;}
#hydra-hero{background:linear-gradient(120deg,rgba(16,39,61,.96),rgba(31,122,140,.88));color:#f9fbfc;border-radius:30px;padding:34px 38px;box-shadow:0 26px 80px rgba(16,39,61,.18);}
#hydra-hero h1{font-size:2.6rem;letter-spacing:-.05em;margin:0 0 12px;}
#hydra-hero p{max-width:720px;font-size:1rem;line-height:1.7;color:rgba(249,251,252,.86);}
.hero-ribbon{display:inline-flex;gap:10px;flex-wrap:wrap;margin-top:18px;}
.hero-ribbon span{border:1px solid rgba(255,255,255,.18);background:rgba(255,255,255,.10);padding:8px 12px;border-radius:999px;font-size:.84rem;}
.hydra-card{background:var(--panel)!important;backdrop-filter:blur(18px);border:1px solid var(--border)!important;border-radius:var(--radius)!important;box-shadow:var(--shadow);}
#run-analysis{background:linear-gradient(120deg,var(--accent),#145a68)!important;border:none!important;color:white!important;min-height:54px!important;border-radius:16px!important;font-weight:700!important;}
.section-kicker{color:#145a68;font-size:.76rem;text-transform:uppercase;letter-spacing:.12em;font-weight:700;margin-bottom:10px;}
.status-shell{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;}
.status-stat{background:white;border:1px solid var(--border);border-radius:18px;padding:18px;}
.status-stat__value{font-size:1.8rem;font-weight:800;color:var(--ink);}
.status-stat__label{font-size:.8rem;color:var(--muted);margin-top:3px;}
.status-chip-grid{grid-column:1/-1;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;}
.model-chip{border-radius:18px;padding:14px 16px;border:1px solid var(--border);background:white;}
.model-chip__label{font-weight:700;color:var(--ink);font-size:.92rem;}
.model-chip__state{margin-top:4px;font-size:.78rem;}
.model-chip.ready .model-chip__state{color:#2d7e55;}
.model-chip.offline .model-chip__state{color:var(--alert);}
.footnote{text-align:center;color:var(--muted);font-size:.8rem;padding:14px 0 6px;}
"""


# ─── GRADIO UI ────────────────────────────────────────────────────────────────

with gr.Blocks(title="HYDRA Clinical Console", css=CUSTOM_CSS) as app:
    gr.HTML(
        """
        <div id="hydra-hero">
          <h1>HYDRA Clinical Console</h1>
          <p>
            Multi-format brain tumour analysis — MRI, CT, DICOM, NIfTI.
            Handles 200–300 slices per patient with full-study aggregation.
          </p>
          <div class="hero-ribbon">
            <span>Gatekeeper screening</span>
            <span>3-branch Council</span>
            <span>Volumetric aggregation</span>
            <span>PDF export</span>
          </div>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=5, min_width=330):
            with gr.Group(elem_classes=["hydra-card"]):
                gr.HTML('<div class="section-kicker">Study Intake</div>')
                upload = gr.File(
                    file_count="multiple",
                    label="Upload DICOM / NIfTI / MRI / CT files",
                    height=170,
                )
                run_btn = gr.Button("Run Clinical Analysis", elem_id="run-analysis")

            with gr.Group(elem_classes=["hydra-card"]):
                gr.HTML('<div class="section-kicker">System Readiness</div>')
                status_html = gr.HTML(build_status_html())

        with gr.Column(scale=8):
            with gr.Tabs():
                with gr.Tab("Diagnostic Report"):
                    report_md = gr.Markdown(
                        "# System Idle\nUpload a study and press **Run Clinical Analysis**."
                    )
                with gr.Tab("Visual Review"):
                    with gr.Row():
                        prob_label = gr.Label(num_top_classes=5, label="Consensus probability")
                        heatmap_img = gr.Image(label="Saliency map")
                with gr.Tab("Export"):
                    pdf_out = gr.File(label="Clinical PDF export", interactive=False)

    gr.HTML('<div class="footnote">HYDRA — Research & educational use only.</div>')

    upload.change(on_upload, inputs=upload, outputs=status_html)
    run_btn.click(
        run_full_diagnostic,
        inputs=upload,
        outputs=[report_md, heatmap_img, prob_label, pdf_out],
    )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
