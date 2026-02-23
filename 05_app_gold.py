import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
import numpy as np
import timm
from ultralytics import YOLO
from monai.networks.nets import SwinUNETR
from monai.visualize import GradCAM
import cv2
import os
import json
import csv
import nibabel as nib
import pydicom
from datetime import datetime
from fpdf import FPDF

# --- OPTIMIZATIONS ---
torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False) 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary', 'Tumor (Generic/CT)']

# --- PERFORMANCE STATS ---
MODEL_STATS = {
    "SwinV2": "Acc: 98.1% | F1: 97.9%",
    "ConvNeXt": "Acc: 97.8% | F1: 97.5%",
    "MONAI": "Acc: 98.3% | F1: 98.0%",
    "Ensemble": "Acc: 98.25% | F1: 97.83%"
}

print(f"🚀 LAUNCHING HYDRA PLATINUM (UX Upgrade + Progress Bars) on {DEVICE}")

# --- MODEL WRAPPERS ---
class MonaiSwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = SwinUNETR(spatial_dims=2, in_channels=3, out_channels=14, feature_size=24)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(384, 5))
    def forward(self, x): return self.head(self.swin.swinViT(x, normalize=True)[-1])

# --- LOAD SYSTEM ---
def load_all():
    sys = {}
    try:
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(1280, 2)
        m.load_state_dict(torch.load("HYDRA_Gatekeeper_Clinical.pth", map_location=DEVICE, weights_only=True))
        sys['GK'] = m.to(DEVICE).eval()
        if DEVICE.type == 'cuda': sys['GK'].half()
        with open("gatekeeper_classes.json", "r") as f: sys['GK_MAP'] = json.load(f)
    except: print("⚠️ Gatekeeper Missing")

    path = "runs/detect/hydra_hunter_clinical/weights/best.pt"
    if os.path.exists(path): sys['HUNTER'] = YOLO(path)
    else: sys['HUNTER'] = YOLO("yolo11n.pt")

    try:
        m = MonaiSwin().to(DEVICE)
        m.load_state_dict(torch.load("HYDRA_MONAI_5C.pth", map_location=DEVICE, weights_only=True))
        s = timm.create_model('swinv2_tiny_window8_256', pretrained=False, num_classes=5).to(DEVICE)
        s.load_state_dict(torch.load("HYDRA_Swin_5C.pth", map_location=DEVICE, weights_only=True))
        c = timm.create_model('convnextv2_nano', pretrained=False, num_classes=5).to(DEVICE)
        c.load_state_dict(torch.load("HYDRA_ConvNext_5C.pth", map_location=DEVICE, weights_only=True))
        
        # --- MIXED PRECISION SETUP ---
        if DEVICE.type == 'cuda':
            s.half().eval() # Swin -> FP16
            c.half().eval() # ConvNeXt -> FP16
            m.eval()        # MONAI -> FP32 (Critical Stability Fix)
        else:
            m.eval(); s.eval(); c.eval()
            
        sys['VOTERS'] = [s, c, m]
        sys['MONAI'] = m 
    except: print("⚠️ Council Missing")
    return sys

SYS = load_all()

# --- TRANSFORMS ---
tfm_gk = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
tfm_cl = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(256), transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])

# --- ADVANCED PREPROCESSING ---
def strip_skull(img_np):
    try:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [c], -1, 255, -1)
            img_np = cv2.bitwise_and(img_np, img_np, mask=mask)
    except: pass
    return img_np

def normalize_medical_slice(img_data):
    img_data = img_data.astype(float)
    img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data) + 1e-8) * 255
    return img_data.astype(np.uint8)

# --- LOADING WITH PROGRESS ---
def load_patient_data(files, progress=None):
    processed_slices = []
    
    dicom_files = [f for f in files if (isinstance(f, str) and f.endswith('.dcm')) or (hasattr(f, 'name') and f.name.endswith('.dcm'))]
    
    if dicom_files:
        dcm_objects = []
        total = len(dicom_files)
        for i, f in enumerate(dicom_files):
            # Update UI every 5 files to not slow down loop
            if progress and i % 5 == 0: progress(i/total, desc=f"Reading DICOM {i}/{total}")
            
            f_path = f.name if hasattr(f, 'name') else f
            try: dcm_objects.append(pydicom.dcmread(f_path))
            except: continue
            
        try: dcm_objects.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except: dcm_objects.sort(key=lambda x: int(x.InstanceNumber))
        
        for ds in dcm_objects:
            img = normalize_medical_slice(ds.pixel_array)
            img = strip_skull(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
            processed_slices.append(Image.fromarray(img))
            
    else:
        try: files.sort(key=lambda x: x.name if hasattr(x, 'name') else x['path'] if isinstance(x, dict) else x)
        except: pass

        total = len(files)
        for i, f in enumerate(files):
            # Update UI
            if progress and i % 10 == 0: progress(i/total, desc=f"Loading Slice {i}/{total}")

            try:
                f_path = f.name if hasattr(f, 'name') else f['path'] if isinstance(f, dict) else f
                
                if f_path.endswith(('.nii', '.nii.gz')):
                    vol = nib.load(f_path).get_fdata()
                    total_slices = vol.shape[2]
                    start, end = int(total_slices * 0.15), int(total_slices * 0.85)
                    for j in range(start, end):
                        slc = normalize_medical_slice(vol[:, :, j])
                        slc_rgb = cv2.cvtColor(slc, cv2.COLOR_GRAY2RGB)
                        slc_rgb = strip_skull(slc_rgb)
                        processed_slices.append(Image.fromarray(slc_rgb))
                    break 
                
                elif f_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                    img = np.array(Image.open(f_path).convert('RGB'))
                    img = strip_skull(img)
                    processed_slices.append(Image.fromarray(img))
                    
            except Exception as e:
                print(f"⚠️ Load Error: {e}")
                continue

    # --- 160 Slice Cap ---
    MAX_SLICES = 160
    if len(processed_slices) > MAX_SLICES:
        step = len(processed_slices) // MAX_SLICES
        processed_slices = processed_slices[::step][:MAX_SLICES]
            
    return processed_slices

def generate_pdf(diagnosis, confidence, entropy, valid, model_details, visual_img, tumor_probs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 15, "HYDRA MEDICAL REPORT", 0, 1, 'C')
    pdf.set_font("Arial", '', 12)
    pdf.ln(10)
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1)
    
    pdf.ln(5); pdf.set_font("Arial", 'B', 14)
    color = (220, 20, 60) if "Tumor" in diagnosis and "No" not in diagnosis else (0, 128, 0)
    pdf.set_text_color(*color)
    pdf.cell(0, 10, f"Result: {diagnosis}", 0, 1)
    pdf.set_text_color(0, 0, 0); pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"Prediction Probability: {confidence*100:.1f}% | Entropy: {entropy:.3f}", 0, 1)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Tumor Type Probability Distribution:", 0, 1)
    pdf.set_font("Arial", '', 10)
    for cls, prob in tumor_probs.items():
        pdf.cell(0, 6, f"- {cls}: {prob*100:.2f}%", 0, 1)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Model Council Metrics:", 0, 1)
    pdf.set_font("Arial", '', 10)
    for model, stats in model_details.items():
        pdf.cell(0, 6, f"{model}: {stats}", 0, 1)

    if visual_img is not None:
        Image.fromarray(visual_img).save("temp_vis.jpg")
        pdf.ln(10); pdf.image("temp_vis.jpg", x=55, w=100)
    
    fname = f"Report_{int(datetime.now().timestamp())}.pdf"
    pdf.output(fname)
    return fname

def generate_gradcam(model, input_tensor, raw_img_np):
    with torch.enable_grad():
        try:
            cam = GradCAM(nn_module=model, target_layers="swin.swinViT.layers.3")
            result = cam(x=input_tensor.float())
        except: return raw_img_np
    
    heatmap = result[0][0].detach().cpu().numpy()
    heatmap = cv2.resize(heatmap, (raw_img_np.shape[1], raw_img_np.shape[0]))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(raw_img_np, 0.6, heatmap, 0.4, 0)

# --- CORE ANALYTICS ---
def analyze_patient(files, progress=gr.Progress()):
    torch.cuda.empty_cache()
    if not files: return "Waiting...", None, None, None
    
    # 1. Load Data (With Progress)
    progress(0, desc="Initializing Loader...")
    slices = load_patient_data(files, progress)
    
    if not slices: return "❌ Error: No valid medical data found.", None, None, None

    total_probs = np.zeros(5)
    valid_slices = 0
    best_visual = None
    max_tumor_conf = 0
    
    model_names = ["SwinV2", "ConvNeXt", "MONAI"]
    model_raw_outputs = {name: [] for name in model_names}

    # 2. Analyze Slices (With Progress)
    for idx, raw in enumerate(progress.tqdm(slices, desc="Analysing Slices...")):
        raw.thumbnail((1024, 1024))
        
        # GATEKEEPER
        if 'GK' in SYS:
            not_brain_idx = SYS['GK_MAP'].get('NotBrain', 1)
            with torch.no_grad():
                inp_gk = tfm_gk(raw).unsqueeze(0).to(DEVICE)
                if DEVICE.type == 'cuda': inp_gk = inp_gk.half()
                if torch.softmax(SYS['GK'](inp_gk), 1)[0][not_brain_idx] > 0.70: continue

        # COUNCIL
        if 'VOTERS' in SYS:
            img_t = tfm_cl(raw)
            img_flip = tfm_cl(ImageOps.mirror(raw))
            batch = torch.stack([img_t, img_flip]).to(DEVICE)
            
            if DEVICE.type == 'cuda':
                batch_fp16 = batch.half()
                batch_fp32 = batch.float()
            else:
                batch_fp16 = batch; batch_fp32 = batch
            
            probs = []
            for i, model in enumerate(SYS['VOTERS']):
                with torch.no_grad():
                    if model == SYS['MONAI']:
                        logits = model(batch_fp32)
                    else:
                        with torch.cuda.amp.autocast():
                            logits = model(batch_fp16)
                        
                    prob = torch.softmax(logits / 1.5, 1).mean(dim=0).float().detach().cpu().numpy()
                    probs.append(prob)
                    model_raw_outputs[model_names[i]].append(prob)

            stacked = np.vstack(probs)
            avg_prob = (0.5 * np.mean(stacked, axis=0)) + (0.5 * np.max(stacked, axis=0))
            total_probs += avg_prob
            valid_slices += 1
            
            tumor_conf = 1.0 - avg_prob[2]
            if tumor_conf > max_tumor_conf:
                max_tumor_conf = tumor_conf
                if 'MONAI' in SYS:
                    best_visual = generate_gradcam(SYS['MONAI'], img_t.unsqueeze(0).to(DEVICE), np.array(raw))
                else:
                    best_visual = np.array(raw)

    if valid_slices == 0: return "⛔ REJECTED: Non-Brain Images.", None, None, None

    # --- REPORT GENERATION ---
    final_probs = total_probs / valid_slices
    no_tumor_prob = final_probs[2]
    tumor_prob = 1.0 - no_tumor_prob
    diagnosis = "Tumor Detected" if tumor_prob > 0.50 else "No Tumor Detected"
    confidence = tumor_prob if tumor_prob > 0.50 else no_tumor_prob
    entropy = -np.sum(final_probs * np.log(final_probs + 1e-8))
    strength = "High" if confidence > 0.85 else "Moderate" if confidence > 0.65 else "Low"
    if best_visual is None: best_visual = np.array(slices[0])

    details_str = ""
    pdf_details = {}
    
    for name in model_names:
        if model_raw_outputs[name]:
            probs_arr = np.array(model_raw_outputs[name])
            mean_probs = np.mean(probs_arr, axis=0)
            pred_idx = np.argmax(mean_probs)
            pred_class = CLASSES[pred_idx]
            stability = 1.0 - np.mean(np.std(probs_arr, axis=0))
            
            slice_predictions = np.argmax(probs_arr, axis=1)
            tumor_burden = (np.sum(slice_predictions != 2) / len(slice_predictions)) * 100
            
            clean_probs = {k: float(v) for k, v in zip(CLASSES, np.round(mean_probs, 3))}

            details_str += f"""
### 🤖 {name} Analysis
* **Prediction:** `{pred_class}`
* **Avg Confidence:** {mean_probs[pred_idx]*100:.1f}%
* **Stability Score:** {stability*100:.1f}%
* **Tumor Burden:** {tumor_burden:.1f}%
"""
            pdf_details[name] = f"{pred_class} | Conf: {mean_probs[pred_idx]*100:.1f}% | Vol: {tumor_burden:.1f}%"

    clean_final_probs = {k: float(v) for k, v in zip(CLASSES, np.round(final_probs, 3))}

    tumor_findings = []
    for tumor_type in CLASSES:
        prob = clean_final_probs[tumor_type] * 100
        tumor_findings.append(f"- {tumor_type}: {prob:.2f}%")
    tumor_findings_str = "\n".join(tumor_findings)

    report = f"""
# 🏥 HYDRA Ultimate Analysis Report
**Scan Mode:** {'3D Volumetric' if len(slices) > 1 else '2D Single-Slice'}
**Slices Analyzed:** {valid_slices} / {len(slices)}

{details_str}

---
### 🏁 Ensemble Final Verdict
* **Diagnosis:** `{diagnosis}`
* **Prediction Probability:** **{confidence*100:.2f}%**
* **Verdict Strength:** **{strength}**
* **Entropy (Uncertainty):** {entropy:.4f}

#### 🧬 Brain Tumor Type Findings (Probability Distribution)
{tumor_findings_str}
"""
    pdf_path = generate_pdf(diagnosis, confidence, entropy, valid_slices, pdf_details, best_visual, clean_final_probs)
    
    return report, best_visual, clean_final_probs, pdf_path

# --- UI HELPER ---
def update_file_count(files):
    if not files: return "No files uploaded."
    return f"✅ **{len(files)} slices ready for analysis.**"

# --- UI LAYOUT ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 HYDRA: Research-Grade Neuro-Oncology System")
    
    with gr.Row():
        file_input = gr.File(
            file_count="multiple", 
            label="📂 Upload Scan (NIfTI, DICOM, TIF, PNG, JPG)",
            height=100
        )
    
    # File Counter Text
    file_count_text = gr.Markdown("No files uploaded.")
    file_input.change(update_file_count, file_input, file_count_text)
    
    with gr.Row():
        with gr.Column(scale=1):
            report_output = gr.Markdown(label="Clinical Report")
            pdf_download = gr.File(label="📄 Download PDF")
        
        with gr.Column(scale=1):
            probs_output = gr.Label(num_top_classes=5, label="Tumor Type Probabilities")
            image_output = gr.Image(label="🧠 Localization Heatmap (Grad-CAM)")
            
    gr.Button("Run Full Analysis").click(
        analyze_patient, 
        file_input, 
        [report_output, image_output, probs_output, pdf_download]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)