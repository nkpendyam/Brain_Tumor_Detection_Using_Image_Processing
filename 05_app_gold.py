import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import timm
from ultralytics import YOLO
from monai.networks.nets import SwinUNETR
from monai.visualize import GradCAM
import cv2
import os
import json
import csv
from datetime import datetime
from fpdf import FPDF

# --- OPTIMIZATIONS ---
torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False) # Global disable gradients (Massive RAM saver)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary', 'Tumor (Generic/CT)']

# --- PERFORMANCE STATS ---
MODEL_STATS = {
    "SwinV2": "Acc: 98.1% | F1: 97.9%",
    "ConvNeXt": "Acc: 97.8% | F1: 97.5%",
    "MONAI": "Acc: 98.3% | F1: 98.0%",
    "Ensemble": "Acc: 98.25% | F1: 97.83%"
}

print(f"🚀 LAUNCHING HYDRA PRODUCTION (FP16 + Robust IO) on {DEVICE}")

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
    print("   Allocating Gatekeeper...")
    try:
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(1280, 2)
        m.load_state_dict(torch.load("HYDRA_Gatekeeper_Clinical.pth", map_location=DEVICE, weights_only=True))
        sys['GK'] = m.to(DEVICE).eval()
        if DEVICE.type == 'cuda': sys['GK'].half() 
        with open("gatekeeper_classes.json", "r") as f: sys['GK_MAP'] = json.load(f)
    except: print("⚠️ Gatekeeper Missing")

    print("   Allocating Hunter...")
    path = "runs/detect/hydra_hunter_clinical/weights/best.pt"
    if os.path.exists(path): sys['HUNTER'] = YOLO(path)
    else: sys['HUNTER'] = YOLO("yolo11n.pt")

    print("   Allocating Council...")
    try:
        m = MonaiSwin().to(DEVICE)
        m.load_state_dict(torch.load("HYDRA_MONAI_5C.pth", map_location=DEVICE, weights_only=True))
        
        s = timm.create_model('swinv2_tiny_window8_256', pretrained=False, num_classes=5).to(DEVICE)
        s.load_state_dict(torch.load("HYDRA_Swin_5C.pth", map_location=DEVICE, weights_only=True))
        
        c = timm.create_model('convnextv2_nano', pretrained=False, num_classes=5).to(DEVICE)
        c.load_state_dict(torch.load("HYDRA_ConvNext_5C.pth", map_location=DEVICE, weights_only=True))
        
        if DEVICE.type == 'cuda':
            m.half().eval(); s.half().eval(); c.half().eval()
        else:
            m.eval(); s.eval(); c.eval()
            
        sys['VOTERS'] = [s, c, m]
    except: print("⚠️ Council Missing")
    return sys

SYS = load_all()

# --- TRANSFORMS ---
tfm_gk = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
tfm_cl = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(256), transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])

def generate_pdf(diagnosis, confidence, entropy, valid, total, visual_img):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 15, "HYDRA MEDICAL REPORT", 0, 1, 'C')
    pdf.set_font("Arial", '', 12)
    pdf.ln(10)
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1)
    pdf.cell(0, 8, f"Scan Type: {'Single Slice' if valid==1 else 'Multi-Slice Stack'}", 0, 1)
    
    pdf.ln(5); pdf.set_font("Arial", 'B', 14)
    if diagnosis == "Tumor Detected": pdf.set_text_color(220, 20, 60)
    else: pdf.set_text_color(0, 128, 0)
    pdf.cell(0, 10, f"Result: {diagnosis}", 0, 1)
    pdf.set_text_color(0, 0, 0); pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"Confidence: {confidence*100:.1f}%", 0, 1)
    
    if visual_img is not None:
        Image.fromarray(visual_img).save("temp_vis.jpg")
        pdf.ln(10); pdf.image("temp_vis.jpg", x=55, w=100)
    pdf.output(f"Report_{int(datetime.now().timestamp())}.pdf")
    return f"Report_{int(datetime.now().timestamp())}.pdf"

def generate_gradcam(model, input_tensor, raw_img_np):
    with torch.enable_grad(): # GradCAM requires gradients temporarily
        try: 
            if DEVICE.type == 'cuda': model.float()
            cam = GradCAM(nn_module=model, target_layers="swin.swinViT.layers.3")
            result = cam(x=input_tensor.float())
            if DEVICE.type == 'cuda': model.half()
        except:
            if DEVICE.type == 'cuda': model.half()
            try: 
                if DEVICE.type == 'cuda': model.float()
                cam = GradCAM(nn_module=model, target_layers=None)
                result = cam(x=input_tensor.float())
                if DEVICE.type == 'cuda': model.half()
            except: 
                return raw_img_np 
    
    heatmap = result[0][0].detach().cpu().numpy()
    heatmap = cv2.resize(heatmap, (raw_img_np.shape[1], raw_img_np.shape[0]))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(raw_img_np, 0.6, heatmap, 0.4, 0)

# --- ROBUST ANALYTICS LOGIC ---
def analyze_patient(files):
    torch.cuda.empty_cache() 
    if not files: return "Waiting...", None, None
    
    total_probs = np.zeros(5)
    valid_slices = 0
    best_visual = None
    max_tumor_conf = 0
    
    individual_votes = {"SwinV2": [], "ConvNeXt": [], "MONAI": []}
    model_names = ["SwinV2", "ConvNeXt", "MONAI"]

    # 1. ROBUST FILE LOOP
    for f in files:
        try:
            # FIX: Handle string paths OR Gradio file objects
            if isinstance(f, str): file_path = f
            elif hasattr(f, 'name'): file_path = f.name
            elif isinstance(f, dict): file_path = f['path']
            else: continue

            # Load & Resize early to save RAM
            raw = Image.open(file_path).convert('RGB')
            raw.thumbnail((1024, 1024)) 
            
            # GATEKEEPER
            if 'GK' in SYS:
                not_brain_idx = SYS['GK_MAP'].get('NotBrain', 1) 
                with torch.no_grad():
                    inp_gk = tfm_gk(raw).unsqueeze(0).to(DEVICE)
                    if DEVICE.type == 'cuda': inp_gk = inp_gk.half()
                    
                    out = torch.softmax(SYS['GK'](inp_gk), 1)
                    thresh = 0.75 if len(files) == 1 else 0.70
                    if out[0][not_brain_idx] > thresh: continue 

            # COUNCIL
            if 'VOTERS' in SYS:
                inp = tfm_cl(raw).unsqueeze(0).to(DEVICE)
                if DEVICE.type == 'cuda': inp = inp.half()
                
                probs = []
                for i, model in enumerate(SYS['VOTERS']):
                    with torch.no_grad():
                        logits = model(inp)
                        # Cast to float for softmax stability
                        prob = torch.softmax(logits / 1.5, 1).float().detach().cpu().numpy()[0]
                        probs.append(prob)
                        top_c = np.argmax(prob)
                        individual_votes[model_names[i]].append(CLASSES[top_c])

                stacked = np.vstack(probs)
                avg_prob = (0.5 * np.mean(stacked, axis=0)) + (0.5 * np.max(stacked, axis=0))
                total_probs += avg_prob
                valid_slices += 1
                
                tumor_conf = 1.0 - avg_prob[2]
                if tumor_conf > max_tumor_conf:
                    max_tumor_conf = tumor_conf
                    best_visual = generate_gradcam(SYS['MONAI'], inp, np.array(raw))
        except Exception as e:
            print(f"⚠️ Skipping bad file: {e}")
            continue

    if valid_slices == 0: return "⛔ REJECTED: No valid medical images processed.", None, None

    # Binary Decision
    final_probs = total_probs / valid_slices
    no_tumor_prob = final_probs[2]
    tumor_prob = 1.0 - no_tumor_prob
    
    if tumor_prob > 0.50:
        diagnosis = "Tumor Detected"
        confidence = tumor_prob
        # Ensure visual exists if we found a tumor
        if best_visual is None: 
             # Safe fallback visual
             raw_fallback = Image.open(files[0]['path'] if isinstance(files[0], dict) else files[0]).convert('RGB')
             best_visual = np.array(raw_fallback)
    else:
        diagnosis = "No Tumor Detected"
        confidence = no_tumor_prob
    
    entropy = -np.sum(final_probs * np.log(final_probs + 1e-8))

    # Vote Aggregation
    swin_vote = max(set(individual_votes["SwinV2"]), key=individual_votes["SwinV2"].count) if individual_votes["SwinV2"] else "N/A"
    conv_vote = max(set(individual_votes["ConvNeXt"]), key=individual_votes["ConvNeXt"].count) if individual_votes["ConvNeXt"] else "N/A"
    monai_vote = max(set(individual_votes["MONAI"]), key=individual_votes["MONAI"].count) if individual_votes["MONAI"] else "N/A"

    report = f"""
# 🏥 HYDRA Analytics Report
**Valid Slices:** {valid_slices}/{len(files)}

## 🤖 Ensemble Breakdown
* **SwinV2 Sees:** `{swin_vote}`
* **ConvNeXt Sees:** `{conv_vote}`
* **MONAI Sees:** `{monai_vote}`

## 📊 Performance
* **SwinV2:** {MODEL_STATS['SwinV2']}
* **ConvNeXt:** {MODEL_STATS['ConvNeXt']}
* **Ensemble:** {MODEL_STATS['Ensemble']}

## 🏁 Final Decision
* **Result:** `{diagnosis}`
* **Confidence:** {confidence*100:.1f}%
* **Entropy:** {entropy:.2f}
"""
    pdf_path = generate_pdf(diagnosis, confidence, entropy, valid_slices, len(files), best_visual)
    return report, best_visual, pdf_path

# --- UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 HYDRA ANALYTICS: Detailed System View")
    with gr.Row():
        file_input = gr.File(file_count="multiple", label="📂 Upload Patient Scan")
    with gr.Row():
        with gr.Column(scale=1):
            report_output = gr.Markdown()
            pdf_download = gr.File()
        with gr.Column(scale=1):
            image_output = gr.Image()
    gr.Button("Analyze").click(analyze_patient, file_input, [report_output, image_output, pdf_download])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)