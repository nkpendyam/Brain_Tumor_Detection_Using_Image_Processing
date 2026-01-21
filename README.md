# 🏥 HYDRA: Clinical-Grade Brain Tumor Analysis System

![Python](https://img.shields.io/badge/Python-3.10-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red) ![Docker](https://img.shields.io/badge/Docker-Ready-blue) ![Gradio](https://img.shields.io/badge/UI-Gradio-orange)

**Hydra** is a production-ready medical AI pipeline designed for the detection, classification, and reporting of brain tumors from MRI and CT scans. It features a robust **Multi-Model Ensemble**, a **Safety Gatekeeper** to reject non-medical images, and a **Dockerized** deployment architecture.

## 🚀 Key Features

* **🛡️ Safety Gatekeeper**: An EfficientNet-B0 classifier that rejects non-brain images (faces, X-rays, documents) before analysis.
* **🧠 Multi-Model Council**: A weighted ensemble of **Swin Transformer V2**, **ConvNeXt V2**, and **MONAI Swin-UNETR** for high-accuracy classification (Acc: >98%).
* **🎯 Tumor Localization**: Integration with **YOLOv11** (Hunter) for object detection.
* **🔬 5-Class Diagnosis**: Capable of distinguishing **Glioma**, **Meningioma**, **Pituitary**, **No Tumor**, and **Generic/CT Tumors**.
* **📊 Clinical Analytics**: Generates downloadable PDF reports with entropy-based uncertainty estimation and Grad-CAM explainability maps.
* **🐳 Dockerized**: Fully containerized with NVIDIA GPU support for one-click deployment.

---

## 🛠️ Architecture

The system operates in three stages:
1.  **Gatekeeper**: Validates input data integrity.
2.  **Hunter**: Scans for tumor regions (Object Detection).
3.  **Council**: Performs deep classification using a voting ensemble.

| Component | Model Architecture | Role |
| :--- | :--- | :--- |
| **Gatekeeper** | EfficientNet-B0 | Input Validation (Brain vs. Non-Brain) |
| **Hunter** | YOLOv11n | Tumor Localization & Bounding Box |
| **Voter 1** | Swin Transformer V2 | Texture & Pattern Recognition |
| **Voter 2** | ConvNeXt V2 | Spatial Feature Extraction |
| **Voter 3** | MONAI Swin-UNETR | Medical-Specific Segmentation Features |

---

## ⚡ Quick Start (Docker)

The easiest way to run Hydra is via Docker. This ensures all CUDA and PyTorch dependencies are handled automatically.

### Prerequisites
* Docker Desktop (Windows) or Docker Engine (Linux)
* NVIDIA GPU (RTX 3060/4060/5060 or higher recommended)

### 1. Run with GPU Support
```bash
docker compose up --build