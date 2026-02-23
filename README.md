# 🏥 Brain Tumor Detection Using Image Processing

## 1. Problem Statement

Brain tumor diagnosis relies heavily on the manual analysis of complex radiological scans (such as MRIs and CTs). This process is time-consuming, susceptible to human error, and subject to variability among radiologists. While automated deep learning systems exist, they frequently suffer from critical flaws: a lack of generalization across different scan modalities (2D slices vs. 3D volumes), an inability to differentiate between multiple specific tumor subtypes, and a dangerous lack of clinical safety mechanisms (often attempting to diagnose non-brain or non-medical images).

There is a critical need for a robust, failsafe clinical tool that not only classifies multiple tumor types with high accuracy but also validates input data and provides explainable, transparent results for medical professionals.

## 2. Proposed Solution

The **HYDRA System** is a comprehensive, automated image processing and classification pipeline. Instead of relying on a single model, HYDRA introduces a multi-tiered, fault-tolerant architecture:

* **Failsafe Input Validation:** A "Gatekeeper" system actively rejects irrelevant or non-clinical data (like chest X-rays or random images) before processing.
* **Advanced Preprocessing:** Automatically normalizes medical slices and performs algorithmic skull-stripping using OpenCV contours to isolate brain tissue.
* **Ensemble Intelligence (The Council):** Utilizes a trio of advanced Vision Transformers and pure Convolutional Neural Networks (SwinV2, ConvNeXt V2, and MONAI SwinUNETR) leveraging mixed-precision inference to classify scans into 5 distinct categories (Glioma, Meningioma, Pituitary, Generic CT Tumor, and No Tumor).
* **Clinical Explainability:** Generates Grad-CAM localization heatmaps to show exactly where the models are looking, calculates diagnostic entropy (uncertainty), and automatically compiles a comprehensive, exportable PDF medical report.

## 3. Detailed Module Design

### Module 1: Data Acquisition & Algorithmic Preprocessing

* **Function:** Handles the ingestion of diverse medical formats (DICOM, NIfTI, PNG, JPG).
* **Process:** Extracts 2D slices from 3D volumetric data, normalizes pixel intensities, and utilizes OpenCV contour detection and Otsu's thresholding to strip the skull and isolate relevant brain matter.

### Module 2: The Gatekeeper (Safety Validation)

* **Function:** Acts as the first line of defense to ensure clinical relevance.
* **Process:** An EfficientNet-B0 model trained on a balanced dataset of brain scans vs. "Not Brain" images (X-rays, human faces). If an image is flagged as non-brain with high confidence (>70%), the pipeline rejects it, preventing false diagnostics.

### Module 3: The Hunter (Localization & Detection)

* **Function:** Rapidly scans the validated brain images to detect the presence and bounding box of anomalous masses.
* **Process:** Utilizes a fine-tuned YOLO11 architecture optimized for high-speed object detection to flag areas of interest.

### Module 4: The Council (Ensemble Classification)

* **Function:** Performs the final, high-fidelity diagnostic classification.
* **Process:** A weighted voting system combining SwinV2 (hierarchical transformer), ConvNeXt V2 (modern CNN), and MONAI (medical-specific transformer). It processes standard and mirrored images simultaneously, outputting a probability distribution across 5 tumor classes.

### Module 5: UI/UX & Clinical Reporting

* **Function:** Bridges the gap between the AI and the medical professional.
* **Process:** A Gradio-based web frontend that displays real-time processing progress, visualizes Grad-CAM heatmaps for explainability, and uses FPDF to generate a downloadable, timestamped clinical report containing confidence scores, tumor burden metrics, and stability metrics.

## 4. Tools and Technologies Used

| Category | Technologies |
| --- | --- |
| **Programming Language** | Python 3.10 |
| **Deep Learning Frameworks** | PyTorch, Ultralytics (YOLO) |
| **Specialized AI Libraries** | TIMM (PyTorch Image Models), MONAI (Medical Open Network for AI) |
| **Image & Data Processing** | OpenCV (cv2), Pillow (PIL), nibabel (NIfTI), pydicom |
| **Data Science & Math** | NumPy, Scikit-learn |
| **Frontend & Utilities** | Gradio, FPDF (PDF generation) |
| **Infrastructure & Deployment** | Docker, NVIDIA CUDA 12.1.1, cuDNN |

## 5. Detailed Component Breakdown

### System Files and Scripts

* `01_data_gold.py`: Automates the downloading and structuring of all Kaggle datasets. It routes specific MRI and CT classes into a unified 5-class structure and gathers negative datasets.
* `02_train_gatekeeper_gold.py`: Trains the EfficientNet-B0 Gatekeeper model using heavy augmentations (random rotation, color jitter) to distinguish brain images from negative samples.
* `03_train_hunter_gold.py`: Executes the training loop for the YOLO detection model using resumable turbo mode, mixed precision (AMP), and mosaic augmentations.
* `04_train_council_gold.py`: The core training script for the ensemble models. It implements stratified train-test splits, calculates balanced class weights, and outputs detailed classification reports (F1-scores).
* `05_app_gold.py`: The main application file. It houses the inference logic, the Gradio user interface, Grad-CAM generation, PDF compilation, and handles the FP16/FP32 mixed-precision stability optimizations.
* 
`Dockerfile`: Containerizes the application using a base image with CUDA 12.1 support. It sets the working directory inside the container , installs Python dependencies , and copies the application code to prepare for deployment.



### Architectures Used

* **EfficientNet-B0:** A lightweight, highly efficient CNN used for the binary Gatekeeper task.
* **YOLO11 (Ultralytics):** State-of-the-art real-time object detection system used for the Hunter module.
* **SwinV2 (Tiny):** A shifted-window Vision Transformer excelling at understanding hierarchical image structures.
* **ConvNeXt V2 (Nano):** A modernized pure Convolutional Network providing diverse feature extraction for the ensemble.
* **SwinUNETR (MONAI):** A specialized architecture designed explicitly for medical image analysis, adapted for robust 2D feature extraction.

### Datasets
* **[Brain Tumor MRI Dataset (Masoud Nickparvar)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset):** The primary Kaggle dataset providing the foundational classes: Glioma, Meningioma, Pituitary, and No Tumor.
* **[Brain Tumor Detection Dataset (Ahmed Hamada)](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection):** A binary CT dataset utilized to inject a 5th class ("Generic Tumor/CT"), ensuring the model can handle non-MRI modalities without crashing.

### Negative Datasets (For Gatekeeper Training)
* **[Chest X-Ray Pneumonia (Paul Timothy Mooney)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia):** Provides 1,500 non-brain medical scans to teach the system what an incorrect medical input looks like.
* **[CelebA Dataset (Jessica Li)](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset):** Provides 1,500 human face images to teach the system to reject standard, non-medical photography.
