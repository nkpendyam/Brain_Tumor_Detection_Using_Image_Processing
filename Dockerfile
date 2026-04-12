# ─────────────────────────────────────────────────────────────────────────────
# HYDRA — Docker Image
# Base: CUDA 12.8 + cuDNN 9 — required for RTX 5060 (Blackwell / SM 120)
# ─────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04

# Suppress interactive timezone prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TZ=UTC

# ── System Dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-distutils \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    git \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# ── Python Dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .

# Install PyTorch for CUDA 12.8 first (Blackwell-compatible wheels)
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

# Install remaining requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# ── Application Code ──────────────────────────────────────────────────────────
COPY . .

# Kaggle credentials (mount at runtime — never bake credentials into image)
# Run: docker run -v ~/.kaggle:/root/.kaggle hydra-ai
ENV KAGGLE_CONFIG_DIR=/root/.kaggle

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 7860
ENV GRADIO_SERVER_NAME=0.0.0.0

CMD ["python", "06_clinical_diagnostic_interface.py"]
