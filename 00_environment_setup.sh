#!/usr/bin/env bash
# =============================================================================
# 00_environment_setup.sh — HYDRA WSL2 Environment Setup
# =============================================================================
# Run this ONCE to prepare your rtx50_env conda environment for the project.
#
# Prerequisites:
#   - WSL2 with Ubuntu 22.04
#   - NVIDIA drivers ≥ 570 installed on Windows host
#   - Miniforge3 installed at ~/miniforge3
#   - CUDA 12.8 toolkit accessible inside WSL
#
# Usage:
#   chmod +x 00_environment_setup.sh
#   bash 00_environment_setup.sh
# =============================================================================

set -e   # Exit on any error

CONDA_BASE="$HOME/miniforge3"
ENV_NAME="rtx50_env"
ENV_PATH="$CONDA_BASE/envs/$ENV_NAME"

echo "======================================================================"
echo "  HYDRA — WSL2 Environment Setup"
echo "  Target environment: $ENV_NAME"
echo "======================================================================"

# ── 0. Source conda ──────────────────────────────────────────────────────────
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ── 1. Check NVIDIA driver ───────────────────────────────────────────────────
echo ""
echo "[STEP 1] Checking NVIDIA driver …"
if ! command -v nvidia-smi &>/dev/null; then
    echo "[CRITICAL] nvidia-smi not found."
    echo "  Install NVIDIA drivers on your Windows host (≥570) and ensure"
    echo "  WSL2 GPU pass-through is enabled."
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "[OK] NVIDIA GPU detected."

# ── 2. Check / create conda environment ──────────────────────────────────────
echo ""
echo "[STEP 2] Checking conda environment …"
if conda env list | grep -q "^$ENV_NAME "; then
    echo "[OK] Environment '$ENV_NAME' already exists."
else
    echo "[INFO] Creating environment '$ENV_NAME' with Python 3.11 …"
    conda create -n "$ENV_NAME" python=3.11 -y
    echo "[OK] Environment created."
fi

# ── 3. Activate environment ───────────────────────────────────────────────────
echo ""
echo "[STEP 3] Activating '$ENV_NAME' …"
conda activate "$ENV_NAME"
echo "[OK] Active: $(which python) — $(python --version)"

# ── 4. Install PyTorch for CUDA 12.8 (RTX 5060 Blackwell) ────────────────────
echo ""
echo "[STEP 4] Installing PyTorch (CUDA 12.8 wheels) …"
echo "  This is required for RTX 5060 (Blackwell / SM_120 architecture)."
echo "  Standard CUDA 12.1 wheels will NOT run on your GPU."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 \
    --upgrade
echo "[OK] PyTorch installed."

# ── 5. Verify PyTorch + CUDA ──────────────────────────────────────────────────
echo ""
echo "[STEP 5] Verifying CUDA availability …"
python -c "
import torch
print(f'  PyTorch version : {torch.__version__}')
print(f'  CUDA available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU             : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM            : {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB')
    t = torch.ones(3, 3, device=\"cuda\")
    print(f'  Tensor on GPU   : {t.device}  ✓')
else:
    print('  [WARN] CUDA not available — check driver version and WSL2 setup.')
"

# ── 6. Install project requirements ───────────────────────────────────────────
echo ""
echo "[STEP 6] Installing project requirements …"
pip install -r requirements.txt
echo "[OK] Requirements installed."

# ── 7. Suppress TensorFlow + MLflow noise ────────────────────────────────────
echo ""
echo "[STEP 7] Configuring environment variables …"
ACTIVATE_DIR="$ENV_PATH/etc/conda/activate.d"
mkdir -p "$ACTIVATE_DIR"
cat > "$ACTIVATE_DIR/hydra_env.sh" <<'EOF'
# Silence TF oneDNN deprecation warnings
export TF_ENABLE_ONEDNN_OPTS=0
# Keep CUDA visible
export CUDA_VISIBLE_DEVICES=0
# Disable ClearML (it's not configured and generates warnings)
export CLEARML_WEB_HOST=""
EOF
echo "[OK] Environment variables set."

# ── 8. Setup Kaggle credentials ───────────────────────────────────────────────
echo ""
echo "[STEP 8] Checking Kaggle credentials …"
KAGGLE_DIR="$HOME/.kaggle"
KAGGLE_JSON="$KAGGLE_DIR/kaggle.json"

if [ -f "$KAGGLE_JSON" ]; then
    echo "[OK] kaggle.json found at $KAGGLE_JSON"
    chmod 600 "$KAGGLE_JSON"
    echo "[OK] Permissions set to 600."
else
    mkdir -p "$KAGGLE_DIR"
    # Copy from project if present
    if [ -f "kaggle.json" ]; then
        cp kaggle.json "$KAGGLE_JSON"
        chmod 600 "$KAGGLE_JSON"
        echo "[OK] kaggle.json copied from project directory."
    else
        echo "[WARN] kaggle.json not found."
        echo "  To download datasets, place your kaggle.json at: $KAGGLE_JSON"
        echo "  Get it from: https://www.kaggle.com/settings → API → Create New Token"
    fi
fi

# ── 9. Install Docker + NVIDIA Container Toolkit (optional) ───────────────────
echo ""
echo "[STEP 9] Docker check (optional) …"
if command -v docker &>/dev/null; then
    echo "[OK] Docker installed: $(docker --version)"
    if docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi &>/dev/null 2>&1; then
        echo "[OK] NVIDIA Container Toolkit — GPU pass-through to Docker works."
    else
        echo "[INFO] NVIDIA Container Toolkit may not be installed."
        echo "  To install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
    fi
else
    echo "[INFO] Docker not installed. It is optional — you can run scripts directly in rtx50_env."
fi

# ── 10. Summary ───────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "  Setup Complete!"
echo "======================================================================"
echo ""
echo "  To run the project:"
echo ""
echo "    conda activate $ENV_NAME"
echo "    cd ~/brain_fn          # or wherever you placed the project"
echo ""
echo "  Then follow COMMANDS.md for the full execution order."
echo ""
echo "  Quick start (if base data already exists and models are trained):"
echo "    python 06_clinical_diagnostic_interface.py"
echo "    → Open: http://localhost:7860"
echo ""
