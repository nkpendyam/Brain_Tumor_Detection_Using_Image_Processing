# Base Image with CUDA 12.1 support (compatible with RTX 5060)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables to prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
# We need python3, pip, git, and libraries for OpenCV (libgl1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create a symlink so 'python' calls python3.10
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# We explicitly upgrade pip first
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Gradio runs on
EXPOSE 7860

# Define the command to run your app
# We use the analytics dashboard version as the default entry point
CMD ["python", "05_app_gold.py"]