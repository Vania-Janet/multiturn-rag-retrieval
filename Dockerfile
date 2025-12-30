# Dockerfile for MT-RAG Benchmark (Task A: Retrieval)
# Reproducible environment for ACL paper experiments

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Metadata
LABEL maintainer="MT-RAG Research Team"
LABEL description="Multi-Turn RAG Benchmark - Retrieval Task A"
LABEL version="1.0"

# Set environment variables for reproducibility
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=0
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create working directory
WORKDIR /workspace

# Copy requirements first for Docker layer caching
COPY requirements-frozen.txt requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r requirements-frozen.txt && \
    pip install --no-cache-dir faiss-gpu

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt', quiet=True)"

# Copy project files
COPY . .

# Set PYTHONPATH
ENV PYTHONPATH=/workspace/src:$PYTHONPATH

# Create necessary directories
RUN mkdir -p indices logs experiments data

# Verify GPU setup
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" || true

# Default command
CMD ["bash"]
