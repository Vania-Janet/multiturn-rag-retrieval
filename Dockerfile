# Multi-Turn RAG Retrieval - Production Docker Image
# Optimized for reproducibility and GPU acceleration

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/cache \
    HUGGINGFACE_HUB_CACHE=/workspace/cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/cache/transformers

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /workspace/mt-rag-benchmark/task_a_retrieval

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p \
    /workspace/cache/huggingface \
    /workspace/cache/transformers \
    data/retrieval_tasks \
    data/submissions \
    experiments \
    indices \
    logs

# Set permissions
RUN chmod +x *.sh 2>/dev/null || true && \
    chmod +x scripts/*.sh 2>/dev/null || true

# Default command
CMD ["/bin/bash"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python --version || exit 1

# Labels
LABEL maintainer="vania-janet" \
      description="Multi-Turn RAG Retrieval System" \
      version="1.0" \
      repository="https://github.com/Vania-Janet/multiturn-rag-retrieval"
