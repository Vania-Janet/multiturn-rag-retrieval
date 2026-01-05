#!/bin/bash
# Automated setup script for MT-RAG Benchmark (Task A: Retrieval)
# Ensures reproducible environment setup

set -e  # Exit on error

echo "================================================"
echo "MT-RAG Benchmark - Automated Setup"
echo "Task A: Retrieval"
echo "================================================"
echo ""

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "⚠️  Warning: This script is optimized for Linux (Ubuntu 22.04)"
    echo "   For macOS, some steps may need manual adjustment."
fi

# Check Python version
echo "1. Checking Python version..."
PYTHON_VERSION=$(python3.11 --version 2>/dev/null || echo "Not found")
if [[ $PYTHON_VERSION == *"3.11"* ]]; then
    echo "✓ Python 3.11 found: $PYTHON_VERSION"
else
    echo "✗ Python 3.11 not found. Please install it first."
    exit 1
fi

# Check CUDA
echo ""
echo "2. Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo "✓ NVIDIA GPU detected"
else
    echo "⚠️  Warning: nvidia-smi not found. GPU support may not be available."
fi

# Create virtual environment
echo ""
echo "3. Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3.11 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
echo ""
echo "4. Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA
echo ""
echo "5. Installing PyTorch with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
echo ""
echo "6. Installing project dependencies..."
if [ -f "requirements-frozen.txt" ]; then
    pip install -r requirements-frozen.txt
    echo "✓ Installed from requirements-frozen.txt (pinned versions)"
else
    pip install -r requirements.txt
    echo "✓ Installed from requirements.txt"
fi

# Install FAISS GPU
echo ""
echo "7. Installing FAISS GPU..."
pip uninstall -y faiss-cpu 2>/dev/null || true
pip install faiss-gpu

# Download NLTK data
echo ""
echo "8. Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True)"
echo "✓ NLTK data downloaded"

# Verify GPU setup
echo ""
echo "9. Verifying GPU setup..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import faiss; print(f'FAISS GPU available: {faiss.get_num_gpus() > 0}')" || echo "⚠️  FAISS GPU check failed"

# Create necessary directories
echo ""
echo "10. Creating directories..."
mkdir -p indices logs experiments data/retrieval_tasks data/passage_level_processed
echo "✓ Directories created"

# Check Docker (optional)
echo ""
echo "11. Checking Docker (optional for ELSER)..."
if command -v docker &> /dev/null; then
    echo "✓ Docker found: $(docker --version)"
else
    echo "⚠️  Docker not found. Install it if you want to use ELSER retrieval."
fi

# Final summary
echo ""
echo "================================================"
echo "✅ Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Activate the environment: source .venv/bin/activate"
echo "2. Build indices: python src/pipeline/indexing/build_indices.py --models bge bge-m3 bm25 --domains clapnq cloud fiqa govt"
echo "3. Run experiments: python scripts/run_experiment.py --experiment replication_bm25 --domain all"
echo ""
echo "For ELSER support:"
echo "  docker-compose up -d elasticsearch"
echo ""
