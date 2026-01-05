#!/bin/bash
# Quick setup script for HuggingFace sync

set -e

echo "================================================"
echo "HuggingFace Large Files Sync - Setup"
echo "================================================"
echo ""

# Check if huggingface-hub is installed
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface-hub..."
    pip install huggingface-hub
fi

# Run setup
echo ""
echo "Running HF authentication setup..."
python scripts/hf_sync.py --setup

echo ""
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Edit configs/hf_sync.yaml with your repo ID"
echo "2. Upload artifacts: python scripts/hf_sync.py --upload-all"
echo "3. Check status: python scripts/hf_sync.py --list"
echo ""
echo "Documentation: scripts/hf_sync.py --help"
echo ""
