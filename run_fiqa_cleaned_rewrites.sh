#!/bin/bash

# Correr experimentos de FiQA con rewrites limpios (sin |user|:)

source .venv/bin/activate

echo "=========================================="
echo "Corriendo FiQA con rewrites limpios"
echo "=========================================="

# Hybrid SPLADE + Voyage con rewrites limpios
echo -e "\n[1/2] Hybrid SPLADE + Voyage-3 (FiQA rewrites limpios)..."
python -m src.pipeline.run \
    --config configs/experiments/02-hybrid/hybrid_splade_voyage.yaml \
    --domain fiqa \
    --force

# Hybrid SPLADE + BGE-1.5 con rewrites limpios  
echo -e "\n[2/2] Hybrid SPLADE + BGE-1.5 (FiQA rewrites limpios)..."
python -m src.pipeline.run \
    --config configs/experiments/02-hybrid/hybrid_splade_bge15.yaml \
    --domain fiqa \
    --force

echo -e "\n=========================================="
echo "✓ Experimentos completados"
echo "=========================================="
