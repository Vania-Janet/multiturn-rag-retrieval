#!/bin/bash

echo "=========================================="
echo "RE-RUNNING GROUND TRUTH REWRITE EXPERIMENTS"
echo "=========================================="

source .venv/bin/activate

# BGE-1.5 + Ground Truth Rewrites
echo ""
echo "=== BGE-1.5 + GT Rewrites ==="
for domain in clapnq cloud fiqa govt; do
    echo "Running BGE-1.5 rewrite for $domain..."
    python -m src.pipeline.run \
        --config configs/experiments/02-hybrid/hybrid_splade_bge15_rewrite.yaml \
        --domain $domain \
        --force
done

# Voyage-3 + Ground Truth Rewrites  
echo ""
echo "=== Voyage-3 + GT Rewrites ==="
for domain in clapnq cloud fiqa govt; do
    echo "Running Voyage-3 rewrite for $domain..."
    python -m src.pipeline.run \
        --config configs/experiments/02-hybrid/hybrid_splade_voyage_rewrite.yaml \
        --domain $domain \
        --force
done

echo ""
echo "=========================================="
echo "✓ ALL GROUND TRUTH REWRITE EXPERIMENTS COMPLETED"
echo "=========================================="
