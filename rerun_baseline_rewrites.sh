#!/bin/bash

echo "=========================================="
echo "RE-RUNNING BASELINE REWRITE EXPERIMENTS"
echo "With cleaned queries (no |user|: prefix)"
echo "=========================================="

source .venv/bin/activate

# Re-run FiQA experiments that failed (had 0 queries)
echo ""
echo "=== Re-running FiQA (was empty before) ==="

echo "Running FiQA with BGE-1.5 rewrite..."
python -m src.pipeline.run \
    --config configs/experiments/02-hybrid/hybrid_splade_bge15_rewrite.yaml \
    --domain fiqa \
    --force

echo "Running FiQA with Voyage-3 rewrite..."
python -m src.pipeline.run \
    --config configs/experiments/02-hybrid/hybrid_splade_voyage_rewrite.yaml \
    --domain fiqa \
    --force

echo ""
echo "=========================================="
echo "✓ BASELINE REWRITE EXPERIMENTS COMPLETED"
echo "=========================================="
