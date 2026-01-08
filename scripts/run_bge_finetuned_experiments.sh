#!/bin/bash

# Run all BGE fine-tuned experiments

set -e

echo "========================================================================"
echo "  RUNNING BGE FINE-TUNED EXPERIMENTS"
echo "========================================================================"
echo ""

export PYTHONPATH=$PWD/src:$PYTHONPATH

# Check if fine-tuned model exists
MODEL_DIR="models/bge-reranker-v2-m3-finetuned"
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Fine-tuned model not found: $MODEL_DIR"
    echo "Run: bash scripts/train_bge_reranker.sh first"
    exit 1
fi

echo "Using fine-tuned model: $MODEL_DIR"
echo ""

# Run experiments in parallel (2 GPUs)
echo "Starting experiments..."
echo ""

# GPU 0: CLAPNQ and GOVT
(
    echo "[GPU 0] Running CLAPNQ..."
    CUDA_VISIBLE_DEVICES=0 python -m pipeline.run \
        --config configs/experiments/05-finetune/finetune_bge_splade_voyage_rewrite.yaml \
        --domain clapnq
    
    echo "[GPU 0] Running GOVT..."
    CUDA_VISIBLE_DEVICES=0 python -m pipeline.run \
        --config configs/experiments/05-finetune/finetune_bge_splade_voyage_rewrite.yaml \
        --domain govt
) &

# GPU 1: CLOUD and FIQA
(
    echo "[GPU 1] Running CLOUD..."
    CUDA_VISIBLE_DEVICES=1 python -m pipeline.run \
        --config configs/experiments/05-finetune/finetune_bge_splade_bge15_rewrite.yaml \
        --domain cloud
    
    echo "[GPU 1] Running FIQA..."
    CUDA_VISIBLE_DEVICES=1 python -m pipeline.run \
        --config configs/experiments/05-finetune/finetune_bge_splade_bge15_rewrite.yaml \
        --domain fiqa
) &

# Wait for both parallel jobs
wait

echo ""
echo "========================================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo "  - experiments/finetune_bge_splade_voyage_rewrite/{clapnq,govt}/"
echo "  - experiments/finetune_bge_splade_bge15_rewrite/{cloud,fiqa}/"
echo ""
echo "Calculate metrics:"
echo "  python3 scripts/recalc_metrics.py --experiment finetune_bge_splade_voyage_rewrite"
echo "  python3 scripts/recalc_metrics.py --experiment finetune_bge_splade_bge15_rewrite"
echo ""
echo "========================================================================"
