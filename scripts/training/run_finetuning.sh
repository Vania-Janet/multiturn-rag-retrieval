#!/bin/bash
# Complete fine-tuning pipeline for domain-specific models

set -e

DOMAIN="clapnq"  # Change to your domain: clapnq, fiqa, govt, cloud

echo "========================================="
echo "Fine-tuning Pipeline for Domain: $DOMAIN"
echo "========================================="

# Step 1: Prepare training data with hard negatives
echo ""
echo "Step 1: Preparing training data with HARD NEGATIVES..."
python scripts/training/prepare_training_data.py \
  --domain $DOMAIN \
  --data_dir data \
  --output_dir data/training/$DOMAIN \
  --num_negatives 10 \
  --context_turns 3

# Step 2: Fine-tune reranker
echo ""
echo "Step 2: Fine-tuning reranker..."
python scripts/training/finetune_reranker.py \
  --domain $DOMAIN \
  --training_data_dir data/training/$DOMAIN \
  --base_model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --output_dir models/finetuned/reranker/$DOMAIN \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 2e-5

echo ""
echo "========================================="
echo "Fine-tuning complete!"
echo "Reranker saved to: models/finetuned/reranker/$DOMAIN"
echo "========================================="
echo ""
echo "To run experiments with fine-tuned models:"
echo "python scripts/run_experiment.py \\"
echo "  --config configs/experiments/05-finetune/A10_finetuned_reranker.yaml \\"
echo "  --domain $DOMAIN \\"
echo "  --output experiments/A10_finetuned_reranker/$DOMAIN"
