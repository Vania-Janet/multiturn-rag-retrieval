#!/bin/bash

# Fine-tune BGE-reranker-v2-m3 on benchmark data
# Uses FlagEmbedding's training script

set -e

echo "========================================================================"
echo "  BGE-RERANKER-V2-M3 FINE-TUNING"
echo "========================================================================"
echo ""

# Configuration
BASE_MODEL="BAAI/bge-reranker-v2-m3"
OUTPUT_DIR="models/bge-reranker-v2-m3-finetuned"
TRAIN_FILE="experiments/05-finetune/bge_rerank_data/train.tsv"
VAL_FILE="experiments/05-finetune/bge_rerank_data/val.tsv"

# Training hyperparameters
BATCH_SIZE=8  # Adjust based on GPU memory
LEARNING_RATE=2e-5
NUM_EPOCHS=3
MAX_LENGTH=512
WARMUP_STEPS=100

# Check if data exists
if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: Training data not found: $TRAIN_FILE"
    echo "Run: python3 scripts/generate_bge_finetune_data.py first"
    exit 1
fi

echo "Training Configuration:"
echo "  Base Model: $BASE_MODEL"
echo "  Output: $OUTPUT_DIR"
echo "  Train Data: $TRAIN_FILE ($(wc -l < $TRAIN_FILE) examples)"
echo "  Val Data: $VAL_FILE ($(wc -l < $VAL_FILE) examples)"
echo "  Batch Size: $BATCH_SIZE per GPU"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Epochs: $NUM_EPOCHS"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check GPU availability
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "Available GPUs: $GPU_COUNT"
echo ""

# Create training script for FlagEmbedding
cat > /tmp/train_bge_reranker.py << 'EOF'
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_callback import TrainerCallback
import torch
from torch.utils.data import Dataset

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model"})
    max_length: int = field(default=512, metadata={"help": "Max sequence length"})

@dataclass
class DataArguments:
    train_data: str = field(metadata={"help": "Path to training data (TSV)"})
    eval_data: Optional[str] = field(default=None, metadata={"help": "Path to eval data"})

class RerankDataset(Dataset):
    """Dataset for reranker training from TSV format"""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load TSV: query \t positive \t negative
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    query, pos, neg = parts
                    self.data.append({
                        'query': query,
                        'positive': pos,
                        'negative': neg
                    })
        
        print(f"Loaded {len(self.data)} training examples from {data_file}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize positive pair
        pos_inputs = self.tokenizer(
            item['query'],
            item['positive'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Tokenize negative pair
        neg_inputs = self.tokenizer(
            item['query'],
            item['negative'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'pos_input_ids': pos_inputs['input_ids'].squeeze(),
            'pos_attention_mask': pos_inputs['attention_mask'].squeeze(),
            'neg_input_ids': neg_inputs['input_ids'].squeeze(),
            'neg_attention_mask': neg_inputs['attention_mask'].squeeze(),
        }

class RerankTrainer(Trainer):
    """Custom trainer for reranker with pairwise loss"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass for positive pairs
        pos_outputs = model(
            input_ids=inputs['pos_input_ids'],
            attention_mask=inputs['pos_attention_mask']
        )
        pos_scores = pos_outputs.logits
        
        # Forward pass for negative pairs
        neg_outputs = model(
            input_ids=inputs['neg_input_ids'],
            attention_mask=inputs['neg_attention_mask']
        )
        neg_scores = neg_outputs.logits
        
        # Pairwise margin loss: positive should score higher than negative
        margin = 0.1
        loss = torch.nn.functional.relu(margin - (pos_scores - neg_scores)).mean()
        
        return (loss, pos_outputs) if return_outputs else loss

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Load model and tokenizer
    print(f"Loading model: {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=1,  # Regression for relevance score
        torch_dtype=torch.float16 if training_args.fp16 else torch.float32
    )
    
    # Load datasets
    train_dataset = RerankDataset(
        data_args.train_data,
        tokenizer,
        model_args.max_length
    )
    
    eval_dataset = None
    if data_args.eval_data:
        eval_dataset = RerankDataset(
            data_args.eval_data,
            tokenizer,
            model_args.max_length
        )
    
    # Create trainer
    trainer = RerankTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {training_args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    print("Training complete!")

if __name__ == "__main__":
    main()
EOF

echo "Starting fine-tuning..."
echo ""

# Run training with both GPUs
CUDA_VISIBLE_DEVICES=0,1 python3 /tmp/train_bge_reranker.py \
    --model_name_or_path "$BASE_MODEL" \
    --max_length $MAX_LENGTH \
    --train_data "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate $LEARNING_RATE \
    --per_device_train_batch_size $BATCH_SIZE \
    --num_train_epochs $NUM_EPOCHS \
    --warmup_steps $WARMUP_STEPS \
    --save_strategy epoch \
    --save_total_limit 2 \
    --logging_steps 10 \
    --fp16 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 2 \
    --report_to none

echo ""
echo "========================================================================"
echo "  TRAINING COMPLETE"
echo "========================================================================"
echo ""
echo "Fine-tuned model saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Update experiment configs with model path:"
echo "     sed -i 's|BAAI/bge-reranker-v2-m3|$OUTPUT_DIR|g' \\"
echo "       configs/experiments/05-finetune/*.yaml"
echo ""
echo "  2. Run experiments:"
echo "     bash scripts/run_bge_finetuned_experiments.sh"
echo ""
echo "========================================================================"
