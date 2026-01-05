#!/usr/bin/env python3
"""
Fine-tune a cross-encoder reranker on domain-specific data.

Uses sentence-transformers to train a binary classification model
that scores query-document relevance.
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple
import random
import numpy as np

import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def load_training_data(train_file: Path) -> List[InputExample]:
    """Load training data as InputExamples."""
    examples = []
    with open(train_file, 'r') as f:
        for line in f:
            pair = json.loads(line)
            examples.append(InputExample(
                texts=[pair['query'], pair['document']],
                label=float(pair['label'])
            ))
    return examples


def load_eval_data(val_file: Path) -> Tuple[List[List[str]], List[int]]:
    """Load validation data for evaluation."""
    sentence_pairs = []
    labels = []
    
    with open(val_file, 'r') as f:
        for line in f:
            pair = json.loads(line)
            sentence_pairs.append([pair['query'], pair['document']])
            labels.append(int(pair['label']))
    
    return sentence_pairs, labels


def finetune_reranker(
    domain: str,
    training_data_dir: Path,
    base_model: str,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100
):
    """Fine-tune cross-encoder reranker."""
    logger.info(f"Fine-tuning reranker for domain: {domain}")
    logger.info(f"Base model: {base_model}")
    
    # Paths
    train_file = training_data_dir / "train.jsonl"
    val_file = training_data_dir / "val.jsonl"
    
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Validation file not found: {val_file}")
    
    # Load data
    logger.info("Loading training data...")
    train_examples = load_training_data(train_file)
    logger.info(f"Loaded {len(train_examples)} training examples")
    
    logger.info("Loading validation data...")
    val_sentence_pairs, val_labels = load_eval_data(val_file)
    logger.info(f"Loaded {len(val_labels)} validation examples")
    
    # Initialize model
    logger.info("Initializing cross-encoder model...")
    model = CrossEncoder(base_model, num_labels=1, max_length=512)
    
    # Create DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # Create evaluator
    evaluator = CEBinaryClassificationEvaluator(
        sentence_pairs=val_sentence_pairs,
        labels=val_labels,
        name='dev'
    )
    
    # Training
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting training for {epochs} epochs...")
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=500,
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        save_best_model=True,
        optimizer_params={'lr': learning_rate}
    )
    
    logger.info(f"Training complete. Model saved to {output_dir}")
    
    # Final evaluation
    logger.info("Running final evaluation...")
    final_score = evaluator(model, output_path=str(output_dir))
    logger.info(f"Final validation score: {final_score}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune cross-encoder reranker")
    parser.add_argument("--domain", required=True, help="Domain name")
    parser.add_argument("--training_data_dir", type=Path, required=True, help="Training data directory")
    parser.add_argument(
        "--base_model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Base cross-encoder model"
    )
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    
    args = parser.parse_args()
    
    finetune_reranker(
        domain=args.domain,
        training_data_dir=args.training_data_dir,
        base_model=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps
    )


if __name__ == "__main__":
    main()
