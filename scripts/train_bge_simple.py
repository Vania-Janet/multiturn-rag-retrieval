#!/usr/bin/env python3
"""
Simple BGE reranker fine-tuning using sentence-transformers

This is a simpler approach than using custom Trainer
"""

import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from torch.utils.data import DataLoader

def load_triplets(tsv_file):
    """Load query-positive-negative triplets from TSV"""
    examples = []
    with open(tsv_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                query, pos, neg = parts
                # Create positive example (query, pos_doc, label=1)
                examples.append(InputExample(texts=[query, pos], label=1.0))
                # Create negative example (query, neg_doc, label=0)
                examples.append(InputExample(texts=[query, neg], label=0.0))
    
    return examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='BAAI/bge-reranker-v2-m3')
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--val_file', type=str, default=None)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=512)
    args = parser.parse_args()
    
    print("=" * 80)
    print("  BGE RERANKER FINE-TUNING (Simple Method)")
    print("=" * 80)
    print()
    print(f"Model: {args.model_name}")
    print(f"Train: {args.train_file}")
    print(f"Output: {args.output_dir}")
    print(f"Batch: {args.batch_size}, Epochs: {args.epochs}, LR: {args.lr}")
    print()
    
    # Load model
    print("Loading model...")
    model = CrossEncoder(args.model_name, num_labels=1, max_length=args.max_length)
    
    # Load training data
    print("Loading training data...")
    train_examples = load_triplets(args.train_file)
    print(f"  Loaded {len(train_examples)} training examples")
    
    # Load validation data if provided
    evaluator = None
    if args.val_file and Path(args.val_file).exists():
        print("Loading validation data...")
        val_examples = load_triplets(args.val_file)
        print(f"  Loaded {len(val_examples)} validation examples")
        
        # Create evaluator
        val_samples = {}
        for ex in val_examples:
            query = ex.texts[0]
            doc = ex.texts[1]
            score = ex.label
            
            if query not in val_samples:
                val_samples[query] = {'query': query, 'positive': [], 'negative': []}
            
            if score > 0.5:
                val_samples[query]['positive'].append(doc)
            else:
                val_samples[query]['negative'].append(doc)
        
        evaluator = CERerankingEvaluator(val_samples, name='validation')
    
    # Create data loader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    
    # Train
    print()
    print("Starting training...")
    print()
    
    warmup_steps = int(len(train_dataloader) * args.epochs * 0.1)
    
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=args.epochs,
        optimizer_params={'lr': args.lr},
        warmup_steps=warmup_steps,
        output_path=args.output_dir,
        save_best_model=True,
        show_progress_bar=True,
    )
    
    # Explicitly save the model
    print()
    print("Saving final model...")
    model.save(args.output_dir)
    print(f"Model saved to: {args.output_dir}")
    
    print()
    print("=" * 80)
    print("  TRAINING COMPLETE!")
    print("=" * 80)
    print()
    print(f"Model files:")
    import os
    for f in os.listdir(args.output_dir):
        print(f"  - {f}")
    print()

if __name__ == "__main__":
    main()
