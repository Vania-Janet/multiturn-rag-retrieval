#!/usr/bin/env python3
"""
Generate Cohere Fine-tuning Training Data
Creates triplets (query, relevant_passages, hard_negatives) from ground truth + hybrid retrieval results

Strategy:
- Anchor: Rewritten query (ground truth)
- Positive: Ground truth relevant passage
- Hard Negatives: Top-10 from hybrid system that are NOT ground truth (teaches discrimination)

Split: 80% train / 20% validation (stratified by domain)
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set

random.seed(42)

print("=" * 90)
print("  GENERATING COHERE FINE-TUNING DATA")
print("=" * 90)
print()

# Paths
DOMAINS = ["clapnq", "govt", "cloud", "fiqa"]
OUTPUT_DIR = Path("experiments/05-finetune/cohere_rerank_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load corpus for text retrieval
def load_corpus(domain: str) -> Dict[str, str]:
    """Load document corpus"""
    corpus = {}
    corpus_dir = Path(f"data/passage_level_processed/{domain}")
    
    if not corpus_dir.exists():
        print(f"WARNING: Corpus not found for {domain}")
        return {}
    
    for file in corpus_dir.glob("*.jsonl"):
        with open(file) as f:
            for line in f:
                doc = json.loads(line)
                corpus[doc['_id']] = doc.get('text', doc.get('title', ''))
    
    return corpus

# Load ground truth qrels
def load_qrels(domain: str) -> Dict[str, Set[str]]:
    """Load ground truth relevant documents"""
    qrels = defaultdict(set)
    qrels_file = Path(f"data/retrieval_tasks/{domain}/qrels/dev.tsv")
    
    with open(qrels_file) as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                task_id, doc_id, rel = parts[0], parts[1], int(parts[2])
                if rel > 0:  # Only positive relevance
                    qrels[task_id].add(doc_id)
    
    return qrels

# Load hybrid retrieval results
def load_hybrid_results(domain: str) -> Dict[str, List[str]]:
    """Load Top-10 from hybrid baseline"""
    results = {}
    
    # Try both possible hybrid config names
    for exp_name in ["hybrid_splade_voyage_rewrite", "hybrid_splade_bge15_rewrite"]:
        results_file = Path(f"experiments/{exp_name}/{domain}/retrieval_results.jsonl")
        
        if results_file.exists():
            with open(results_file) as f:
                for line in f:
                    result = json.loads(line)
                    task_id = result['task_id']
                    top10_docs = [ctx['document_id'] for ctx in result['contexts'][:10]]
                    results[task_id] = top10_docs
            break
    
    return results

# Generate training examples
def generate_training_examples(domain: str) -> List[Dict]:
    """Generate triplets for one domain"""
    print(f"Processing {domain.upper()}...")
    
    corpus = load_corpus(domain)
    qrels = load_qrels(domain)
    hybrid_results = load_hybrid_results(domain)
    
    # Load rewritten queries
    queries_file = Path(f"data/retrieval_tasks/{domain}/{domain}_rewrite.jsonl")
    queries = {}
    
    with open(queries_file) as f:
        for line in f:
            q = json.loads(line)
            task_id = q.get('task_id') or q.get('_id')
            queries[task_id] = q.get('rewrite') or q.get('text')
    
    training_examples = []
    
    for task_id in qrels.keys():
        if task_id not in queries or task_id not in hybrid_results:
            continue
        
        query_text = queries[task_id]
        relevant_doc_ids = qrels[task_id]
        top10_docs = hybrid_results[task_id]
        
        # Get relevant passage texts
        relevant_passages = []
        for doc_id in relevant_doc_ids:
            if doc_id in corpus:
                text = corpus[doc_id]
                # Truncate if too long (Cohere limit: query + passage < 510 tokens)
                if len(text.split()) > 400:
                    text = ' '.join(text.split()[:400]) + '...'
                relevant_passages.append(text)
        
        if not relevant_passages:
            continue
        
        # Get hard negatives: Top-10 docs that are NOT relevant
        hard_negatives = []
        for doc_id in top10_docs:
            if doc_id not in relevant_doc_ids and doc_id in corpus:
                text = corpus[doc_id]
                if len(text.split()) > 400:
                    text = ' '.join(text.split()[:400]) + '...'
                hard_negatives.append(text)
        
        # Need at least 1 relevant and some hard negatives
        if len(relevant_passages) >= 1 and len(hard_negatives) >= 3:
            # Limit to ~5 hard negatives as recommended
            hard_negatives = hard_negatives[:5]
            
            training_examples.append({
                "query": query_text,
                "relevant_passages": relevant_passages,
                "hard_negatives": hard_negatives,
                "metadata": {
                    "domain": domain,
                    "task_id": task_id,
                    "num_relevant": len(relevant_passages),
                    "num_hard_neg": len(hard_negatives)
                }
            })
    
    print(f"  SUCCESS: {len(training_examples)} training examples created")
    return training_examples

# Stratified split
def stratified_split(examples_by_domain: Dict[str, List], train_ratio: float = 0.8):
    """Split data maintaining domain proportions"""
    train_data = []
    val_data = []
    
    for domain, examples in examples_by_domain.items():
        random.shuffle(examples)
        split_idx = int(len(examples) * train_ratio)
        
        train_data.extend(examples[:split_idx])
        val_data.extend(examples[split_idx:])
        
        print(f"  {domain.upper():8s}: {len(examples):3d} total → {split_idx:3d} train, {len(examples)-split_idx:2d} val")
    
    return train_data, val_data

# Main processing
print("Step 1: Loading data from all domains...")
print()

examples_by_domain = {}
for domain in DOMAINS:
    examples_by_domain[domain] = generate_training_examples(domain)

total_examples = sum(len(ex) for ex in examples_by_domain.values())
print()
print(f"SUCCESS: Total examples: {total_examples}")
print()

# Stratified split
print("Step 2: Creating stratified 80/20 split...")
print()
train_data, val_data = stratified_split(examples_by_domain, train_ratio=0.8)

print()
print(f"SUCCESS: Train set: {len(train_data)} examples")
print(f"SUCCESS: Validation set: {len(val_data)} examples")
print()

# Remove metadata before saving (Cohere doesn't need it)
def clean_for_export(examples: List[Dict]) -> List[Dict]:
    """Remove metadata field for Cohere upload"""
    return [{k: v for k, v in ex.items() if k != 'metadata'} for ex in examples]

# Save to JSONL
print("Step 3: Saving training files...")
print()

train_file = OUTPUT_DIR / "train.jsonl"
val_file = OUTPUT_DIR / "validation.jsonl"

with open(train_file, 'w') as f:
    for example in clean_for_export(train_data):
        f.write(json.dumps(example) + '\n')

with open(val_file, 'w') as f:
    for example in clean_for_export(val_data):
        f.write(json.dumps(example) + '\n')

print(f"SUCCESS: Train data saved to: {train_file}")
print(f"SUCCESS: Validation data saved to: {val_file}")
print()

# Statistics
print("=" * 90)
print("  DATASET STATISTICS")
print("=" * 90)
print()

avg_relevant = sum(len(ex['relevant_passages']) for ex in train_data) / len(train_data)
avg_hard_neg = sum(len(ex['hard_negatives']) for ex in train_data) / len(train_data)

print(f"Training set: {len(train_data)} examples")
print(f"  Avg relevant passages per query: {avg_relevant:.2f}")
print(f"  Avg hard negatives per query: {avg_hard_neg:.2f}")
print()

print(f"Validation set: {len(val_data)} examples")
print()

# Check requirements
print("Cohere Requirements Check:")
print()
if len(train_data) >= 256:
    print(f"  PASS: Training examples: {len(train_data)} >= 256 (minimum)")
else:
    print(f"  WARN: Training examples: {len(train_data)} < 256 (need more!)")

if len(val_data) >= 64:
    print(f"  PASS: Validation examples: {len(val_data)} >= 64 (recommended)")
else:
    print(f"  WARN: Validation examples: {len(val_data)} < 64 (recommended 64+)")

print(f"  PASS: Avg hard negatives: {avg_hard_neg:.2f} (recommended ~5)")
print(f"  PASS: Split is stratified by domain")
print()

# Show sample
print("=" * 90)
print("  SAMPLE TRAINING EXAMPLE")
print("=" * 90)
print()

sample = train_data[0]
print(f"Query: {sample['query'][:100]}...")
print(f"Relevant passages: {len(sample['relevant_passages'])}")
print(f"  → {sample['relevant_passages'][0][:120]}...")
print(f"Hard negatives: {len(sample['hard_negatives'])}")
print(f"  → {sample['hard_negatives'][0][:120]}...")
print()

print("=" * 90)
print("SUCCESS: Data generation complete!")
print()
print("Next steps:")
print("1. Review the generated files:")
print(f"   - {train_file}")
print(f"   - {val_file}")
print("2. Upload to Cohere for fine-tuning")
print("3. Wait for training to complete (~hours)")
print("4. Update experiment configs with fine-tuned model ID")
print("=" * 90)
