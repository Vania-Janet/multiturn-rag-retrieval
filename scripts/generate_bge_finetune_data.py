#!/usr/bin/env python3
"""
Generate fine-tuning data for BGE-reranker-v2-m3

BGE reranker expects TSV format:
query \t positive_passage \t negative_passage

We'll create multiple training examples per query using:
- Each relevant passage as positive
- Hard negatives from Top-10 that aren't relevant
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Configuration
DOMAINS = ["clapnq", "govt", "cloud", "fiqa"]
DATA_DIR = Path("data/retrieval_tasks")
CORPUS_DIR = Path("data/passage_level_processed")
HYBRID_RESULTS_DIR = Path("experiments")

# Output
OUTPUT_DIR = Path("experiments/05-finetune/bge_rerank_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_corpus(domain: str) -> Dict[str, str]:
    """Load corpus for a domain"""
    corpus = {}
    corpus_dir = CORPUS_DIR / domain
    
    if not corpus_dir.exists():
        print(f"WARNING: Corpus not found for {domain}")
        return {}
    
    for file in corpus_dir.glob("*.jsonl"):
        with open(file) as f:
            for line in f:
                doc = json.loads(line)
                corpus[doc['id']] = doc['text']
    
    return corpus

def load_qrels(domain: str) -> Dict[str, List[str]]:
    """Load ground truth relevant documents from TSV format"""
    qrels_file = f"data/retrieval_tasks/{domain}/qrels/dev.tsv"
    qrels = defaultdict(list)
    
    if not Path(qrels_file).exists():
        print(f"WARNING: Qrels not found at {qrels_file}")
        return qrels
    
    import csv
    with open(qrels_file) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 3:
                query_id, corpus_id, score = row[0], row[1], int(row[2])
                if score > 0:  # Only relevant docs
                    qrels[query_id].append(corpus_id)
    
    return qrels

def load_queries(domain: str) -> Dict[str, str]:
    """Load rewritten queries"""
    query_file = DATA_DIR / domain / f"{domain}_rewrite.jsonl"
    queries = {}
    
    if not query_file.exists():
        print(f"WARNING: Query file not found for {domain}")
        return queries
    
    with open(query_file) as f:
        for line in f:
            item = json.loads(line)
            # Use _id as query_id and text as query
            queries[item['_id']] = item['text']
    
    return queries

def load_hybrid_results(domain: str) -> Dict[str, List[str]]:
    """Load hybrid baseline Top-10 results for hard negatives"""
    # Try to find the best hybrid baseline
    possible_paths = [
        f"experiments/hybrid_splade_voyage_rewrite/{domain}/retrieval_results.jsonl",
        f"experiments/hybrid_splade_bge15_rewrite/{domain}/retrieval_results.jsonl",
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            results = {}
            with open(path) as f:
                for line in f:
                    item = json.loads(line)
                    # Get Top-10 doc IDs (use document_id key)
                    top_ids = [ctx['document_id'] for ctx in item['contexts'][:10]]
                    results[item['task_id']] = top_ids
            return results
    
    print(f"WARNING: No hybrid results found for {domain}")
    return {}

def generate_training_pairs(
    domain: str,
    queries: Dict[str, str],
    corpus: Dict[str, str],
    qrels: Dict[str, List[str]],
    hybrid_results: Dict[str, List[str]]
) -> List[Tuple[str, str, str]]:
    """
    Generate (query, positive, negative) triplets
    
    For each query:
    - Use each relevant doc as positive
    - Sample hard negatives from Top-10 that aren't relevant
    """
    triplets = []
    
    for query_id, query_text in queries.items():
        relevant_ids = qrels.get(query_id, [])
        if not relevant_ids:
            continue
        
        # Get hard negatives: Top-10 docs that aren't relevant
        top_10 = hybrid_results.get(query_id, [])
        hard_negatives = [doc_id for doc_id in top_10 if doc_id not in relevant_ids]
        
        if not hard_negatives:
            continue
        
        # Create multiple training examples
        for pos_id in relevant_ids[:3]:  # Limit to 3 relevant docs per query
            pos_text = corpus.get(pos_id, "")
            if not pos_text:
                continue
            
            # Sample 2 hard negatives per positive
            num_negs = min(2, len(hard_negatives))
            sampled_negs = random.sample(hard_negatives, num_negs)
            
            for neg_id in sampled_negs:
                neg_text = corpus.get(neg_id, "")
                if not neg_text:
                    continue
                
                # Truncate to reasonable length (BGE handles up to 512 tokens)
                pos_text_trunc = ' '.join(pos_text.split()[:400])
                neg_text_trunc = ' '.join(neg_text.split()[:400])
                
                triplets.append((query_text, pos_text_trunc, neg_text_trunc))
    
    return triplets

def stratified_split(
    triplets_by_domain: Dict[str, List[Tuple[str, str, str]]],
    train_ratio: float = 0.8
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """Split into train/val while maintaining domain proportions"""
    train_data = []
    val_data = []
    
    random.seed(42)
    
    for domain, triplets in triplets_by_domain.items():
        random.shuffle(triplets)
        split_idx = int(len(triplets) * train_ratio)
        
        train_data.extend(triplets[:split_idx])
        val_data.extend(triplets[split_idx:])
        
        print(f"  {domain.upper()}: {len(triplets)} triplets -> {split_idx} train, {len(triplets) - split_idx} val")
    
    # Final shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    return train_data, val_data

# Main execution
print("=" * 90)
print("  GENERATING BGE-RERANKER FINE-TUNING DATA")
print("=" * 90)
print()

print("Step 1: Loading data from all domains...")
print()

triplets_by_domain = {}

for domain in DOMAINS:
    print(f"Processing {domain.upper()}...")
    
    corpus = load_corpus(domain)
    qrels = load_qrels(domain)
    queries = load_queries(domain)
    hybrid_results = load_hybrid_results(domain)
    
    if not all([corpus, qrels, queries, hybrid_results]):
        print(f"  WARNING: Missing data for {domain}, skipping...")
        continue
    
    triplets = generate_training_pairs(domain, queries, corpus, qrels, hybrid_results)
    triplets_by_domain[domain] = triplets
    
    print(f"  SUCCESS: {len(triplets)} training triplets created")

total_triplets = sum(len(t) for t in triplets_by_domain.values())
print()
print(f"SUCCESS: Total triplets: {total_triplets}")
print()

# Stratified split
print("Step 2: Creating stratified 80/20 split...")
print()

train_data, val_data = stratified_split(triplets_by_domain, train_ratio=0.8)

print()
print(f"SUCCESS: Train set: {len(train_data)} triplets")
print(f"SUCCESS: Validation set: {len(val_data)} triplets")
print()

# Save to TSV format (query \t positive \t negative)
print("Step 3: Saving training files...")
print()

train_file = OUTPUT_DIR / "train.tsv"
val_file = OUTPUT_DIR / "val.tsv"

with open(train_file, 'w', encoding='utf-8') as f:
    for query, pos, neg in train_data:
        f.write(f"{query}\t{pos}\t{neg}\n")

with open(val_file, 'w', encoding='utf-8') as f:
    for query, pos, neg in val_data:
        f.write(f"{query}\t{pos}\t{neg}\n")

print(f"SUCCESS: Train data saved to: {train_file}")
print(f"SUCCESS: Validation data saved to: {val_file}")
print()

# Statistics
print("BGE Fine-tuning Requirements Check:")
print()
if len(train_data) >= 100:
    print(f"  PASS: Training triplets: {len(train_data)} >= 100 (recommended)")
else:
    print(f"  WARN: Training triplets: {len(train_data)} < 100 (need more)")

if len(val_data) >= 20:
    print(f"  PASS: Validation triplets: {len(val_data)} >= 20 (recommended)")
else:
    print(f"  WARN: Validation triplets: {len(val_data)} < 20 (need more)")

print(f"  PASS: Format: TSV (query \\t positive \\t negative)")
print(f"  PASS: Text truncated to ~400 words (~512 tokens)")
print()

print("=" * 90)
print("SUCCESS: Data generation complete!")
print()
print("Next steps:")
print("  1. Run fine-tuning: bash scripts/train_bge_reranker.sh")
print("  2. Wait ~2-4 hours for training (depends on GPU)")
print("  3. Update configs with fine-tuned model path")
print("  4. Run experiments with fine-tuned model")
print()
print("=" * 90)
