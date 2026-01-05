#!/usr/bin/env python3
"""
Prepare training data for fine-tuning from retrieval tasks and qrels.

Generates positive and HARD negative query-document pairs for training rerankers.
Hard negatives are documents retrieved by BM25 but not relevant according to qrels.
"""

import json
import random
import argparse
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict

try:
    from rank_bm25 import BM25Okapi
    from nltk.tokenize import word_tokenize
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank_bm25 not available. Install with: pip install rank-bm25")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random.seed(42)


def load_corpus(corpus_path: Path) -> Dict[str, str]:
    """Load corpus documents."""
    corpus = {}
    with open(corpus_path, 'r') as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc['id']] = doc['text']
    return corpus


def load_qrels(qrels_path: Path) -> Dict[str, Set[str]]:
    """Load qrels (query -> set of relevant doc IDs)."""
    qrels = defaultdict(set)
    with open(qrels_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                query_id, doc_id, score = parts[0], parts[1], parts[2]
                if int(score) > 0:
                    qrels[query_id].add(doc_id)
    return dict(qrels)


def load_queries(tasks_path: Path, context_turns: int = 3) -> Dict[str, str]:
    """
    Load queries from tasks file with multi-turn context.
    
    Args:
        tasks_path: Path to tasks file
        context_turns: Number of previous turns to include for context
        
    Returns:
        Dict mapping task_id to standalone query text
    """
    queries = {}
    with open(tasks_path, 'r') as f:
    load_bm25_index(index_path: Path) -> Tuple[BM25Okapi, List[str]]:
    """Load pre-built BM25 index if available."""
    model_path = index_path / "index.pkl"
    ids_path = index_path / "doc_ids.json"
    
    if not model_path.exists() or not ids_path.exists():
        return None, None
    
    with open(model_path, 'rb') as f:
        bm25 = pickle.load(f)
    
    with open(ids_path, 'r') as f:
        doc_ids = json.load(f)
    
    return bm25, doc_ids


def build_bm25_on_the_fly(corpus: Dict[str, str]) -> Tuple[BM25Okapi, List[str]]:
    """Build BM25 index on the fly from corpus."""
    if not BM25_AVAILABLE:
        return None, None
    
    logger.info("Building BM25 index on the fly...")
    doc_ids = list(corpus.keys())
    tokenized_corpus = [word_tokenize(corpus[doc_id].lower()) for doc_id in doc_ids]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, doc_ids


def get_hard_negatives(
    query_text: str,
    positive_ids: Set[str],
    bm25: BM25Okapi,
    doc_ids: List[str],
    num_negatives: int = 5,
    top_k_retrieve: int = 50
) -> List[str]:
    """
    Get hard negatives using BM25 retrieval.
    
    Hard negatives are documents that BM25 considers relevant but are not in qrels.
    This forces the reranker to learn fine-grained distinctions.
    
    Args:
        query_text: Query string
        positive_ids: Set10,
    use_hard_negatives: bool = True,
    context_turns: int = 3
):
    """
    Prepare training data for a domain with hard negatives.
    
    Args:
        domain: Domain name
        data_dir: Base data directory
        output_dir: Output directory for training files
        num_negatives: Number of negative samples per positive (default: 10 for 1:10 ratio)
        use_hard_negatives: Use BM25-retrieved hard negatives (default: True)
        context_turns: Number of conversation turns to include for context
    """
    logger.info(f"Preparing training data for domain: {domain}")
    logger.info(f"Hard negatives: {'enabled' if use_hard_negatives else 'disabled'}")
    logger.info(f"Negative ratio: 1:{num_negatives}")
    
    # Paths
    corpus_path = data_dir / "passage_level_processed" / domain / "corpus.jsonl"
    tasks_path = data_dir / "retrieval_tasks" / domain / f"{domain}_tasks.jsonl"
    qrels_path = data_dir / "retrieval_tasks" / domain / "qrels" / "dev.tsv"
    bm25_index_path = data_dir.parent / "indices" / domain / "bm25"
    
    # Load data
    logger.info("Loading corpus...")
    corpus = load_corpus(corpus_path)
    all_doc_ids = list(corpus.keys())
    
    logger.info("Loading queries with multi-turn context...")
    queries = load_queries(tasks_path, context_turns=context_turns)
    
    logger.info("Loading qrels...")
    qrels = load_qrels(qrels_path)
    
    # Load or build BM25 for hard negatives
    bm25, bm25_doc_ids = None, None
    if use_hard_negatives:
        logger.info("Loading BM25 index for hard negative mining...")
        bm25, bm25_doc_ids = load_bm25_index(bm25_index_path)
        
        if bm25 is None:
            logger.warning(f"BM25 index not found at {bm25_index_path}")
            if BM25_AVAILABLE:
                bm25, bm25_doc_ids = build_bm25_on_the_fly(corpus)
            else:
                logger.warning("Falling back to random negatives (install rank-bm25 for hard negatives)")
    
    # Generate training pairs
    logger.info("Generating training pairs with hard negatives...")
    training_pairs = []
    
    for query_id, relevant_docs in qrels.items():
        if query_id not in queries:
            continue
            
        query_text = queries[query_id]
        
        # Add positive pairs
        for doc_id in relevant_docs:
            if doc_id in corpus:
                training_pairs.append({
                    "query": query_text,
                    "document": corpus[doc_id],
                    "label": 1
                })
        
        # Add HARD negative pairs
        if use_hard_negatives and bm25 is not None:
            negative_ids = get_hard_negatives(
                query_text, 
                relevant_docs, 
                bm25, 
                bm25_doc_ids, 
                num_negatives=num_negatives,
                top_k_retrieve=100
            )
        else:
            # Fallback to random negatives
            negative_pool = [doc_id for doc_id in all_doc_ids if doc_id not in relevant_docs]
            num_to_sample = min(num_negatives, len(negative_pool))
            negative_ids = random.sample(negative_pool, num_to_sample)
        
                # Single turn - use as is with hard negatives")
    parser.add_argument("--domain", required=True, help="Domain name (clapnq, fiqa, govt, cloud)")
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="Data directory")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for training data")
    parser.add_argument("--num_negatives", type=int, default=10, help="Number of negative samples per query (default: 10 for 1:10 ratio)")
    parser.add_argument("--no_hard_negatives", action="store_true", help="Disable hard negative mining (use random negatives)")
    parser.add_argument("--context_turns", type=int, default=3, help="Number of conversation turns to include for context")
    
    args = parser.parse_args()
    
    prepare_training_data(
        domain=args.domain,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_negatives=args.num_negatives,
        use_hard_negatives=not args.no_hard_negatives,
        context_turns=args.context_turn
            
            queries[task_id] = query
    return queries


def sample_negatives(
    query_id: str,
    positive_ids: Set[str],
    all_doc_ids: List[str],
    num_negatives: int = 5
) -> List[str]:
    """Sample random negative documents."""
    negative_pool = [doc_id for doc_id in all_doc_ids if doc_id not in positive_ids]
    num_to_sample = min(num_negatives, len(negative_pool))
    return random.sample(negative_pool, num_to_sample)


def prepare_training_data(
    domain: str,
    data_dir: Path,
    output_dir: Path,
    num_negatives: int = 5
):
    """Prepare training data for a domain."""
    logger.info(f"Preparing training data for domain: {domain}")
    
    # Paths
    corpus_path = data_dir / "passage_level_processed" / domain / "corpus.jsonl"
    tasks_path = data_dir / "retrieval_tasks" / domain / f"{domain}_tasks.jsonl"
    qrels_path = data_dir / "retrieval_tasks" / domain / "qrels" / "dev.tsv"
    
    # Load data
    logger.info("Loading corpus...")
    corpus = load_corpus(corpus_path)
    all_doc_ids = list(corpus.keys())
    
    logger.info("Loading queries...")
    queries = load_queries(tasks_path)
    
    logger.info("Loading qrels...")
    qrels = load_qrels(qrels_path)
    
    # Generate training pairs
    logger.info("Generating training pairs...")
    training_pairs = []
    
    for query_id, relevant_docs in qrels.items():
        if query_id not in queries:
            continue
            
        query_text = queries[query_id]
        
        # Add positive pairs
        for doc_id in relevant_docs:
            if doc_id in corpus:
                training_pairs.append({
                    "query": query_text,
                    "document": corpus[doc_id],
                    "label": 1
                })
        
        # Add negative pairs
        negative_ids = sample_negatives(query_id, relevant_docs, all_doc_ids, num_negatives)
        for doc_id in negative_ids:
            if doc_id in corpus:
                training_pairs.append({
                    "query": query_text,
                    "document": corpus[doc_id],
                    "label": 0
                })
    
    # Shuffle
    random.shuffle(training_pairs)
    
    # Split train/val (90/10)
    split_idx = int(len(training_pairs) * 0.9)
    train_pairs = training_pairs[:split_idx]
    val_pairs = training_pairs[split_idx:]
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = output_dir / "train.jsonl"
    val_file = output_dir / "val.jsonl"
    
    logger.info(f"Saving {len(train_pairs)} training pairs to {train_file}")
    with open(train_file, 'w') as f:
        for pair in train_pairs:
            f.write(json.dumps(pair) + '\n')
    
    logger.info(f"Saving {len(val_pairs)} validation pairs to {val_file}")
    with open(val_file, 'w') as f:
        for pair in val_pairs:
            f.write(json.dumps(pair) + '\n')
    
    # Stats
    train_pos = sum(1 for p in train_pairs if p['label'] == 1)
    val_pos = sum(1 for p in val_pairs if p['label'] == 1)
    
    logger.info(f"\nDataset statistics:")
    logger.info(f"  Training: {len(train_pairs)} pairs ({train_pos} positive, {len(train_pairs)-train_pos} negative)")
    logger.info(f"  Validation: {len(val_pairs)} pairs ({val_pos} positive, {len(val_pairs)-val_pos} negative)")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for fine-tuning")
    parser.add_argument("--domain", required=True, help="Domain name (clapnq, fiqa, govt, cloud)")
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="Data directory")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for training data")
    parser.add_argument("--num_negatives", type=int, default=5, help="Number of negative samples per query")
    
    args = parser.parse_args()
    
    prepare_training_data(
        domain=args.domain,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_negatives=args.num_negatives
    )


if __name__ == "__main__":
    main()
