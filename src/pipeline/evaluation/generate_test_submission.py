#!/usr/bin/env python3
"""
Generate Task A submission for test set using best configurations per domain.

Best configurations:
- clapnq: hybrid_splade_voyage_rewrite_own (Cohere rewrites) 
- govt: hybrid_splade_voyage_rewrite_own (Cohere rewrites)
- fiqa: hybrid_splade_voyage_rewrite (GT rewrites)
- ibmcloud: hybrid_splade_voyage_rewrite (GT rewrites)
"""

import os
os.environ['HF_HOME'] = '/workspace/cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/workspace/cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/cache/transformers'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import json
import sys
from pathlib import Path
from typing import Dict, List
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.retrieval import get_sparse_retriever, get_dense_retriever, HybridRetriever
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Domain to config mapping (best performing configurations)
DOMAIN_CONFIGS = {
    'clapnq': 'configs/experiments/02-hybrid/hybrid_splade_voyage_rewrite_own.yaml',
    'govt': 'configs/experiments/02-hybrid/hybrid_splade_voyage_rewrite_own.yaml',
    'fiqa': 'configs/experiments/02-hybrid/hybrid_splade_voyage_rewrite.yaml',
    'ibmcloud': 'configs/experiments/02-hybrid/hybrid_splade_voyage_rewrite.yaml',
}


def load_test_data(test_file: Path) -> List[Dict]:
    """Load test queries from JSONL file."""
    logger.info(f"Loading test data from {test_file}")
    queries = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line.strip()))
    logger.info(f"Loaded {len(queries)} test queries")
    return queries


def extract_query_text(input_messages: List[Dict]) -> str:
    """Extract the last user message as the query text."""
    for msg in reversed(input_messages):
        if msg.get('speaker') == 'user':
            return msg['text']
    return ""


def load_query_rewrites(domain: str, query_mode: str) -> Dict[str, str]:
    """
    Load query rewrites for a domain.
    
    Args:
        domain: Domain name (clapnq, govt, fiqa, ibmcloud)
        query_mode: 'rewrite' or 'rewrite_own'
        
    Returns:
        Dict mapping task_id to rewritten query
    """
    # Map ibmcloud to cloud
    file_domain = "cloud" if domain == "ibmcloud" else domain
    
    if query_mode == "rewrite":
        query_file = f"data/retrieval_tasks/{file_domain}/{file_domain}_rewrite.jsonl"
    elif query_mode == "rewrite_own":
        query_file = f"data/rewrite_cohere/{file_domain}_command-r-rewrite.txt"
    else:
        return {}
    
    query_map = {}
    try:
        with open(query_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                query_map[item['_id']] = item['text']
        logger.info(f"Loaded {len(query_map)} rewrites for {domain} from {query_file}")
    except Exception as e:
        logger.warning(f"Could not load rewrites for {domain}: {e}")
    
    return query_map


def initialize_retriever(domain: str, config_path: str) -> HybridRetriever:
    """Initialize a hybrid retriever for the given domain."""
    logger.info(f"Initializing retriever for {domain} with config: {config_path}")
    
    # Map ibmcloud to cloud for file paths
    file_domain = "cloud" if domain == "ibmcloud" else domain
    
    # Load config
    config = load_config(config_path)
    
    # Initialize sparse retriever (SPLADE)
    sparse_method = config["retrieval"].get("sparse", {}).get("method", "splade")
    sparse_index = f"indices/{file_domain}/{sparse_method}"
    
    sparse_retriever = get_sparse_retriever(
        model_name=sparse_method,
        index_path=sparse_index,
        config=config["retrieval"].get("sparse", {})
    )
    logger.info(f"  Sparse retriever ({sparse_method}) initialized")
    
    # Initialize dense retriever (Voyage)
    dense_model = config["retrieval"].get("dense", {}).get("model_name", "voyage-3-large")
    
    # Handle Voyage model selection
    if "voyage" in dense_model.lower():
        if file_domain == "fiqa":
            dense_model = "voyage-finance-2"
        else:
            dense_model = "voyage-3-large"
        dense_index = f"indices/{file_domain}/voyage"
    else:
        dense_index = f"indices/{file_domain}/bge"
    
    dense_retriever = get_dense_retriever(
        model_name=dense_model,
        index_path=dense_index,
        config=config["retrieval"].get("dense", {})
    )
    logger.info(f"  Dense retriever ({dense_model}) initialized")
    
    # Create hybrid retriever
    fusion_method = config["retrieval"].get("fusion_method", "rrf")
    fusion_params = {"k": config["retrieval"].get("rrf_k", 60)}
    
    hybrid_retriever = HybridRetriever(
        sparse_retriever=sparse_retriever,
        dense_retriever=dense_retriever,
        fusion_method=fusion_method,
        fusion_params=fusion_params
    )
    logger.info(f"  Hybrid retriever initialized with {fusion_method} fusion")
    
    return hybrid_retriever, config


def generate_predictions(test_queries: List[Dict]) -> List[Dict]:
    """Generate predictions for all test queries."""
    predictions = []
    
    # Group queries by domain
    queries_by_domain = {}
    for query_item in test_queries:
        domain = query_item['Collection']
        if domain not in queries_by_domain:
            queries_by_domain[domain] = []
        queries_by_domain[domain].append(query_item)
    
    logger.info(f"Grouped queries into {len(queries_by_domain)} domains")
    for domain, queries in queries_by_domain.items():
        logger.info(f"  {domain}: {len(queries)} queries")
    
    # Process each domain
    for domain, domain_queries in queries_by_domain.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing domain: {domain.upper()}")
        logger.info(f"{'='*80}")
        
        # Get config for this domain
        config_path = DOMAIN_CONFIGS.get(domain)
        if not config_path:
            logger.error(f"No config found for domain: {domain}")
            for query_item in domain_queries:
                predictions.append({
                    'task_id': query_item['task_id'],
                    'Collection': domain,
                    'contexts': []
                })
            continue
        
        # Initialize retriever
        try:
            retriever, config = initialize_retriever(domain, config_path)
        except Exception as e:
            logger.error(f"Failed to initialize retriever for {domain}: {e}")
            for query_item in domain_queries:
                predictions.append({
                    'task_id': query_item['task_id'],
                    'Collection': domain,
                    'contexts': []
                })
            continue
        
        # Load query rewrites if needed
        query_mode = config["data"].get("query_mode", "last_turn")
        query_rewrites = {}
        if "rewrite" in query_mode:
            query_rewrites = load_query_rewrites(domain, query_mode)
        
        # Process queries for this domain
        for i, query_item in enumerate(domain_queries):
            task_id = query_item['task_id']
            input_messages = query_item['input']
            
            # Get query text
            # Try to use rewrite if available, otherwise fall back to last turn
            if query_rewrites and task_id in query_rewrites:
                query_text = query_rewrites[task_id]
                if (i+1) % 20 == 1:  # Log occasionally
                    logger.info(f"[{i+1}/{len(domain_queries)}] Using rewrite for {task_id[:30]}...")
            else:
                query_text = extract_query_text(input_messages)
                if (i+1) % 20 == 1:  # Log occasionally
                    logger.info(f"[{i+1}/{len(domain_queries)}] Using last turn for {task_id[:30]}...")
            
            if not query_text:
                logger.warning(f"  No query text found!")
                predictions.append({
                    'task_id': task_id,
                    'Collection': domain,
                    'contexts': []
                })
                continue
            
            # Retrieve documents
            try:
                results = retriever.retrieve(query_text, top_k=10)
                
                # Format contexts
                contexts = []
                for result in results:
                    contexts.append({
                        'document_id': result['id'],
                        'score': float(result['score'])
                    })
                
                predictions.append({
                    'task_id': task_id,
                    'Collection': domain,
                    'contexts': contexts
                })
                
                logger.info(f"  → Retrieved {len(contexts)} documents")
                
            except Exception as e:
                logger.error(f"  Error: {e}")
                predictions.append({
                    'task_id': task_id,
                    'Collection': domain,
                    'contexts': []
                })
    
    return predictions


def save_predictions(predictions: List[Dict], output_file: Path):
    """Save predictions to JSONL file with UTF-8 encoding."""
    logger.info(f"Saving {len(predictions)} predictions to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            json.dump(pred, f, ensure_ascii=False)
            f.write('\n')
    
    logger.info(f"Saved predictions to {output_file}")


def main():
    # Change to project root
    project_root = Path(__file__).parent.parent.parent.parent
    os.chdir(project_root)
    
    # Paths
    test_file = Path('src/pipeline/evaluation/rag_taskAC.jsonl')
    output_dir = Path('experiments/final_submission_taskA')
    output_file = output_dir / 'task_a_predictions.jsonl'
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Test file: {test_file}")
    logger.info(f"Output file: {output_file}")
    
    # Load test data
    test_queries = load_test_data(test_file)
    
    # Generate predictions
    predictions = generate_predictions(test_queries)
    
    # Save predictions
    save_predictions(predictions, output_file)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUBMISSION GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output file: {output_file}")
    logger.info(f"Total predictions: {len(predictions)}")
    
    # Count by domain
    from collections import Counter
    domain_counts = Counter(p['Collection'] for p in predictions)
    logger.info(f"\nPredictions by domain:")
    for domain, count in sorted(domain_counts.items()):
        logger.info(f"  {domain}: {count}")
    
    # Count empty predictions
    empty_count = sum(1 for p in predictions if not p['contexts'])
    logger.info(f"\nEmpty predictions: {empty_count}/{len(predictions)}")
    
    logger.info("\nNext steps:")
    logger.info(f"1. Validate format:")
    logger.info(f"   python src/pipeline/evaluation/format_checker.py {output_file} --mode taskA")
    logger.info(f"2. Check sample:")
    logger.info(f"   head -3 {output_file}")


if __name__ == '__main__':
    main()
