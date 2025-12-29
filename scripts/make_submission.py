#!/usr/bin/env python3
"""
Create submission file for MT-RAG Task A evaluation.

Generates properly formatted JSONL submission from experiment results.
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create submission file for Task A"
    )
    
    parser.add_argument(
        "--experiment", "-e",
        required=True,
        help="Experiment name (e.g., A6_hybrid_rerank)"
    )
    
    parser.add_argument(
        "--domain", "-d",
        required=True,
        choices=["clapnq", "fiqa", "govt", "cloud"],
        help="Domain name"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output submission file path (default: submissions/{experiment}_{domain}.jsonl)"
    )
    
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("experiments"),
        help="Experiments directory (default: experiments/)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of documents to include per query (default: 100)"
    )
    
    parser.add_argument(
        "--run-name",
        help="Run name for submission (default: {experiment}_{domain})"
    )
    
    return parser.parse_args()


def load_retrieval_results(
    experiment_dir: Path,
    domain: str
) -> List[Dict[str, Any]]:
    """Load retrieval results from experiment directory."""
    
    # Check for reranking results first, fall back to retrieval
    rerank_file = experiment_dir / domain / "reranking" / "run.trec"
    retrieval_file = experiment_dir / domain / "retrieval" / "run.trec"
    
    results_file = rerank_file if rerank_file.exists() else retrieval_file
    
    if not results_file.exists():
        raise FileNotFoundError(f"No results found at {results_file}")
    
    # TODO: Parse TREC format
    # Format: query_id Q0 doc_id rank score run_name
    results = []
    
    return results


def create_submission_record(
    task_id: str,
    retrieved_docs: List[Dict[str, Any]],
    run_name: str
) -> Dict[str, Any]:
    """
    Create a submission record in Task A format.
    
    Format:
    {
        "task_id": "conv_id::turn",
        "run_name": "team_method",
        "retrieved_docs": [
            {"document_id": "doc_123", "score": 0.95},
            ...
        ]
    }
    """
    return {
        "task_id": task_id,
        "run_name": run_name,
        "retrieved_docs": retrieved_docs
    }


def main():
    args = parse_args()
    
    logger = setup_logger("make_submission")
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_dir = Path("submissions")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{args.experiment}_{args.domain}.jsonl"
    
    # Determine run name
    run_name = args.run_name or f"{args.experiment}_{args.domain}"
    
    logger.info(f"Creating submission for {args.experiment} on {args.domain}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Run name: {run_name}")
    
    # Load experiment results
    experiment_dir = args.input_dir / args.experiment
    
    if not experiment_dir.exists():
        logger.error(f"Experiment directory not found: {experiment_dir}")
        sys.exit(1)
    
    logger.info(f"Loading results from {experiment_dir}")
    
    # Load retrieval results
    try:
        results = load_retrieval_results(experiment_dir, args.domain)
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        sys.exit(1)
    
    # Group by task_id and create submission records
    from collections import defaultdict
    results_by_task = defaultdict(list)
    
    for result in results:
        task_id = result["task_id"]
        results_by_task[task_id].append({
            "document_id": result["doc_id"],
            "score": result["score"]
        })
    
    # Create submission records
    submission_records = []
    
    for task_id, docs in results_by_task.items():
        # Sort by score and take top-k
        docs_sorted = sorted(docs, key=lambda x: x["score"], reverse=True)
        docs_topk = docs_sorted[:args.top_k]
        
        record = create_submission_record(task_id, docs_topk, run_name)
        submission_records.append(record)
    
    # Write submission file
    logger.info(f"Writing {len(submission_records)} records to {output_path}")
    
    with open(output_path, "w") as f:
        for record in submission_records:
            f.write(json.dumps(record) + "\n")
    
    logger.info("✓ Submission file created successfully")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("SUBMISSION SUMMARY")
    logger.info("="*80)
    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Domain: {args.domain}")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Total queries: {len(submission_records)}")
    logger.info(f"Docs per query: up to {args.top_k}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1024:.2f} KB")
    logger.info("="*80)
    
    logger.info("\n✓ Next step: Validate submission")
    logger.info(f"  python scripts/validate_submission.py {output_path}")


if __name__ == "__main__":
    main()
