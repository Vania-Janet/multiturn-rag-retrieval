import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

from .retrieval import (
    get_sparse_retriever,
    get_dense_retriever,
    HybridRetriever,
    set_seed,
    LatencyMonitor,
    analyze_hard_failures,
    analyze_performance_by_turn,
    analyze_query_variance,
    calculate_wilcoxon_significance,
    bootstrap_confidence_interval,
    apply_bonferroni_correction
)
from .evaluation.run_retrieval_eval import compute_results, load_qrels

logger = logging.getLogger(__name__)

def load_queries(query_file: str) -> List[Dict[str, Any]]:
    """Load queries from a JSONL file."""
    queries = []
    with open(query_file, 'r') as f:
        for line in f:
            queries.append(json.loads(line))
    return queries

def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save retrieval results to a JSONL file."""
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

def load_corpus(corpus_path: Union[str, Path]) -> Dict[str, str]:
    """Load corpus into memory (ID -> Text)."""
    corpus = {}
    logger.info(f"Loading corpus from {corpus_path}")
    path_obj = Path(corpus_path)
    files = []
    if path_obj.is_dir():
        files = list(path_obj.glob("*.jsonl"))
    elif path_obj.suffix == '.jsonl':
        files = [path_obj]
        
    for file_path in files:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    doc_id = doc.get("_id") or doc.get("id")
                    text = doc.get("text", "")
                    if doc_id:
                        corpus[doc_id] = text
                except json.JSONDecodeError:
                    continue
    logger.info(f"Loaded {len(corpus)} documents")
    return corpus

def run_pipeline(config: Dict[str, Any], output_dir: Path, domain: str):
    """
    Run the full retrieval pipeline:
    1. Setup & Seeding
    2. Indexing (if needed - usually pre-built)
    3. Retrieval
    4. Evaluation
    5. Analysis & Stats
    """
    
    # 1. Setup & Seeding
    seed = config.get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed set to {seed}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Initialize Retriever
    retrieval_type = config.get("retrieval", {}).get("type", "sparse")
    logger.info(f"Initializing {retrieval_type} retriever for domain {domain}")
    
    if retrieval_type == "sparse":
        method = config["retrieval"].get("method", "bm25")
        # Determine index path based on method
        if method == "bm25":
            default_index_path = f"indices/{domain}/bm25"
        else:
            default_index_path = f"indices/{domain}/elser" # Placeholder, ELSER uses ES
            
        index_path = config["retrieval"].get("index_path", default_index_path)
        
        retriever = get_sparse_retriever(
            model_name=method,
            index_path=index_path,
            config=config["retrieval"]
        )
    elif retrieval_type == "dense":
        model_name = config["retrieval"].get("model_name", "BAAI/bge-large-en-v1.5")
        
        # Determine default index path based on model
        if "bge-m3" in model_name.lower():
            default_index_path = f"indices/{domain}/bge-m3"
        elif "bge" in model_name.lower():
            default_index_path = f"indices/{domain}/bge"
        else:
            default_index_path = f"indices/{domain}/dense"
            
        index_path = config["retrieval"].get("index_path", default_index_path)
        
        retriever = get_dense_retriever(
            model_name=model_name,
            index_path=index_path,
            config=config["retrieval"]
        )
    elif retrieval_type == "hybrid":
        # Example hybrid setup - would need more specific config handling in a real scenario
        sparse = get_sparse_retriever(
            method="bm25", 
            index_name=f"{domain}_sparse",
            config=config["retrieval"].get("sparse", {})
        )
        dense = get_dense_retriever(
            model_name="BAAI/bge-large-en-v1.5",
            index_path=f"indices/{domain}/dense",
            config=config["retrieval"].get("dense", {})
        )
        retriever = HybridRetriever(sparse, dense, alpha=config["retrieval"].get("alpha", 0.5))
    else:
        raise ValueError(f"Unknown retrieval type: {retrieval_type}")

    # 3. Load Data
    query_file = config["data"]["query_file"]
    qrels_file = config["data"]["qrels_file"]
    
    # Load Corpus
    corpus_path = config["data"].get("corpus_path")
    corpus = {}
    if corpus_path:
        corpus = load_corpus(corpus_path)
    else:
        logger.warning("No corpus path provided. Results will not contain text.")
    
    logger.info(f"Loading queries from {query_file}")
    queries = load_queries(query_file)
    
    # 4. Retrieval Loop with Latency Monitoring
    latency_monitor = LatencyMonitor()
    results = []
    
    logger.info(f"Starting retrieval for {len(queries)} queries...")
    
    query_mode = config["data"].get("query_mode", "last_turn")
    logger.info(f"Using query extraction mode: {query_mode}")

    for query in queries:
        # Support both BEIR format (_id + text) and full history format (input list)
        query_text = ""
        query_id = query.get("task_id") or query.get("_id")
        turn_id = query.get("turn", 0)
        
        # Check if it's BEIR format (has 'text' field directly)
        if "text" in query:
            # BEIR format - text is already prepared
            query_text = query["text"]
            # Note: query_mode is ignored for BEIR format since text is pre-processed
        elif "input" in query:
            # Full history format - need to process based on query_mode
            conversation_history = query.get("input", [])
            
            if not conversation_history:
                logger.warning(f"Empty history for task {query_id}")
                continue

            if query_mode == "last_turn":
                if conversation_history[-1]["speaker"] == "user":
                    query_text = conversation_history[-1]["text"]
                else:
                    # Fallback: find last user turn
                    for turn in reversed(conversation_history):
                        if turn["speaker"] == "user":
                            query_text = turn["text"]
                            break
            elif query_mode == "full_history":
                # Concatenate all user questions
                user_turns = [turn["text"] for turn in conversation_history if turn["speaker"] == "user"]
                query_text = " ".join(user_turns)
            elif query_mode == "full_context":
                # Concatenate all turns (User + Agent)
                all_turns = [turn["text"] for turn in conversation_history]
                query_text = " ".join(all_turns)
            else:
                logger.warning(f"Unknown query_mode {query_mode}, defaulting to last_turn")
                if conversation_history[-1]["speaker"] == "user":
                    query_text = conversation_history[-1]["text"]
        else:
            logger.warning(f"Unknown query format for {query_id}")
            continue

        if not query_text:
            logger.warning(f"Could not extract query for task {query_id}")
            continue
        
        with latency_monitor:
            # Retrieve top-k
            top_k = config["retrieval"].get("top_k", 100)
            # Use retrieve() instead of search()
            retrieved_results = retriever.retrieve(query_text, top_k=top_k)
        
        # Format result
        contexts = []
        for res in retrieved_results:
            doc_id = res["id"]
            score = res["score"]
            text = corpus.get(doc_id, "")
            contexts.append({"document_id": doc_id, "score": score, "text": text})

        result_entry = {
            "task_id": query_id,
            "question": query_text,
            "contexts": contexts,
            "Collection": domain,
            "turn_id": turn_id
        }
        results.append(result_entry)
        
    # Save raw results
    results_file = output_dir / "retrieval_results.jsonl"
    save_results(results, results_file)
    logger.info(f"Results saved to {results_file}")
    
    # 5. Evaluation
    logger.info("Running evaluation...")
    qrels = load_qrels(qrels_file)
    
    # Convert results to dict format for evaluation
    results_dict = {}
    for r in results:
        results_dict[r["task_id"]] = {ctx["document_id"]: ctx["score"] for ctx in r["contexts"]}
        
    scores_global, scores_per_query = compute_results(results_dict, qrels)
    
    # Save metrics
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(scores_global, f, indent=2)
    logger.info(f"Metrics saved to {metrics_file}")
    
    # 6. Statistical Analysis & Robustness
    logger.info("Running statistical and robustness analysis...")
    
    # Create DataFrame for analysis
    data_for_df = []
    for r in results:
        qid = r["task_id"]
        # scores_per_query is from pytrec_eval, keyed by query_id
        # It contains keys like 'ndcg_cut_10', 'recall_10', etc.
        metrics = scores_per_query.get(qid, {})
        
        row = {
            "task_id": qid,
            "turn": r["turn_id"],
            "collection": r["Collection"],
            "ndcg_at_10": metrics.get("ndcg_cut_10", 0.0),
            "recall_at_10": metrics.get("recall_10", 0.0)
        }
        data_for_df.append(row)
        
    df_results = pd.DataFrame(data_for_df)
    
    analysis_report = {
        "latency": latency_monitor.report(),
        "hard_failures": analyze_hard_failures(df_results, metric_col="ndcg_at_10").to_dict(orient="records"),
        "performance_by_turn": analyze_performance_by_turn(df_results, metric_col="ndcg_at_10", turn_col="turn").to_dict()
    }
    
    # Bootstrap CI for NDCG@10
    ndcg_10_scores = df_results["ndcg_at_10"].tolist()
    ci_lower, ci_upper = bootstrap_confidence_interval(ndcg_10_scores)
    analysis_report["bootstrap_ci_ndcg_10"] = {"lower": ci_lower, "upper": ci_upper}
    
    # Save analysis report
    analysis_file = output_dir / "analysis_report.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis_report, f, indent=2)
    logger.info(f"Analysis report saved to {analysis_file}")
    
    return scores_global
