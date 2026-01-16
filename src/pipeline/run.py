import os

# Set HuggingFace cache to workspace volume BEFORE any imports
os.environ['HF_HOME'] = '/workspace/cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/workspace/cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/cache/transformers'
os.environ['HF_HUB_CACHE'] = '/workspace/cache/huggingface'

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import re

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
from .query_transform import get_rewriter
from .reranking import CohereReranker, BGEReranker, FineTunedBGEReranker
from .evaluation.run_retrieval_eval import compute_results, load_qrels, prepare_results_dict
from .retrieval.fusion import reciprocal_rank_fusion
from .utils.parent_context import build_parent_store, get_parent_id

logger = logging.getLogger(__name__)

def parse_conversation_from_text(text: str) -> List[Dict[str, str]]:
    """
    Parses flat text with |speaker|: tags into a structured conversation history.
    """
    if not text or ("|user|:" not in text and "|agent|:" not in text and "|model|:" not in text):
        return []
    
    turns = []
    # Split while keeping delimiters
    parts = re.split(r'(\|user\|:|\|agent\|:|\|model\|:)', text)
    
    current_speaker = None
    for part in parts:
        part = part.strip()
        if not part: continue
        
        if part in ["|user|:", "|agent|:", "|model|:"]:
            current_speaker = part.replace("|", "").replace(":", "")
            if current_speaker == "model": current_speaker = "agent"
        elif current_speaker:
            turns.append({"speaker": current_speaker, "text": part})
            
    return turns

def apply_rrf(result_lists: List[List[Dict[str, Any]]], k: int = 60, top_k: int = 100) -> List[Dict[str, Any]]:
    """
    Apply Reciprocal Rank Fusion to merge multiple ranked lists.
    
    Wrapper around fusion.reciprocal_rank_fusion that adds top_k truncation.
    
    Args:
        result_lists: List of retrieval result lists, each containing dicts with 'id' and 'score'
        k: RRF parameter (default 60)
        top_k: Number of results to return
        
    Returns:
        Merged and re-ranked results, truncated to top_k
    """
    # Use the canonical RRF implementation from fusion.py
    fused_results = reciprocal_rank_fusion(result_lists, k=k)
    
    # Truncate to top_k
    return fused_results[:top_k]

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

def run_pipeline(config: Dict[str, Any], output_dir: Path, domain: str, force: bool = False, baseline_path: Optional[Path] = None, num_comparisons: int = 1):
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
    
    # Setup query transformation if enabled
    query_rewriter = None
    if config.get("query_transform", {}).get("enabled", False):
        rewriter_type = config["query_transform"].get("rewriter_type", "identity")
        rewriter_config = config["query_transform"].get("rewriter_config", {})
        query_rewriter = get_rewriter(rewriter_type, **rewriter_config)
        logger.info(f"Query transformation enabled: {rewriter_type}")
    
    # 2. Initialize Retriever
    retrieval_type = config.get("retrieval", {}).get("type", "sparse")
    logger.info(f"Initializing {retrieval_type} retriever for domain {domain}")
    
    # Inject domain into retrieval config for retrievers that need it (like ELSER)
    if "retrieval" in config:
        config["retrieval"]["domain"] = domain
    
    if retrieval_type == "sparse":
        method = config["retrieval"].get("method", "bm25")
        # Determine index path based on method
        if method == "bm25":
            default_index_path = f"indices/{domain}/bm25"
        elif method == "splade":
            default_index_path = f"indices/{domain}/splade"
        else:
            default_index_path = f"indices/{domain}/elser" # Placeholder, ELSER uses ES
            
        index_path = config["retrieval"].get("index_path", default_index_path)
        
        retriever = get_sparse_retriever(
            model_name=method,
            index_path=index_path,
            config=config["retrieval"]
        )
    elif retrieval_type == "dense":
        model_name = config["retrieval"].get("model_name", "BAAI/bge-base-en-v1.5")
        
        # Handle Voyage model selection logic for Dense retrieval
        if "voyage" in model_name.lower():
            if domain == "fiqa":
                model_name = "voyage-finance-2"
                logger.info(f"Domain is 'fiqa': Forcing Voyage model to '{model_name}'")
            else:
                # User confirmed indices are voyage-3-large
                model_name = "voyage-3-large"
                logger.info(f"Domain is '{domain}': Using Voyage model '{model_name}' to match index")
            # Update config so retriever gets correct name
            config["retrieval"]["model_name"] = model_name
            default_index_path = f"indices/{domain}/voyage"
        # Determine default index path based on model
        elif "bge-m3" in model_name.lower():
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
        # Hybrid retrieval combining sparse and dense
        sparse_method = config["retrieval"].get("sparse", {}).get("method", "bm25")
        sparse_index = f"indices/{domain}/{sparse_method}"
        
        dense_model = config["retrieval"].get("dense", {}).get("model_name", "BAAI/bge-base-en-v1.5")
        
        # Handle Voyage model selection logic
        if "voyage" in dense_model.lower():
            if domain == "fiqa":
                dense_model = "voyage-finance-2"
                logger.info(f"Domain is 'fiqa': Forcing Voyage model to '{dense_model}'")
            else:
                dense_model = "voyage-3-large"
                logger.info(f"Domain is '{domain}': Using Voyage model '{dense_model}' to match index")
            
            # Update config with selected model so retriever gets correct name
            config["retrieval"]["dense"]["model_name"] = dense_model
            dense_index = f"indices/{domain}/voyage"
        elif "bge-m3" in dense_model.lower():
            dense_index = f"indices/{domain}/bge-m3"
        else:
            dense_index = f"indices/{domain}/bge"

        sparse = get_sparse_retriever(
            model_name=sparse_method,
            index_path=sparse_index,
            config=config["retrieval"].get("sparse", {})
        )
        dense = get_dense_retriever(
            model_name=dense_model,
            index_path=dense_index,
            config=config["retrieval"].get("dense", {})
        )
        
        fusion_method = config["retrieval"].get("fusion_method", "rrf")
        kwargs = {}
        if fusion_method == "linear" or fusion_method == "weighted_sum":
             alpha = config["retrieval"].get("alpha", 0.5)
             kwargs["sparse_weight"] = alpha
             kwargs["dense_weight"] = 1 - alpha

        retriever = HybridRetriever(
            sparse, 
            dense, 
            fusion_method=fusion_method,
            **kwargs
        )
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
    
    # Initialize Parent Store if needed
    parent_store = {}
    use_parent_context = config.get("reranking", {}).get("use_parent_context", False)
    if use_parent_context and corpus:
        parent_store = build_parent_store(corpus)

    logger.info(f"Loading queries from {query_file}")
    queries = load_queries(query_file)
    
    # 4. Retrieval Loop with Latency Monitoring
    latency_monitor = LatencyMonitor()
    results = []
    
    logger.info(f"Starting retrieval for {len(queries)} queries...")
    
    query_mode = config["data"].get("query_mode", "last_turn")
    logger.info(f"Using query extraction mode: {query_mode}")

    # Initialize Reranker if enabled
    reranker = None
    if config.get("reranking", {}).get("enabled", False):
        reranker_type = config["reranking"].get("reranker_type") or config["reranking"].get("type", "cohere")
        if reranker_type == "cohere":
            reranker = CohereReranker(
                model_name=config["reranking"].get("model_name", "rerank-v4.0-pro"),
                config=config["reranking"]
            )
            logger.info(f"Reranking enabled: {reranker_type} ({config['reranking'].get('model_name')})")
        elif reranker_type == "bge":
            reranker = BGEReranker(
                model_name=config["reranking"].get("model_name", "BAAI/bge-reranker-v2-m3"),
                config=config["reranking"]
            )
            logger.info(f"Reranking enabled: {reranker_type} ({config['reranking'].get('model_name')})")
        elif reranker_type == "finetuned_bge":
            reranker = FineTunedBGEReranker(
                model_name=config["reranking"].get("model_name", "pedrovo9/bge-reranker-v2-m3-multirag-finetuned"),
                config=config["reranking"]
            )
            logger.info(f"Reranking enabled: {reranker_type} ({config['reranking'].get('model_name')})")
        else:
            logger.warning(f"Unknown reranker type: {reranker_type}, reranking disabled")

    # Track query rewrites for analysis
    query_rewrite_log = []

    # OPTIMIZATION: Batch query rewriting for vLLM
    query_rewrites_map = {}
    if query_rewriter and hasattr(query_rewriter, 'batch_rewrite'):
        logger.info(f"Preparing batch query rewriting for {len(queries)} queries...")
        batch_queries = []
        batch_indices = []
        
        for idx, query in enumerate(queries):
            query_text = ""
            query_id = query.get("task_id") or query.get("_id")
            
            # Helper to consolidate input sources
            query_input = query.get("input", [])
            query_text_raw = query.get("text", "")
            
            # If input is missing but text has history tags, parse it
            if not query_input and query_text_raw and ("|user|:" in query_text_raw or "|agent|:" in query_text_raw):
                query_input = parse_conversation_from_text(query_text_raw)

            # Extract query based on mode
            if query_input:
                conversation_history = query_input
                if conversation_history:
                    if query_mode == "last_turn":
                        if conversation_history[-1]["speaker"] == "user":
                            query_text = conversation_history[-1]["text"]
                        else:
                            for turn in reversed(conversation_history):
                                if turn["speaker"] == "user":
                                    query_text = turn["text"]
                                    break
                    elif query_mode == "full_history":
                        user_turns = [turn["text"] for turn in conversation_history if turn["speaker"] == "user"]
                        query_text = " ".join(user_turns)
                    elif query_mode == "full_context":
                        all_turns = [turn["text"] for turn in conversation_history]
                        query_text = " ".join(all_turns)
            elif query_text_raw:
                query_text = query_text_raw
                if query_text:
                    query_text = query_text.replace("|user|:", "").replace("|agent|:", "").replace("|model|:", "").strip()
            
            if query_text:
                # Prepare context for rewriters that support it
                conversation_context = None
                if query_input and query_rewriter.__class__.__name__ in ["ContextualRewriter", "LLMRewriter", "VLLMRewriter", "HyDERewriter"]:
                    conversation_context = [turn["text"] for turn in query_input[:-1]]
                
                batch_queries.append((query_text, conversation_context))
                batch_indices.append((idx, query_id))
        
        if batch_queries:
            logger.info(f"Running batch rewrite for {len(batch_queries)} queries...")
            batch_results = query_rewriter.batch_rewrite(batch_queries)
            
            # Store results in map
            for (idx, query_id), rewrites in zip(batch_indices, batch_results):
                query_rewrites_map[idx] = rewrites
            
            logger.info(f"✓ Batch rewriting completed for {len(batch_results)} queries")

    for idx, query in enumerate(queries):
        # Support both BEIR format (_id + text) and full history format (input list)
        query_text = ""
        query_id = query.get("task_id") or query.get("_id")
        
        # Extract turn_id from query or from the ID format (conversation_id<::>turn or conversation_id::turn)
        turn_id = query.get("turn")
        if turn_id is None and query_id:
            # Try to extract from ID format: hash<::>turn or hash::turn
            if "<::>" in query_id:
                turn_id = int(query_id.split("<::>")[-1])
            elif "::" in query_id:
                turn_id = int(query_id.split("::")[-1])
            else:
                turn_id = 0
        elif turn_id is None:
            turn_id = 0
        
        # Helper to consolidate input sources
        query_input = query.get("input", [])
        query_text_raw = query.get("text", "")
        
        # If input is missing but text has history tags, parse it
        if not query_input and query_text_raw and ("|user|:" in query_text_raw or "|agent|:" in query_text_raw):
            query_input = parse_conversation_from_text(query_text_raw)
            
        if query_input:
            # Full history format - need to process based on query_mode
            conversation_history = query_input
            
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
            elif query_mode == "rewrite":
                # Use rewrite if available, fallback to input handling or text
                if "rewrite" in query and query["rewrite"]:
                    query_text = query["rewrite"]
                else:
                    # Fallback to last_turn behavior if rewrite field missing
                    if conversation_history[-1]["speaker"] == "user":
                        query_text = conversation_history[-1]["text"]
            else:
                logger.warning(f"Unknown query_mode {query_mode}, defaulting to last_turn")
                if conversation_history[-1]["speaker"] == "user":
                    query_text = conversation_history[-1]["text"]
        
        elif "text" in query:
            # BEIR format - text is already prepared
            query_text = query["text"]
            # Clean artifacts like |user|: if present
            if query_text:
                query_text = query_text.replace("|user|:", "").replace("|agent|:", "").replace("|model|:", "").strip()
            # Note: query_mode is ignored for BEIR format since text is pre-processed
        else:
            logger.warning(f"Unknown query format for {query_id}")
            continue

        if not query_text:
            logger.warning(f"Could not extract query for task {query_id}")
            continue
        
        # Apply query transformation if enabled
        queries_to_retrieve = [query_text]
        if query_rewriter:
            # Use pre-computed batch results if available
            if idx in query_rewrites_map:
                queries_to_retrieve = query_rewrites_map[idx]
                logger.debug(f"Using batch-rewritten query ({len(queries_to_retrieve)} variants)")
                # Log the rewrite for analysis
                query_rewrite_log.append({
                    "task_id": query_id,
                    "original": query_text,
                    "rewritten": queries_to_retrieve
                })
            else:
                # Fallback to single rewrite (for non-batch rewriters or cache misses)
                conversation_context = None
                if "input" in query and query_rewriter.__class__.__name__ in ["ContextualRewriter", "LLMRewriter", "VLLMRewriter", "HyDERewriter"]:
                    conversation_context = [turn["text"] for turn in query.get("input", [])[:-1]]
                
                queries_to_retrieve = query_rewriter.rewrite(query_text, context=conversation_context)
                logger.debug(f"Rewrote query into {len(queries_to_retrieve)} variants")
                # Log the rewrite for analysis
                query_rewrite_log.append({
                    "task_id": query_id,
                    "original": query_text,
                    "rewritten": queries_to_retrieve
                })
        
        # Handle multiple query variants
        all_retrieved_results = []
        merge_strategy = config.get("query_transform", {}).get("merge_strategy", "replace")
        
        with latency_monitor:
            top_k = config["retrieval"].get("top_k", 100)
            
            if merge_strategy == "replace" or len(queries_to_retrieve) == 1:
                retrieved_results = retriever.retrieve(queries_to_retrieve[0], top_k=top_k)
            elif merge_strategy == "rrf":
                # Retrieve for each variant and fuse with RRF
                all_results = []
                for variant_query in queries_to_retrieve:
                    variant_results = retriever.retrieve(variant_query, top_k=top_k)
                    all_results.append(variant_results)
                
                # Apply Reciprocal Rank Fusion
                rrf_k = config.get("fusion", {}).get("k", 60)
                retrieved_results = apply_rrf(all_results, rrf_k, top_k)
            else:
                logger.warning(f"Unknown merge_strategy: {merge_strategy}, using first query only")
                retrieved_results = retriever.retrieve(queries_to_retrieve[0], top_k=top_k)
        
        # Format result
        contexts = []
        logger.info(f"DEBUG: Fusion Results Count: {len(retrieved_results)}")
        for res in retrieved_results:
            doc_id = res["id"]
            score = res["score"]
            
            # CRITICAL FIX: Swap for parent text if expansion is enabled
            if use_parent_context:
                parent_id = get_parent_id(doc_id)
                # Use parent text if available, fallback to chunk text
                # We specifically check parent_store first
                if parent_id in parent_store:
                    text = parent_store[parent_id]
                else:
                    text = corpus.get(doc_id, "")
            else:
                text = corpus.get(doc_id, "")
                
            contexts.append({"document_id": doc_id, "score": score, "text": text})

        # Apply Reranking if enabled
        if reranker:
            # Use the rewritten query for reranking if available (critical fix)
            rerank_query = queries_to_retrieve[0] if queries_to_retrieve else query_text
            
            # Rerank the contexts
            # contexts already contain the correct text (including optional parent context) for reranking
            reranker_top_k = config["reranking"].get("top_k_candidates", config["reranking"].get("top_k", 100))
            reranked_contexts = reranker.rerank(
                query=rerank_query,
                documents=contexts,
                top_k=reranker_top_k
            )
            
            # Update contexts with reranked results - preserve all fields
            # Use rerank_score as the new score
            contexts = []
            for res in reranked_contexts:
                # Preserve all fields from reranker (includes rerank_score, original_score)
                context_entry = {
                    "document_id": res["document_id"],
                    "score": res.get("rerank_score", res["score"]),  # Use rerank_score if available
                    "text": res["text"]
                }
                # Add rerank_score and original_score if they exist
                if "rerank_score" in res:
                    context_entry["rerank_score"] = res["rerank_score"]
                if "original_score" in res:
                    context_entry["original_score"] = res["original_score"]
                contexts.append(context_entry)
            logger.debug(f"Reranked {len(contexts)} documents")

        # Truncate to top-10 for final output conformity
        contexts = contexts[:10]

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
    
    # Convert results to dict format for evaluation - NO ID MAPPING NEEDED
    # If lastturn.jsonl is used, IDs already match qrels format
    results_dict = {}
    for r in results:
        results_dict[r["task_id"]] = {ctx["document_id"]: ctx["score"] for ctx in r["contexts"]}
        
    # Extract k_values from config
    metrics_config = config.get("evaluation", {}).get("metrics", [])
    k_values = set()
    for m in metrics_config:
        if "@" in m:
            try:
                k = int(m.split("@")[1])
                k_values.add(k)
            except ValueError:
                pass
    
    if not k_values:
        k_values = {1, 3, 5, 10, 20, 100}
    
    # Ensure 5 and 10 are always present for analysis
    k_values.add(5)
    k_values.add(10)
    k_values = sorted(list(k_values))

    scores_global, scores_per_query = compute_results(results_dict, qrels, k_values=k_values)
    
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
        
        # Extract turn from turn_id field or from task_id
        turn = r.get("turn_id", 0)
        if turn == 0 and qid:
            # Try to extract from ID format: hash<::>turn or hash::turn
            if "<::>" in qid:
                turn = int(qid.split("<::>")[-1])
            elif "::" in qid:
                turn = int(qid.split("::")[-1])
        
        row = {
            "task_id": qid,
            "turn": turn,
            "collection": r["Collection"],
            "ndcg_at_10": metrics.get("ndcg_cut_10", 0.0),
            "recall_at_10": metrics.get("recall_10", 0.0),
            "ndcg_at_5": metrics.get("ndcg_cut_5", 0.0),
            "recall_at_5": metrics.get("recall_5", 0.0),
            "recall_at_20": metrics.get("recall_20", 0.0),
            "recall_at_100": metrics.get("recall_100", 0.0),
            "precision_at_5": metrics.get("P_5", 0.0),
            "precision_at_10": metrics.get("P_10", 0.0)
        }
        data_for_df.append(row)
        
    df_results = pd.DataFrame(data_for_df)
    
    # Use NDCG@5 for primary analysis as requested
    primary_metric = "ndcg_at_5"
    
    analysis_report = {
        "latency": latency_monitor.report(),
        "hard_failures": analyze_hard_failures(df_results, metric_col=primary_metric).to_dict(orient="records"),
        "performance_by_turn": analyze_performance_by_turn(df_results, metric_col=primary_metric, turn_col="turn").reset_index().to_dict(orient="records")
    }
    
    # Bootstrap CI for NDCG@5
    ndcg_5_scores = df_results[primary_metric].tolist()
    bootstrap_result = bootstrap_confidence_interval(ndcg_5_scores)
    analysis_report[f"bootstrap_ci_{primary_metric}"] = bootstrap_result
    
    # Significance Testing (Wilcoxon)
    if baseline_path and Path(baseline_path).exists():
        logger.info(f"Comparing against baseline: {baseline_path}")
        try:
            # Load baseline results
            baseline_results, _ = prepare_results_dict(baseline_path)
            
            # Evaluate baseline with SAME qrels
            _, baseline_scores_per_query = compute_results(baseline_results, qrels, k_values=k_values)
            
            # Align scores
            current_ndcg = []
            baseline_ndcg = []
            
            for qid in scores_per_query:
                if qid in baseline_scores_per_query:
                    current_ndcg.append(scores_per_query[qid].get("ndcg_cut_5", 0.0))
                    baseline_ndcg.append(baseline_scores_per_query[qid].get("ndcg_cut_5", 0.0))
            
            if current_ndcg:
                wilcoxon_results = calculate_wilcoxon_significance(baseline_ndcg, current_ndcg)
                
                # Apply Bonferroni Correction
                bonferroni_results = apply_bonferroni_correction(wilcoxon_results["p_value"], num_tests=num_comparisons)
                wilcoxon_results["bonferroni"] = bonferroni_results
                
                analysis_report["significance_test"] = wilcoxon_results
                logger.info(f"Wilcoxon Test Results (NDCG@5): {wilcoxon_results}")
            else:
                logger.warning("No overlapping queries found between current run and baseline.")
                
        except Exception as e:
            logger.error(f"Failed to run significance test: {e}")

    # Query Variance Analysis (Turn-based)
    try:
        variance_by_turn = analyze_query_variance(df_results, group_by_col="turn", metric_col=primary_metric)
        # Convert numpy types to Python native types for JSON serialization
        variance_dict = variance_by_turn.to_dict()
        # Convert all numpy types to Python types
        for key in variance_dict:
            for subkey in variance_dict[key]:
                if hasattr(variance_dict[key][subkey], 'item'):
                    variance_dict[key][subkey] = variance_dict[key][subkey].item()
        analysis_report["variance_by_turn"] = variance_dict
    except Exception as e:
        logger.warning(f"Could not calculate variance by turn: {e}")

    # Save analysis report
    analysis_file = output_dir / "analysis_report.json"
    # Use default handler for numpy types
    def convert_numpy(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(analysis_file, 'w') as f:
        json.dump(analysis_report, f, indent=2, default=convert_numpy)
    logger.info(f"Analysis report saved to {analysis_file}")
    
    # Save query rewrite log if any rewrites were performed
    if query_rewrite_log:
        rewrite_log_file = output_dir / "query_rewrites.jsonl"
        with open(rewrite_log_file, 'w') as f:
            for entry in query_rewrite_log:
                f.write(json.dumps(entry) + '\n')
        logger.info(f"Query rewrite log saved to {rewrite_log_file} ({len(query_rewrite_log)} queries)")
    
    return scores_global

if __name__ == "__main__":
    import argparse
    import yaml
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config file")
    parser.add_argument("--domain", type=str, required=True, help="Domain name (e.g., clapnq, cloud, fiqa, govt)")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--baseline", type=str, default=None, help="Path to baseline evaluation results for comparison")
    parser.add_argument("--force", action="store_true", help="Force rerun even if results exist")
    
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Replace {domain} placeholders in config paths
    def replace_domain_placeholders(obj, domain):
        if isinstance(obj, dict):
            return {k: replace_domain_placeholders(v, domain) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_domain_placeholders(item, domain) for item in obj]
        elif isinstance(obj, str):
            return obj.replace("{domain}", domain)
        return obj
    
    config = replace_domain_placeholders(config, args.domain)
        
    # Determine output directory
    if args.output_dir:
        output_path = Path(args.output_dir)
    else:
        # Default: experiments/<experiment_name>/<domain>
        exp_name = config.get("experiment", {}).get("name", "experiment")
        output_path = Path("experiments") / exp_name / args.domain
        
    config["output_dir"] = str(output_path)
    
    run_pipeline(
        config=config,
        output_dir=output_path,
        domain=args.domain,
        force=args.force,
        baseline_path=args.baseline
    )
