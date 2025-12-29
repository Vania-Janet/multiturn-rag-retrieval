"""
Result fusion strategies for combining multiple retrievers.

Implements RRF, linear combination, and other fusion methods.
"""

from typing import List, Dict, Any
from collections import defaultdict


def reciprocal_rank_fusion(
    result_lists: List[List[Dict[str, Any]]], 
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) for combining multiple result lists.
    
    RRF score for document d:
        RRF(d) = sum over all rankers r of: 1 / (k + rank_r(d))
    
    Args:
        result_lists: List of result lists from different retrievers
            Each result dict should have "doc_id" and optionally "score"
        k: Constant to avoid division by zero (default: 60)
        
    Returns:
        Fused and re-ranked results
    """
    # Calculate RRF scores
    rrf_scores = defaultdict(float)
    doc_info = {}  # Store document metadata
    
    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            doc_id = result["doc_id"]
            rrf_scores[doc_id] += 1.0 / (k + rank)
            
            # Keep document info from first occurrence
            if doc_id not in doc_info:
                doc_info[doc_id] = result
    
    # Sort by RRF score
    sorted_docs = sorted(
        rrf_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Build result list
    fused_results = []
    for doc_id, rrf_score in sorted_docs:
        result = doc_info[doc_id].copy()
        result["rrf_score"] = rrf_score
        result["fusion_method"] = "rrf"
        fused_results.append(result)
    
    return fused_results


def linear_combination(
    sparse_results: List[Dict[str, Any]],
    dense_results: List[Dict[str, Any]],
    sparse_weight: float = 0.5,
    dense_weight: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Linear combination of sparse and dense scores.
    
    Final score = sparse_weight * sparse_score + dense_weight * dense_score
    
    Note: Scores are normalized to [0, 1] before combination.
    
    Args:
        sparse_results: Results from sparse retriever
        dense_results: Results from dense retriever
        sparse_weight: Weight for sparse scores
        dense_weight: Weight for dense scores
        
    Returns:
        Combined and re-ranked results
    """
    # Normalize scores
    def normalize_scores(results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not results:
            return {}
        
        scores = [r["score"] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score
        
        if score_range == 0:
            return {r["doc_id"]: 1.0 for r in results}
        
        return {
            r["doc_id"]: (r["score"] - min_score) / score_range 
            for r in results
        }
    
    sparse_scores = normalize_scores(sparse_results)
    dense_scores = normalize_scores(dense_results)
    
    # Combine scores
    all_doc_ids = set(sparse_scores.keys()) | set(dense_scores.keys())
    combined_scores = {}
    doc_info = {}
    
    # Store document info
    for result in sparse_results + dense_results:
        if result["doc_id"] not in doc_info:
            doc_info[result["doc_id"]] = result
    
    # Calculate combined scores
    for doc_id in all_doc_ids:
        sparse_score = sparse_scores.get(doc_id, 0.0)
        dense_score = dense_scores.get(doc_id, 0.0)
        
        combined_scores[doc_id] = (
            sparse_weight * sparse_score + 
            dense_weight * dense_score
        )
    
    # Sort by combined score
    sorted_docs = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Build result list
    fused_results = []
    for doc_id, combined_score in sorted_docs:
        result = doc_info[doc_id].copy()
        result["combined_score"] = combined_score
        result["fusion_method"] = "linear"
        result["sparse_weight"] = sparse_weight
        result["dense_weight"] = dense_weight
        fused_results.append(result)
    
    return fused_results


def weighted_sum_fusion(
    result_lists: List[List[Dict[str, Any]]],
    weights: List[float]
) -> List[Dict[str, Any]]:
    """
    Weighted sum fusion for N retrievers.
    
    Args:
        result_lists: List of result lists
        weights: Weight for each retriever (should sum to 1.0)
        
    Returns:
        Fused results
    """
    if len(result_lists) != len(weights):
        raise ValueError("Number of result lists must match number of weights")
    
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("Weights should sum to 1.0")
    
    # Normalize each result list
    normalized_lists = []
    for results in result_lists:
        if not results:
            normalized_lists.append({})
            continue
        
        scores = [r["score"] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score
        
        if score_range == 0:
            normalized = {r["doc_id"]: 1.0 for r in results}
        else:
            normalized = {
                r["doc_id"]: (r["score"] - min_score) / score_range
                for r in results
            }
        
        normalized_lists.append(normalized)
    
    # Combine scores
    all_doc_ids = set()
    doc_info = {}
    
    for results in result_lists:
        for result in results:
            all_doc_ids.add(result["doc_id"])
            if result["doc_id"] not in doc_info:
                doc_info[result["doc_id"]] = result
    
    # Calculate weighted scores
    weighted_scores = {}
    for doc_id in all_doc_ids:
        score = sum(
            weight * normalized.get(doc_id, 0.0)
            for weight, normalized in zip(weights, normalized_lists)
        )
        weighted_scores[doc_id] = score
    
    # Sort and build results
    sorted_docs = sorted(
        weighted_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    fused_results = []
    for doc_id, score in sorted_docs:
        result = doc_info[doc_id].copy()
        result["fused_score"] = score
        result["fusion_method"] = "weighted_sum"
        fused_results.append(result)
    
    return fused_results
