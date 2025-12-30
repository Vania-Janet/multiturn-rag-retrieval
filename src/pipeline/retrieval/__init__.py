"""
Retrieval module for RAG pipeline.

Provides sparse, dense, and hybrid retrieval implementations.
"""

from .sparse import SparseRetriever, BM25Retriever, ELSERRetriever, get_sparse_retriever
from .dense import DenseRetriever, BGERetriever, get_dense_retriever
from .hybrid import HybridRetriever
from .fusion import reciprocal_rank_fusion, linear_combination, weighted_sum_fusion
from .reproducibility import (
    set_seed, 
    calculate_wilcoxon_significance, 
    bootstrap_confidence_interval, 
    report_stability,
    apply_bonferroni_correction
)
from .analysis import (
    LatencyMonitor,
    analyze_hard_failures,
    analyze_performance_by_turn,
    analyze_query_variance
)

__all__ = [
    # Sparse
    "SparseRetriever",
    "BM25Retriever",
    "ELSERRetriever", 
    "get_sparse_retriever",
    
    # Dense
    "DenseRetriever",
    "BGERetriever",
    "get_dense_retriever",
    
    # Hybrid
    "HybridRetriever",
    
    # Fusion
    "reciprocal_rank_fusion",
    "linear_combination",
    "weighted_sum_fusion",

    # Reproducibility & Stats
    "set_seed",
    "calculate_wilcoxon_significance",
    "bootstrap_confidence_interval",
    "report_stability",
    "apply_bonferroni_correction",
    
    # Analysis
    "LatencyMonitor",
    "analyze_hard_failures",
    "analyze_performance_by_turn",
    "analyze_query_variance"
]
