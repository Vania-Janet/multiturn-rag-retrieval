"""
Retrieval module for RAG pipeline.

Provides sparse, dense, and hybrid retrieval implementations.
"""

from .sparse import SparseRetriever, BM25Retriever, ELSERRetriever, SPLADERetriever, get_sparse_retriever
from .dense import DenseRetriever, BGEM3Retriever, get_dense_retriever
from .hybrid import HybridRetriever
from .fusion import reciprocal_rank_fusion, linear_combination, weighted_sum_fusion

__all__ = [
    # Sparse
    "SparseRetriever",
    "BM25Retriever",
    "ELSERRetriever", 
    "SPLADERetriever",
    "get_sparse_retriever",
    
    # Dense
    "DenseRetriever",
    "BGEM3Retriever",
    "get_dense_retriever",
    
    # Hybrid
    "HybridRetriever",
    
    # Fusion
    "reciprocal_rank_fusion",
    "linear_combination",
    "weighted_sum_fusion",
]
