"""
Hybrid retrieval combining sparse and dense methods.

Implements combination strategies for multi-retriever systems.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from .sparse import SparseRetriever, get_sparse_retriever
from .dense import DenseRetriever, get_dense_retriever
from .fusion import reciprocal_rank_fusion, linear_combination


class HybridRetriever:
    """
    Hybrid retriever combining sparse and dense methods.
    """
    
    def __init__(
        self,
        sparse_retriever: SparseRetriever,
        dense_retriever: DenseRetriever,
        fusion_method: str = "rrf",
        fusion_params: Optional[Dict[str, Any]] = None,
        sparse_weight: float = 0.5,
        dense_weight: float = 0.5
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            sparse_retriever: Initialized sparse retriever instance
            dense_retriever: Initialized dense retriever instance
            fusion_method: Method to combine results ("rrf" or "linear")
            fusion_params: Additional parameters for fusion (e.g., {"k": 60} for RRF)
            sparse_weight: Weight for sparse results in linear fusion (default: 0.5)
            dense_weight: Weight for dense results in linear fusion (default: 0.5)
        """
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.fusion_method = fusion_method
        self.fusion_params = fusion_params or {}
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
    
    def retrieve(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve using hybrid method.
        
        Args:
            query: Search query
            top_k: Number of final results to return
            
        Returns:
            Fused and ranked results
        """
        # Retrieve from sparse (get more for better fusion)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=top_k * 2)
        
        # Retrieve from dense (get more for better fusion)
        dense_results = self.dense_retriever.retrieve(query, top_k=top_k * 2)
        
        # Fuse results
        if self.fusion_method == "rrf":
            fused_results = reciprocal_rank_fusion(
                [sparse_results, dense_results],
                k=self.fusion_params.get("k", 60)
            )
        elif self.fusion_method == "linear":
            fused_results = linear_combination(
                sparse_results=sparse_results,
                dense_results=dense_results,
                sparse_weight=self.sparse_weight,
                dense_weight=self.dense_weight
            )
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        # Return top-k
        return fused_results[:top_k]
    
    def retrieve_separate(
        self, 
        query: str, 
        top_k: int = 100
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve separately without fusion (for analysis).
        
        Returns:
            Dict with "sparse" and "dense" result lists
        """
        return {
            "sparse": self.sparse_retriever.retrieve(query, top_k),
            "dense": self.dense_retriever.retrieve(query, top_k)
        }
