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
        sparse_config: Dict[str, Any],
        dense_config: Dict[str, Any],
        fusion_method: str = "rrf",
        fusion_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            sparse_config: Configuration for sparse retriever
                {
                    "model": "splade",
                    "index_path": "indices/clapnq/splade",
                    "weight": 0.6
                }
            dense_config: Configuration for dense retriever
                {
                    "model": "bge-m3",
                    "index_path": "indices/clapnq/bge-m3",
                    "weight": 0.4
                }
            fusion_method: Method to combine results ("rrf" or "linear")
            fusion_params: Additional parameters for fusion
        """
        self.sparse_config = sparse_config
        self.dense_config = dense_config
        self.fusion_method = fusion_method
        self.fusion_params = fusion_params or {}
        
        # Initialize retrievers
        self.sparse_retriever = get_sparse_retriever(
            model_name=sparse_config["model"],
            index_path=Path(sparse_config["index_path"]),
            config=sparse_config
        )
        
        self.dense_retriever = get_dense_retriever(
            model_name=dense_config["model"],
            index_path=Path(dense_config["index_path"]),
            config=dense_config
        )
    
    def retrieve(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve using hybrid method.
        
        Args:
            query: Search query
            top_k: Number of final results to return
            
        Returns:
            Fused and ranked results
        """
        # Retrieve from sparse
        sparse_results = self.sparse_retriever.retrieve(
            query, 
            top_k=self.sparse_config.get("top_k", top_k * 2)
        )
        
        # Retrieve from dense
        dense_results = self.dense_retriever.retrieve(
            query,
            top_k=self.dense_config.get("top_k", top_k * 2)
        )
        
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
                sparse_weight=self.sparse_config.get("weight", 0.5),
                dense_weight=self.dense_config.get("weight", 0.5)
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
