"""
BGE Reranker implementation using FlagEmbedding library.
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import FlagEmbedding for BGE reranker
try:
    from FlagEmbedding import FlagReranker
    BGE_RERANKER_AVAILABLE = True
except ImportError:
    BGE_RERANKER_AVAILABLE = False
    logger.warning("FlagEmbedding not available. Install with: pip install FlagEmbedding")


class BGEReranker:
    """
    BGE Reranker using BAAI FlagEmbedding models.
    
    Supports models like:
    - BAAI/bge-reranker-v2-m3
    - BAAI/bge-reranker-large
    - BAAI/bge-reranker-base
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize BGE reranker.
        
        Args:
            model_name: Hugging Face model name
            config: Additional configuration (batch_size, use_fp16, etc.)
        """
        if not BGE_RERANKER_AVAILABLE:
            raise ImportError(
                "FlagEmbedding is required for BGE reranker. "
                "Install with: pip install FlagEmbedding"
            )
        
        self.model_name = model_name
        self.config = config or {}
        
        # Initialize model
        use_fp16 = self.config.get("use_fp16", True)
        logger.info(f"Loading BGE reranker: {model_name} (fp16={use_fp16})")
        
        self.model = FlagReranker(
            model_name,
            use_fp16=use_fp16
        )
        
        logger.info(f"✓ BGE reranker loaded: {model_name}")
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using BGE reranker.
        
        Args:
            query: Search query
            documents: List of documents to rerank
                Each dict should have "text" or "content" field
            top_k: Number of top results to return (None = all)
            
        Returns:
            Reranked documents with rerank scores
        """
        if not documents:
            return []
        
        # Prepare query-document pairs
        pairs = []
        for doc in documents:
            text = doc.get("text", doc.get("content", ""))
            pairs.append([query, text])
        
        # Get reranking scores
        batch_size = self.config.get("batch_size", 32)
        scores = self.model.compute_score(
            pairs,
            batch_size=batch_size,
            normalize=True  # Normalize scores to [0, 1]
        )
        
        # Handle single document case (returns float instead of list)
        if not isinstance(scores, list):
            scores = [scores]
        
        # Add scores to documents
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
            doc["original_score"] = doc.get("score", 0.0)
        
        # Sort by rerank score
        reranked = sorted(
            documents,
            key=lambda x: x["rerank_score"],
            reverse=True
        )
        
        # Return top-k
        if top_k:
            reranked = reranked[:top_k]
        
        return reranked
    
    def batch_rerank(
        self,
        queries: List[str],
        document_lists: List[List[Dict[str, Any]]],
        top_k: Optional[int] = None,
        batch_size: int = 32
    ) -> List[List[Dict[str, Any]]]:
        """
        Rerank multiple query-document sets.
        
        Args:
            queries: List of queries
            document_lists: List of document lists (one per query)
            top_k: Number of top results per query
            batch_size: Batch size for model inference
            
        Returns:
            List of reranked document lists
        """
        results = []
        for query, docs in zip(queries, document_lists):
            reranked = self.rerank(query, docs, top_k)
            results.append(reranked)
        return results
