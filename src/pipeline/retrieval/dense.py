"""
Dense retrieval implementations (BGE 1.5, etc.).

Provides unified interface for semantic/dense retrieval methods.
"""

import os
import json
import logging
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass

logger = logging.getLogger(__name__)

class DenseRetriever:
    """Base class for dense retrieval methods."""
    
    def __init__(self, index_path: Path, config: Dict[str, Any]):
        """
        Initialize dense retriever.
        
        Args:
            index_path: Path to the directory containing index files
            config: Retrieval configuration
        """
        self.index_path = Path(index_path)
        self.config = config
        self.model = None
        self.index = None
        self.doc_ids = []
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode query to dense vector."""
        raise NotImplementedError
    
    def retrieve(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Retrieve documents for a query."""
        raise NotImplementedError

class BGERetriever(DenseRetriever):
    """BGE 1.5 dense retrieval."""
    
    def __init__(self, index_path: Path, config: Dict[str, Any]):
        super().__init__(index_path, config)
        self.model_name = config.get("model_name", "BAAI/bge-base-en-v1.5")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading BGE model: {self.model_name} on {self.device}")
        # Use DataParallel if multiple GPUs are available and we are on CUDA
        if self.device == "cuda" and torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs for encoding")
            # SentenceTransformer handles multi-gpu via device='cuda' usually, 
            # but for explicit parallel encoding we rely on its internal implementation or just use the primary GPU for single query latency.
            # For retrieval (single query), using one GPU is often faster due to overhead.
            pass

        self.model = SentenceTransformer(self.model_name, device=self.device)
        if self.device == "cuda":
            self.model.half()
            
        # Load FAISS index
        faiss_path = self.index_path / "index.faiss"
        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {faiss_path}")
            
        logger.info(f"Loading FAISS index from {faiss_path}")
        self.index = faiss.read_index(str(faiss_path))
        
        # Move index to GPU if available
        if self.device == "cuda":
            try:
                # Use all available GPUs for the index
                res = faiss.StandardGpuResources()
                # If multiple GPUs, use index_cpu_to_all_gpus
                if torch.cuda.device_count() > 1:
                    logger.info(f"Moving FAISS index to {torch.cuda.device_count()} GPUs")
                    self.index = faiss.index_cpu_to_all_gpus(self.index)
                else:
                    logger.info("Moving FAISS index to GPU")
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception as e:
                logger.warning(f"Failed to move FAISS index to GPU: {e}. Continuing with CPU index.")
        
        # Load Doc IDs
        ids_path = self.index_path / "doc_ids.json"
        if not ids_path.exists():
            raise FileNotFoundError(f"Doc IDs not found at {ids_path}")
            
        with open(ids_path, 'r') as f:
            self.doc_ids = json.load(f)
            
        if len(self.doc_ids) != self.index.ntotal:
            logger.warning(f"Index size ({self.index.ntotal}) does not match doc IDs count ({len(self.doc_ids)})")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query using BGE model."""
        # Get instruction from config, default to BGE 1.5 instruction
        default_instruction = "Represent this sentence for searching relevant passages: "
        instruction = self.config.get("query_instruction", default_instruction)
        
        query_with_instruction = instruction + query
        
        embedding = self.model.encode(
            query_with_instruction,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embedding
    
    def retrieve(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Retrieve using BGE embeddings."""
        query_embedding = self.encode_query(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if idx < len(self.doc_ids):
                results.append({
                    "id": self.doc_ids[idx],
                    "score": float(score)
                })
        return results

def get_dense_retriever(
    model_name: str, 
    index_path: Path, 
    config: Dict[str, Any]
) -> DenseRetriever:
    """Factory function to get dense retriever by name."""
    # Map common names to BGERetriever
    if "bge" in model_name.lower():
        return BGERetriever(index_path, config)
    
    if "voyage" in model_name.lower():
        from .voyage import VoyageRetriever
        return VoyageRetriever(index_path, config)
    
    raise ValueError(f"Unknown dense retriever: {model_name}")
