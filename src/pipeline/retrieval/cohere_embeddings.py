"""
Cohere embeddings dense retrieval implementation.
"""

import os
import json
import logging
import hashlib
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

try:
    import cohere
    import faiss
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

load_dotenv()
logger = logging.getLogger(__name__)

CACHE_DIR = Path(".cache/embeddings/cohere")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

from .dense import DenseRetriever

class CohereRetriever(DenseRetriever):
    """Cohere embeddings dense retrieval."""
    
    def __init__(self, index_path: Path, config: Dict[str, Any]):
        super().__init__(index_path, config)
        self.model_name = config.get("model_name", "embed-english-v3.0")
        
        if not COHERE_AVAILABLE:
            logger.warning("Cohere or FAISS not installed. Install with: pip install cohere faiss-cpu")
            self.client = None
        else:
            api_key = os.getenv("COHERE_API_KEY")
            if not api_key:
                logger.warning("COHERE_API_KEY not found in environment")
                self.client = None
            else:
                self.client = cohere.Client(api_key=api_key)
            
        # Load FAISS index
        faiss_path = self.index_path / "index.faiss"
        if not faiss_path.exists():
            # Check for alternative filename
            alt_faiss_path = self.index_path / "faiss_index.bin"
            if alt_faiss_path.exists():
                faiss_path = alt_faiss_path
            else:
                raise FileNotFoundError(f"FAISS index not found at {faiss_path} or {alt_faiss_path}")
            
        logger.info(f"Loading FAISS index from {faiss_path}")
        self.index = faiss.read_index(str(faiss_path))
        
        # Move index to GPU if available
        try:
            import torch
            if torch.cuda.is_available():
                try:
                    res = faiss.StandardGpuResources()
                    if torch.cuda.device_count() > 1:
                        logger.info(f"Moving FAISS index to {torch.cuda.device_count()} GPUs")
                        self.index = faiss.index_cpu_to_all_gpus(self.index)
                    else:
                        logger.info("Moving FAISS index to GPU")
                        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                except Exception as e:
                    logger.warning(f"Failed to move FAISS index to GPU: {e}. Continuing with CPU index.")
        except ImportError:
            pass
        
        # Load Doc IDs
        ids_path = self.index_path / "doc_ids.json"
        if not ids_path.exists():
            # Check for alternative filenames
            for alt_name in ["documents.pkl", "doc_ids.pkl"]:
                alt_ids_path = self.index_path / alt_name
                if alt_ids_path.exists():
                    logger.info(f"Loading doc IDs from {alt_ids_path}")
                    with open(alt_ids_path, 'rb') as f:
                        data = pickle.load(f)
                        if isinstance(data, list):
                            if data and isinstance(data[0], str):
                                self.doc_ids = data
                            elif data and isinstance(data[0], dict):
                                self.doc_ids = [d.get("id") or d.get("_id") for d in data]
                            else:
                                self.doc_ids = data
                        else:
                            self.doc_ids = []
                    ids_path = alt_ids_path
                    break
            
            if not ids_path.exists():
                raise FileNotFoundError(f"Doc IDs not found at {self.index_path}")
        else:
            logger.info(f"Loading doc IDs from {ids_path}")
            with open(ids_path, 'r') as f:
                self.doc_ids = json.load(f)
        
        if len(self.doc_ids) != self.index.ntotal:
            logger.warning(f"Index size ({self.index.ntotal}) does not match doc IDs count ({len(self.doc_ids)})")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query using Cohere embeddings API."""
        if not self.client:
            raise RuntimeError("Cohere client not initialized")
        
        # Check cache first
        cache_key = hashlib.md5(f"{self.model_name}:{query}".encode()).hexdigest()
        cache_path = CACHE_DIR / f"{cache_key}.npy"
        
        if cache_path.exists():
            return np.load(cache_path)
        
        try:
            # Cohere v4 API with search_document input type for queries
            response = self.client.embed(
                texts=[query],
                model=self.model_name,
                input_type="search_query",  # Important for search use case
                embedding_types=["float"]
            )
            
            embedding = np.array(response.embeddings.float[0], dtype=np.float32)
            
            # Cache result
            np.save(cache_path, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error encoding query with Cohere: {e}")
            raise
    
    def retrieve(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Retrieve using Cohere embeddings."""
        query_embedding = self.encode_query(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize for cosine similarity (Cohere embeddings are already normalized but ensure it)
        faiss.normalize_L2(query_embedding)
        
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
