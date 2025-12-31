"""
Voyage AI dense retrieval implementation.
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
    import voyageai
    import faiss
    VOYAGE_AVAILABLE = True
except ImportError:
    VOYAGE_AVAILABLE = False

load_dotenv()
logger = logging.getLogger(__name__)

CACHE_DIR = Path(".cache/embeddings/voyage")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

from .dense import DenseRetriever

class VoyageRetriever(DenseRetriever):
    """Voyage AI dense retrieval."""
    
    def __init__(self, index_path: Path, config: Dict[str, Any]):
        super().__init__(index_path, config)
        self.model_name = config.get("model_name", "voyage-large-2")
        
        if not VOYAGE_AVAILABLE:
            logger.warning("Voyage AI or FAISS not installed. Install with: pip install voyageai faiss-cpu")
            self.client = None
        else:
            api_key = os.getenv("VOYAGE_API_KEY")
            if not api_key:
                logger.warning("VOYAGE_API_KEY not found in environment")
                self.client = None
            else:
                self.client = voyageai.Client(api_key=api_key)
            
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
        
        # Move index to GPU if available (Voyage uses API for encoding, but FAISS search can be on GPU)
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
            # Check for alternative filename (pickle)
            alt_ids_path = self.index_path / "documents.pkl"
            if alt_ids_path.exists():
                logger.info(f"Loading doc IDs from {alt_ids_path}")
                with open(alt_ids_path, 'rb') as f:
                    data = pickle.load(f)
                    # Handle different pickle formats
                    if isinstance(data, list):
                        # If list of strings, use as is
                        if data and isinstance(data[0], str):
                            self.doc_ids = data
                        # If list of dicts, extract IDs
                        elif data and isinstance(data[0], dict):
                            self.doc_ids = [d.get("id") or d.get("_id") for d in data]
                        else:
                            logger.warning("Unknown format in documents.pkl, using raw list")
                            self.doc_ids = data
                    else:
                        logger.warning("documents.pkl is not a list")
                        self.doc_ids = []
            else:
                raise FileNotFoundError(f"Doc IDs not found at {ids_path} or {alt_ids_path}")
        else:
            with open(ids_path, 'r') as f:
                self.doc_ids = json.load(f)
            
        if len(self.doc_ids) != self.index.ntotal:
            logger.warning(f"Index size ({self.index.ntotal}) does not match doc IDs count ({len(self.doc_ids)})")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query using Voyage API."""
        if not self.client:
            raise RuntimeError("Voyage client not initialized")
            
        # Check cache
        cache_key_str = f"{self.model_name}_{query}"
        cache_key = hashlib.md5(cache_key_str.encode('utf-8')).hexdigest()
        cache_file = CACHE_DIR / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)
                # logger.debug(f"Cache hit for Voyage embedding: {cache_key}")
                return embedding
            except Exception as e:
                logger.warning(f"Failed to read cache {cache_file}: {e}")

        # Voyage API call
        # Note: Voyage embeddings are already normalized by default usually, but we can check
        response = self.client.embed(
            [query], 
            model=self.model_name, 
            input_type="query"
        )
        
        embedding = np.array(response.embeddings[0], dtype=np.float32)
        
        # Normalize if needed (FAISS cosine similarity usually expects normalized vectors)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        # Save to cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to write cache {cache_file}: {e}")
            
        return embedding
    
    def retrieve(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Retrieve using Voyage embeddings."""
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
