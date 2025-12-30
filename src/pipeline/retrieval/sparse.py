"""
Sparse retrieval implementations (BM25, ELSER).

Provides unified interface for lexical/sparse retrieval methods.
"""

import os
import json
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
    from nltk.tokenize import word_tokenize
    from elasticsearch import Elasticsearch
except ImportError:
    pass

logger = logging.getLogger(__name__)

class SparseRetriever:
    """Base class for sparse retrieval methods."""
    
    def __init__(self, index_path: Path, config: Dict[str, Any]):
        """
        Initialize sparse retriever.
        
        Args:
            index_path: Path to the directory containing index files
            config: Retrieval configuration
        """
        self.index_path = Path(index_path)
        self.config = config
    
    def retrieve(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Retrieve documents for a query."""
        raise NotImplementedError

class BM25Retriever(SparseRetriever):
    """BM25 retrieval using rank_bm25."""
    
    def __init__(self, index_path: Path, config: Dict[str, Any]):
        super().__init__(index_path, config)
        
        # Load BM25 model
        model_path = self.index_path / "index.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"BM25 index not found at {model_path}")
            
        logger.info(f"Loading BM25 index from {model_path}")
        with open(model_path, 'rb') as f:
            self.bm25 = pickle.load(f)
            
        # Load Doc IDs
        ids_path = self.index_path / "doc_ids.json"
        if not ids_path.exists():
            raise FileNotFoundError(f"Doc IDs not found at {ids_path}")
            
        with open(ids_path, 'r') as f:
            self.doc_ids = json.load(f)
            
    def retrieve(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Retrieve using BM25."""
        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_n = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_n:
            results.append({
                "id": self.doc_ids[idx],
                "score": float(scores[idx])
            })
        return results

class ELSERRetriever(SparseRetriever):
    """ELSER (Elastic Learned Sparse Encoder) retrieval."""
    
    def __init__(self, index_path: Path, config: Dict[str, Any]):
        super().__init__(index_path, config)
        self.es_url = os.getenv("ELASTICSEARCH_URL")
        self.es_user = os.getenv("ELASTICSEARCH_USER")
        self.es_password = os.getenv("ELASTICSEARCH_PASSWORD")
        self.domain = config.get("domain")
        self.index_name = f"mt-rag-{self.domain}-elser"
        self.model_id = ".elser_model_2"
        
        if not self.es_url:
            raise ValueError("ELASTICSEARCH_URL environment variable not set")
            
        self.es = Elasticsearch(
            self.es_url,
            basic_auth=(self.es_user, self.es_password) if self.es_user else None,
            verify_certs=False
        )
        
        if not self.es.ping():
            raise ConnectionError("Could not connect to Elasticsearch")
            
    def retrieve(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Retrieve using ELSER."""
        search_body = {
            "query": {
                "text_expansion": {
                    "ml.tokens": {
                        "model_id": self.model_id,
                        "model_text": query
                    }
                }
            },
            "size": top_k,
            "_source": ["id"]
        }
        
        response = self.es.search(index=self.index_name, body=search_body)
        
        results = []
        for hit in response['hits']['hits']:
            results.append({
                "id": hit['_source']['id'],
                "score": hit['_score']
            })
        return results

def get_sparse_retriever(
    model_name: str, 
    index_path: Path, 
    config: Dict[str, Any]
) -> SparseRetriever:
    """Factory function to get sparse retriever by name."""
    retrievers = {
        "bm25": BM25Retriever,
        "elser": ELSERRetriever,
    }
    
    if model_name not in retrievers:
        raise ValueError(f"Unknown sparse retriever: {model_name}")
    
    return retrievers[model_name](index_path, config)
