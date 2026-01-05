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
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    import scipy.sparse
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
        
        # 1. INITIALIZE ELASTICSEARCH CONNECTION FIRST
        if not self.es_url:
            self.es_url = "http://localhost:9200"
            logger.warning("ELASTICSEARCH_URL not set, defaulting to localhost:9200")
            
        self.es = Elasticsearch(
            self.es_url,
            basic_auth=(self.es_user, self.es_password) if self.es_user else None,
            verify_certs=False
        )
        
        if not self.es.ping():
            raise ConnectionError(f"Could not connect to Elasticsearch at {self.es_url}")
        
        logger.info(f"Connected to Elasticsearch at {self.es_url}")
        
        # 2. NOW SEARCH FOR THE INDEX (uses self.es)
        # Use the exact index name format from the user's data
        # e.g., mt-rag-clapnq-elser-512-100-20240503
        # We will try to find the correct index name by pattern matching
        try:
            indices = self.es.cat.indices(format="json")
            matching_indices = [
                i['index'] for i in indices 
                if f"mt-rag-{self.domain}-elser" in i['index']
            ]
            if matching_indices:
                # Sort by date/length to get most specific/recent
                self.index_name = sorted(matching_indices)[-1]
                logger.info(f"Found ELSER index: {self.index_name}")
            else:
                self.index_name = f"mt-rag-{self.domain}-elser"
                logger.warning(f"No matching index found, using default: {self.index_name}")
        except Exception as e:
            logger.warning(f"Failed to list indices: {e}")
            self.index_name = f"mt-rag-{self.domain}-elser"

        self.model_id = ".elser_model_2"
            
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

class SPLADERetriever(SparseRetriever):
    """SPLADE retrieval."""
    
    def __init__(self, index_path: Path, config: Dict[str, Any]):
        super().__init__(index_path, config)
        self.model_name = config.get("model_name", "naver/splade-cocondenser-ensembledistil")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Index
        index_file = self.index_path / "index.npz"
        if not index_file.exists():
            raise FileNotFoundError(f"SPLADE index not found at {index_file}")
            
        logger.info(f"Loading SPLADE index from {index_file}")
        self.index_matrix = scipy.sparse.load_npz(index_file)
        
        # Load Doc IDs
        ids_path = self.index_path / "doc_ids.json"
        if not ids_path.exists():
            raise FileNotFoundError(f"Doc IDs not found at {ids_path}")
            
        with open(ids_path, 'r') as f:
            self.doc_ids = json.load(f)
            
        # Load Model
        logger.info(f"Loading SPLADE model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def encode_query(self, query: str):
        with torch.no_grad():
            inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            values = torch.log(1 + torch.relu(logits)) * attention_mask
            sparse_vec = torch.max(values, dim=1).values
            return sparse_vec.cpu().numpy()

    def retrieve(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Retrieve using SPLADE."""
        query_vec = self.encode_query(query) # Shape (1, vocab_size)
        
        # Sparse Dot Product
        query_sparse = scipy.sparse.csr_matrix(query_vec)
        scores = self.index_matrix.dot(query_sparse.T).toarray().flatten()
        
        # Get top-k
        if len(scores) > top_k:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        else:
            top_indices = np.argsort(scores)[::-1]
            
        results = []
        for idx in top_indices:
            if scores[idx] > -100: # Simple threshold, though usually positive
                results.append({
                    "id": self.doc_ids[idx],
                    "score": float(scores[idx])
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
        "splade": SPLADERetriever,
    }
    
    if model_name not in retrievers:
        raise ValueError(f"Unknown sparse retriever: {model_name}")
    
    return retrievers[model_name](index_path, config)
