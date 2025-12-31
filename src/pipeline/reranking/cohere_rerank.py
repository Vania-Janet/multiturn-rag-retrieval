"""
Cohere Rerank implementation.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

load_dotenv()
logger = logging.getLogger(__name__)

class CohereReranker:
    """
    Cohere Rerank API wrapper.
    """
    
    def __init__(
        self, 
        model_name: str = "rerank-english-v3.0",
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.model_name = model_name
        self.config = config or {}
        
        if not COHERE_AVAILABLE:
            logger.warning("Cohere not installed. Install with: pip install cohere")
            self.client = None
            return

        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            logger.warning("COHERE_API_KEY not found in environment")
            self.client = None
        else:
            self.client = cohere.Client(self.api_key)
            
    def rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using Cohere API.
        """
        if not self.client or not documents:
            return documents
            
        try:
            # Prepare documents for Cohere (needs list of strings or dicts)
            # We'll use list of strings for simplicity if 'text' is available
            docs_to_rerank = []
            for doc in documents:
                text = doc.get("text", "")
                if not text:
                    # Fallback to other fields if text is missing
                    text = doc.get("content", "")
                docs_to_rerank.append(text)
                
            response = self.client.rerank(
                model=self.model_name,
                query=query,
                documents=docs_to_rerank,
                top_n=top_k or len(documents)
            )
            
            # Map results back to original documents
            reranked_docs = []
            for result in response.results:
                original_doc = documents[result.index]
                # Create a copy to avoid modifying original in place if needed
                doc_copy = original_doc.copy()
                doc_copy["score"] = result.relevance_score
                doc_copy["original_score"] = original_doc.get("score", 0.0)
                reranked_docs.append(doc_copy)
                
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}")
            return documents
