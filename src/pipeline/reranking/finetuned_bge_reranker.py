"""
Fine-tuned BGE Reranker implementation using transformers.

Uses the fine-tuned model: pedrovo9/bge-reranker-v2-m3-multirag-finetuned
"""

from typing import List, Dict, Any, Optional
import logging
import torch

logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Install with: pip install transformers")


class FineTunedBGEReranker:
    """
    Fine-tuned BGE Reranker for multi-domain RAG.
    
    Uses pedrovo9/bge-reranker-v2-m3-multirag-finetuned model fine-tuned on
    multi-turn conversational RAG data with proper train/test/val splits.
    
    Training details:
    - Base model: BAAI/bge-reranker-v2-m3
    - Training strategy: Pairwise learning (1:2 ratio)
    - Hard negatives: BM25
    - Epochs: 3
    - Domains: ClapNQ, Cloud, FiQA, Govt
    """
    
    def __init__(
        self,
        model_name: str = "pedrovo9/bge-reranker-v2-m3-multirag-finetuned",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize fine-tuned BGE reranker.
        
        Args:
            model_name: Hugging Face model name (default: pedrovo9/bge-reranker-v2-m3-multirag-finetuned)
            config: Additional configuration (batch_size, use_fp16, etc.)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers is required for fine-tuned BGE reranker. "
                "Install with: pip install transformers"
            )
        
        self.model_name = model_name
        self.config = config or {}
        use_fp16 = self.config.get("use_fp16", True)
        
        logger.info(f"Loading fine-tuned BGE reranker: {model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = self.model.to(self.device)
            if use_fp16:
                self.model = self.model.half()
                logger.info("Using FP16 precision")
        else:
            self.device = torch.device("cpu")
            logger.warning("CUDA not available, using CPU")
        
        self.model.eval()
        logger.info(f"✓ Fine-tuned BGE reranker loaded on {self.device}: {model_name}")
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using fine-tuned BGE reranker.
        
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
        
        # Extract texts
        texts = []
        for doc in documents:
            text = doc.get("text", doc.get("content", ""))
            texts.append(text)
        
        # Compute scores
        scores = self._compute_scores(query, texts)
        
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
    
    def _compute_scores(
        self,
        query: str,
        texts: List[str]
    ) -> List[float]:
        """
        Compute reranking scores for query-document pairs.
        
        Args:
            query: Search query
            texts: List of document texts
            
        Returns:
            List of scores (sigmoid of logits)
        """
        batch_size = self.config.get("batch_size", 32)
        scores = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize query-document pairs
                inputs = self.tokenizer(
                    [query] * len(batch_texts),
                    batch_texts,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get scores
                logits = self.model(**inputs).logits
                batch_scores = torch.sigmoid(logits[:, 0]).cpu().tolist()
                scores.extend(batch_scores)
        
        return scores
    
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
        # Update batch size in config for this call
        old_batch_size = self.config.get("batch_size")
        self.config["batch_size"] = batch_size
        
        results = []
        for query, docs in zip(queries, document_lists):
            reranked = self.rerank(query, docs, top_k)
            results.append(reranked)
        
        # Restore old batch size
        if old_batch_size is not None:
            self.config["batch_size"] = old_batch_size
        
        return results
