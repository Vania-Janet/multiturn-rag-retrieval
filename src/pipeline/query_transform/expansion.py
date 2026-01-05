"""
Query expansion strategies for improving recall.

This module implements various query expansion techniques:
- Synonym expansion
- PRF (Pseudo-Relevance Feedback)
- Multi-query generation
- Domain-specific expansion
"""

import logging
from typing import List, Set, Optional, Dict
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class QueryExpander(ABC):
    """Base class for query expansion strategies."""
    
    @abstractmethod
    def expand(self, query: str, top_docs: Optional[List[str]] = None) -> List[str]:
        """
        Expand a query with additional terms.
        
        Args:
            query: Original query string
            top_docs: Optional list of top retrieved documents for PRF
            
        Returns:
            List of expansion terms
        """
        pass


class IdentityExpander(QueryExpander):
    """No-op expander that returns no additional terms."""
    
    def expand(self, query: str, top_docs: Optional[List[str]] = None) -> List[str]:
        return []


class SynonymExpander(QueryExpander):
    """
    Expand queries with synonyms from WordNet or similar.
    
    Useful for improving recall on lexical retrieval (BM25).
    """
    
    def __init__(self, max_synonyms: int = 3):
        self.max_synonyms = max_synonyms
        
    def expand(self, query: str, top_docs: Optional[List[str]] = None) -> List[str]:
        """Expand query with synonyms."""
        
        # TODO: Implement synonym lookup (WordNet, custom dictionary)
        logger.warning("SynonymExpander not fully implemented")
        return []


class PRFExpander(QueryExpander):
    """
    Pseudo-Relevance Feedback (PRF) expansion.
    
    Extracts important terms from top retrieved documents and adds them to query.
    Uses TF-IDF scoring to identify the most discriminative terms.
    """
    
    def __init__(
        self,
        num_docs: int = 3,
        num_terms: int = 5,
        min_term_freq: int = 2,
        stopwords: Optional[Set[str]] = None
    ):
        self.num_docs = num_docs
        self.num_terms = num_terms
        self.min_term_freq = min_term_freq
        
        # Default English stopwords
        if stopwords is None:
            self.stopwords = {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                'to', 'was', 'will', 'with', 'this', 'but', 'they', 'have', 'had',
                'what', 'when', 'where', 'who', 'which', 'why', 'how', 'can', 'could',
                'would', 'should', 'may', 'might', 'must', 'shall', 'or', 'not', 'no',
                'yes', 'do', 'does', 'did', 'been', 'being', 'am', 'were', 'you', 'your'
            }
        else:
            self.stopwords = stopwords
        
    def expand(self, query: str, top_docs: Optional[List[str]] = None) -> List[str]:
        """
        Extract expansion terms from top documents using TF-IDF.
        
        Args:
            query: Original query
            top_docs: Top retrieved documents (text content)
            
        Returns:
            List of expansion terms ranked by TF-IDF score
        """
        
        if not top_docs or len(top_docs) == 0:
            logger.warning("No documents provided for PRF expansion")
            return []
        
        # Use only top N documents
        docs_to_analyze = top_docs[:self.num_docs]
        
        # Extract query terms (to avoid re-adding them)
        query_terms = set(self._tokenize(query.lower()))
        
        # Score terms from documents
        term_scores = self._extract_terms(docs_to_analyze)
        
        # Filter out query terms, stopwords, and rare terms
        filtered_terms = [
            (term, score) for term, score in term_scores.items()
            if term not in query_terms 
            and term not in self.stopwords
            and len(term) > 2  # Skip very short terms
        ]
        
        # Sort by score and take top N
        top_terms = sorted(filtered_terms, key=lambda x: x[1], reverse=True)[:self.num_terms]
        
        # Return just the terms (not scores)
        expansion_terms = [term for term, score in top_terms]
        
        logger.debug(f"PRF expansion: {len(expansion_terms)} terms from {len(docs_to_analyze)} docs")
        return expansion_terms
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization by splitting on whitespace and punctuation."""
        import re
        # Split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _extract_terms(self, docs: List[str]) -> Dict[str, float]:
        """
        Extract and score terms from documents using TF-IDF.
        
        TF-IDF = TF(term, doc) * IDF(term, all_docs)
        TF = term_freq / max_term_freq in doc
        IDF = log(num_docs / num_docs_with_term)
        """
        import math
        from collections import Counter
        
        # Tokenize all documents
        doc_tokens = [self._tokenize(doc) for doc in docs]
        
        # Calculate document frequency (DF) for each term
        doc_frequency = Counter()
        for tokens in doc_tokens:
            unique_terms = set(tokens)
            for term in unique_terms:
                doc_frequency[term] += 1
        
        # Filter terms that appear too rarely
        valid_terms = {
            term for term, df in doc_frequency.items()
            if df >= self.min_term_freq or df >= max(1, len(docs) * 0.3)
        }
        
        # Calculate TF-IDF for each term across all documents
        term_tfidf = Counter()
        num_docs = len(docs)
        
        for tokens in doc_tokens:
            # Calculate TF for this document
            term_freq = Counter(tokens)
            max_freq = max(term_freq.values()) if term_freq else 1
            
            # Add TF-IDF contribution from this document
            for term, freq in term_freq.items():
                if term in valid_terms:
                    tf = freq / max_freq
                    idf = math.log(num_docs / doc_frequency[term])
                    term_tfidf[term] += tf * idf
        
        return dict(term_tfidf)


class MultiQueryExpander(QueryExpander):
    """
    Generate multiple query variations using LLM.
    
    Each variation targets a different aspect of the information need.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        num_queries: int = 3
    ):
        self.model_name = model_name
        self.num_queries = num_queries
        
        try:
            from openai import OpenAI
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                logger.warning("OPENAI_API_KEY not found. MultiQueryExpander will fail.")
                self.client = None
        except ImportError:
            logger.warning("OpenAI not installed.")
            self.client = None
        
    def expand(self, query: str, top_docs: Optional[List[str]] = None) -> List[str]:
        """
        Generate multiple query variations.
        
        Returns list of alternative queries (not expansion terms).
        """
        if not self.client:
            return [query]

        prompt = f"""Generate {self.num_queries} alternative queries for the following question. Each query should target a different aspect or rephrasing.
        
Original: {query}

Alternative queries (one per line, no numbering):"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates diverse search queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                n=1
            )
            content = response.choices[0].message.content.strip()
            queries = [line.strip() for line in content.split('\n') if line.strip()]
            return queries[:self.num_queries]
        except Exception as e:
            logger.error(f"MultiQuery expansion failed: {e}")
            return [query]


class DomainExpander(QueryExpander):
    """
    Domain-specific query expansion.
    
    Uses domain-specific terminology and acronyms to expand queries.
    """
    
    def __init__(self, domain: str, expansion_dict: Optional[Dict[str, List[str]]] = None):
        self.domain = domain
        self.expansion_dict = expansion_dict or self._load_default_expansions(domain)
        
    def expand(self, query: str, top_docs: Optional[List[str]] = None) -> List[str]:
        """Expand with domain-specific terms."""
        
        query_lower = query.lower()
        expansions = []
        
        for term, synonyms in self.expansion_dict.items():
            if term in query_lower:
                expansions.extend(synonyms)
        
        return list(set(expansions))  # Remove duplicates
    
    def _load_default_expansions(self, domain: str) -> Dict[str, List[str]]:
        """Load default expansion dictionary for domain."""
        
        # Example domain-specific expansions
        DOMAIN_EXPANSIONS = {
            "cloud": {
                "aws": ["amazon web services", "ec2", "s3"],
                "gcp": ["google cloud platform", "compute engine"],
                "azure": ["microsoft azure", "azure cloud"],
            },
            "fiqa": {
                "401k": ["retirement account", "pension"],
                "ira": ["individual retirement account"],
                "etf": ["exchange traded fund"],
            },
            "govt": {
                "dhs": ["department of homeland security"],
                "fbi": ["federal bureau of investigation"],
                "irs": ["internal revenue service"],
            },
        }
        
        return DOMAIN_EXPANSIONS.get(domain, {})


class BackTranslationExpander(QueryExpander):
    """
    Back-translation for query expansion.
    
    Translates query to another language and back to generate paraphrases.
    """
    
    def __init__(self, intermediate_lang: str = "es"):
        self.intermediate_lang = intermediate_lang
        
    def expand(self, query: str, top_docs: Optional[List[str]] = None) -> List[str]:
        """Generate paraphrases via back-translation."""
        
        # TODO: Implement translation (using Opus-MT or similar)
        logger.warning("BackTranslationExpander not fully implemented")
        return []


def get_expander(expander_type: str, **kwargs) -> QueryExpander:
    """
    Factory function to get a query expander.
    
    Args:
        expander_type: Type of expander ('identity', 'synonym', 'prf', 'multi', 'domain', 'backtrans')
        **kwargs: Additional arguments for the expander
        
    Returns:
        QueryExpander instance
    """
    
    expanders = {
        'identity': IdentityExpander,
        'synonym': SynonymExpander,
        'prf': PRFExpander,
        'multi': MultiQueryExpander,
        'domain': DomainExpander,
        'backtrans': BackTranslationExpander,
    }
    
    if expander_type not in expanders:
        raise ValueError(f"Unknown expander type: {expander_type}. Available: {list(expanders.keys())}")
    
    return expanders[expander_type](**kwargs)
