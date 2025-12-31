"""
Query rewriting strategies for improving retrieval.

This module implements various query transformation techniques including:
- LLM-based query rewriting
- Query decomposition
- Multi-query generation
- Back-translation
"""

import logging
import os
import json
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
from dotenv import load_dotenv

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

load_dotenv()
logger = logging.getLogger(__name__)

CACHE_DIR = Path(".cache/rewrites")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_cache_key(prefix: str, data: Dict[str, Any]) -> str:
    """Generate a stable cache key from a dictionary of data."""
    # Sort keys to ensure stability
    serialized = json.dumps(data, sort_keys=True)
    hash_obj = hashlib.md5(serialized.encode('utf-8'))
    return f"{prefix}_{hash_obj.hexdigest()}"

class QueryRewriter(ABC):
    """Base class for query rewriting strategies."""
    
    @abstractmethod
    def rewrite(self, query: str, context: Optional[List[str]] = None) -> List[str]:
        """
        Rewrite a query into one or more variations.
        
        Args:
            query: Original query string
            context: Optional conversation context
            
        Returns:
            List of rewritten queries
        """
        pass


class IdentityRewriter(QueryRewriter):
    """No-op rewriter that returns the original query."""
    
    def rewrite(self, query: str, context: Optional[List[str]] = None) -> List[str]:
        return [query]


class LLMRewriter(QueryRewriter):
    """
    LLM-based query rewriting.
    
    Uses a language model to rephrase queries for better retrieval.
    """
    
    SYSTEM_PROMPT = """You are an information retrieval assistant.
Your task is to rewrite conversational user questions into standalone,
search-optimized queries suitable for document retrieval.

Follow these rules strictly:
- Preserve the original user intent exactly.
- Resolve all coreferences and implicit references using the conversation history.
- Do NOT add new facts, assumptions, or external knowledge.
- Do NOT answer the question.
- Do NOT explain your reasoning.
- Output ONLY the rewritten query as a single sentence."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_rewrites: int = 1,
        top_p: float = 1.0,
        max_tokens: int = 100
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_rewrites = max_rewrites
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI not installed. Install with: pip install openai")
            self.client = None
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not found in environment")
                self.client = None
            else:
                self.client = OpenAI(api_key=api_key)
        
    def rewrite(self, query: str, context: Optional[List[str]] = None) -> List[str]:
        """Generate query variations using LLM."""
        
        if not self.client:
            logger.warning("OpenAI client not available - returning original query")
            return [query]
        
        # Check cache
        cache_data = {
            "type": "llm_rewriter",
            "model": self.model_name,
            "temp": self.temperature,
            "max_rewrites": self.max_rewrites,
            "query": query,
            "context": context,
            "system_prompt": self.SYSTEM_PROMPT
        }
        cache_key = get_cache_key("llm", cache_data)
        cache_file = CACHE_DIR / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_result = json.load(f)
                logger.debug(f"Cache hit for rewrite: {cache_key}")
                return cached_result
            except Exception as e:
                logger.warning(f"Failed to read cache {cache_file}: {e}")

        try:
            user_prompt = self._build_user_prompt(query, context)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                n=1
            )
            
            content = response.choices[0].message.content.strip()
            
            result = []
            if self.max_rewrites == 1:
                result = [content]
            else:
                queries = [line.strip() for line in content.split('\n') if line.strip()]
                result = queries[:self.max_rewrites] if queries else [query]
            
            # Save to cache
            try:
                with open(cache_file, 'w') as f:
                    json.dump(result, f)
            except Exception as e:
                logger.warning(f"Failed to write cache {cache_file}: {e}")
                
            return result
                
        except Exception as e:
            logger.error(f"LLM rewriting failed: {e}")
            return [query]
    
    def _build_user_prompt(self, query: str, context: Optional[List[str]] = None) -> str:
        """Build user prompt for query rewriting."""
        
        conversation_history = ""
        if context and len(context) > 0:
            conversation_history = "\n".join(context[-6:])
        else:
            conversation_history = "(No prior conversation)"
        
        if self.max_rewrites == 1:
            return f"""Conversation history:
{conversation_history}

Current user question:
{query}

Rewrite the current question into a standalone, context-complete query
that can be used directly for document retrieval."""
        else:
            return f"""Conversation history:
{conversation_history}

Current user question:
{query}

Generate {self.max_rewrites} different rewritten standalone queries that preserve the same intent
but vary in phrasing or focus. Each query must be suitable for document retrieval.
Return each query on a separate line. Do not number them."""


class QueryDecomposer(QueryRewriter):
    """
    Decompose complex queries into simpler sub-queries.
    
    Useful for multi-hop questions that require multiple retrieval steps.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        
    def rewrite(self, query: str, context: Optional[List[str]] = None) -> List[str]:
        """Decompose complex query into sub-queries."""
        
        # TODO: Implement decomposition logic
        logger.warning("QueryDecomposer not fully implemented - returning original query")
        return [query]


class ContextualRewriter(QueryRewriter):
    """
    Add conversational context to standalone queries.
    
    Resolves pronouns and references from previous turns.
    """
    
    SYSTEM_PROMPT = """You are an information retrieval assistant.
Your task is to rewrite conversational user questions into standalone,
search-optimized queries suitable for document retrieval.

Follow these rules strictly:
- Preserve the original user intent exactly.
- Resolve all coreferences and implicit references using the conversation history.
- Do NOT add new facts, assumptions, or external knowledge.
- Do NOT answer the question.
- Do NOT explain your reasoning.
- Output ONLY the rewritten query as a single sentence."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 100,
        include_context: bool = True,
        context_turns: int = 3
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.include_context = include_context
        self.context_turns = context_turns
        
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI not installed. Install with: pip install openai")
            self.client = None
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not found in environment")
                self.client = None
            else:
                self.client = OpenAI(api_key=api_key)
        
    def rewrite(self, query: str, context: Optional[List[str]] = None) -> List[str]:
        """
        Rewrite query to be self-contained given conversation context.
        
        Example:
            Context: ["What is the capital of France?", "Paris"]
            Query: "What's its population?"
            Output: ["What is the population of Paris?"]
        """
        
        if not self.include_context or not context or len(context) == 0:
            return [query]
        
        if not self.client:
            logger.warning("OpenAI client not available - returning original query")
            return [query]
            
        # Check cache
        cache_data = {
            "type": "contextual_rewriter",
            "model": self.model_name,
            "temp": self.temperature,
            "query": query,
            "context": context,
            "system_prompt": self.SYSTEM_PROMPT
        }
        cache_key = get_cache_key("contextual", cache_data)
        cache_file = CACHE_DIR / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_result = json.load(f)
                logger.debug(f"Cache hit for contextual rewrite: {cache_key}")
                return cached_result
            except Exception as e:
                logger.warning(f"Failed to read cache {cache_file}: {e}")
        
        try:
            conversation_history = "\n".join(context[-self.context_turns * 2:])
            
            user_prompt = f"""Conversation history:
{conversation_history}

Current user question:
{query}

Rewrite the current question into a standalone, context-complete query
that can be used directly for document retrieval."""
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                n=1
            )
            
            rewritten = response.choices[0].message.content.strip()
            result = [rewritten]
            
            # Save to cache
            try:
                with open(cache_file, 'w') as f:
                    json.dump(result, f)
            except Exception as e:
                logger.warning(f"Failed to write cache {cache_file}: {e}")
                
            return result
            
        except Exception as e:
            logger.error(f"Contextual rewriting failed: {e}")
            return [query]


class TemplateRewriter(QueryRewriter):
    """
    Rule-based query rewriting using templates.
    
    Fast and deterministic alternative to LLM-based rewriting.
    """
    
    TEMPLATES = [
        "What is {query}?",
        "Tell me about {query}",
        "Explain {query}",
        "Information about {query}",
    ]
    
    def __init__(self, templates: Optional[List[str]] = None):
        self.templates = templates or self.TEMPLATES
        
    def rewrite(self, query: str, context: Optional[List[str]] = None) -> List[str]:
        """Generate variations using templates."""
        
        rewrites = [query]  # Always include original
        
        # Apply templates
        for template in self.templates[:2]:  # Limit to 2 templates
            if "{query}" in template:
                rewrites.append(template.format(query=query))
        
        return rewrites


class HyDERewriter(QueryRewriter):
    """
    Hypothetical Document Embeddings (HyDE).
    
    Generates a hypothetical answer to the query, then uses that for retrieval.
    Instead of searching with the question, we search with a hypothetical document
    that would answer it, which often yields better semantic matching.
    
    Reference: https://arxiv.org/abs/2212.10496
    """
    
    SYSTEM_PROMPT = """You are a knowledgeable assistant that writes informative passages.
Your task is to write a concise, factual passage that would directly answer the given question.

Follow these rules strictly:
- Write as if you are a document that contains the answer.
- Be specific and factual.
- Keep the passage between 2-4 sentences.
- Use natural language suitable for document retrieval.
- Do NOT include phrases like "The answer is" or "According to".
- Do NOT use conversational language.
- Write in a neutral, encyclopedic style."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 150
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI not installed. Install with: pip install openai")
            self.client = None
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not found in environment")
                self.client = None
            else:
                self.client = OpenAI(api_key=api_key)
        
    def rewrite(self, query: str, context: Optional[List[str]] = None) -> List[str]:
        """
        Generate hypothetical document that would answer the query.
        
        Returns the generated hypothetical passage for retrieval.
        """
        
        if not self.client:
            logger.warning("OpenAI client not available - returning original query")
            return [query]
            
        # Check cache
        cache_data = {
            "type": "hyde_rewriter",
            "model": self.model_name,
            "temp": self.temperature,
            "query": query,
            "context": context,
            "system_prompt": self.SYSTEM_PROMPT
        }
        cache_key = get_cache_key("hyde", cache_data)
        cache_file = CACHE_DIR / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_result = json.load(f)
                logger.debug(f"Cache hit for HyDE: {cache_key}")
                return cached_result
            except Exception as e:
                logger.warning(f"Failed to read cache {cache_file}: {e}")
        
        try:
            # Build user prompt
            user_prompt = f"""Question: {query}

Write a passage that would answer this question."""
            
            # If context provided, resolve references first
            if context and len(context) > 0:
                context_str = "\n".join(context[-4:])
                user_prompt = f"""Conversation context:
{context_str}

Question: {query}

Write a passage that would answer this question, taking into account the conversation context."""
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                n=1
            )
            
            hypothetical_doc = response.choices[0].message.content.strip()
            result = [hypothetical_doc]
            
            # Save to cache
            try:
                with open(cache_file, 'w') as f:
                    json.dump(result, f)
            except Exception as e:
                logger.warning(f"Failed to write cache {cache_file}: {e}")
                
            return result
            
        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            return [query]


def get_rewriter(rewriter_type: str, **kwargs) -> QueryRewriter:
    """
    Factory function to get a query rewriter.
    
    Args:
        rewriter_type: Type of rewriter ('identity', 'llm', 'decompose', 'contextual', 'template', 'hyde')
        **kwargs: Additional arguments for the rewriter
        
    Returns:
        QueryRewriter instance
    """
    
    rewriters = {
        'identity': IdentityRewriter,
        'llm': LLMRewriter,
        'decompose': QueryDecomposer,
        'contextual': ContextualRewriter,
        'template': TemplateRewriter,
        'hyde': HyDERewriter,
    }
    
    if rewriter_type not in rewriters:
        raise ValueError(f"Unknown rewriter type: {rewriter_type}. Available: {list(rewriters.keys())}")
    
    return rewriters[rewriter_type](**kwargs)
