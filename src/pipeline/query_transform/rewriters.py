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

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

load_dotenv()
logger = logging.getLogger(__name__)

# Set HuggingFace cache to workspace volume (100 GB) instead of container disk (20 GB)
os.environ['HF_HOME'] = '/workspace/cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/workspace/cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/cache/transformers'
os.environ['HF_HUB_CACHE'] = '/workspace/cache/huggingface'

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
    LLM-based query rewriting (Query Condensation).
    
    Uses a language model to condense the conversation history and current query
    into a single standalone search query.
    """
    
    SYSTEM_PROMPT = """You are a helpful assistant that resolves coreferences in conversational search queries.

Your task: Rewrite the LAST user turn into a standalone query by replacing pronouns and vague references with their explicit entities from the conversation history.

CRITICAL RULES:
1. START with the LAST user turn - preserve its complete content, wording, and structure
2. ONLY replace pronouns (it, that, this, they) and vague references with explicit entities from prior turns
3. DO NOT summarize, shorten, or paraphrase the last turn
4. DO NOT omit any details from the last turn (error messages, codes, specific problems, technical terms)
5. If the last turn mentions a specific error, problem, or technical detail, it MUST appear in the output
6. Preserve ALL keywords exactly: technical terms, acronyms, product names, error messages, version numbers
7. Add context from prior turns ONLY if needed to clarify pronouns - do not merge multiple questions

Example:
Turn 1: "How do I configure auto-scaling in AWS?"
Turn 2: "What are the pricing options for it?"
Output: "What are the pricing options for auto-scaling in AWS?"

DO NOT answer the question. Output ONLY the standalone query."""
    
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

Current user question (LAST TURN):
{query}

Rewrite the LAST TURN into a standalone query by replacing pronouns with explicit entities from history.
DO NOT summarize or omit any details from the last turn. Preserve all error messages, technical terms, and specific problems."""
        else:
            return f"""Conversation history:
{conversation_history}

Current user question:
{query}

Generate {self.max_rewrites} different rewritten standalone queries that preserve the same intent but use these variation strategies:
1. Use synonyms and semantically related terms (e.g., "machine learning" → "ML algorithms")
2. Expand with specific aspects or dimensions (e.g., "performance" → "speed and accuracy")
3. Reformulate with different linguistic structure (e.g., active/passive voice, questions/statements)

IMPORTANT: While varying the linguistic style, preserve ALL technical terms, acronyms, product names, and specific keywords exactly as written.
Each query must be standalone and suitable for document retrieval.
Return each query on a separate line. Do not number them."""


class VLLMRewriter(QueryRewriter):
    """
    vLLM-based query rewriting (Query Condensation and Multi-Query Generation).
    
    Uses a local vLLM server to condense the conversation history and current query
    into standalone search queries, optimized for sparse or dense retrieval.
    """
    
    # Condensation prompts optimized by retrieval type
    CONDENSATION_PROMPT_SPARSE = """You are a helpful assistant that resolves coreferences in conversational search queries for KEYWORD-BASED SEARCH (BM25/SPLADE).

Your task: Rewrite the LAST user turn into a standalone query by replacing pronouns and vague references with their explicit entities from the conversation history.

CRITICAL RULES FOR KEYWORD SEARCH:
1. START with the LAST user turn - preserve its complete content, wording, and structure
2. ONLY replace pronouns (it, that, this, they) and vague references with explicit entities from prior turns
3. PRESERVE ALL technical terms, acronyms, product names, error messages, version numbers, and domain-specific keywords EXACTLY
4. ADD relevant keywords from context that improve lexical matching (e.g., if discussing "AWS EC2", include both terms)
5. DO NOT summarize, shorten, or paraphrase - keep all specific details from the last turn
6. If the last turn mentions a specific error, problem, or technical detail, it MUST appear in the output
7. Expand acronyms if their full form appears in context and helps matching (e.g., "ML" → "machine learning ML")
8. Include synonyms and related technical terms when they appear in context (e.g., "API" and "endpoint")

Example:
Turn 1: "How do I configure auto-scaling in AWS EC2?"
Turn 2: "What are the pricing options for it?"
Output: "What are the pricing options for auto-scaling in AWS EC2?"

DO NOT answer the question. Output ONLY the standalone query with preserved keywords."""

    CONDENSATION_PROMPT_DENSE = """You are a helpful assistant that resolves coreferences in conversational search queries for SEMANTIC SEARCH (Dense Embeddings).

Your task: Rewrite the LAST user turn into a standalone query by replacing pronouns and vague references with their explicit entities from the conversation history.

CRITICAL RULES FOR SEMANTIC SEARCH:
1. START with the LAST user turn - preserve its complete content, wording, and structure
2. ONLY replace pronouns (it, that, this, they) and vague references with explicit entities from prior turns
3. DO NOT summarize, shorten, or paraphrase the last turn
4. DO NOT omit any details from the last turn (error messages, codes, specific problems, technical terms)
5. Preserve ALL keywords exactly: technical terms, acronyms, product names, error messages, version numbers
6. Add context from prior turns ONLY if needed to clarify pronouns - do not merge multiple questions
7. Focus on preserving semantic meaning and intent rather than adding extra keywords

Example:
Turn 1: "How do I configure auto-scaling in AWS?"
Turn 2: "What are the pricing options for it?"
Output: "What are the pricing options for auto-scaling in AWS?"

DO NOT answer the question. Output ONLY the standalone query."""

    MULTIQUERY_PROMPT_SPARSE = """You are a helpful assistant that generates diverse search queries for KEYWORD-BASED SEARCH (BM25/SPLADE).

Your task: Generate {num_queries} different standalone queries that use DIFFERENT KEYWORDS and TERMS to find relevant documents.

CRITICAL RULES FOR KEYWORD SEARCH:
1. Each query should use DIFFERENT technical terms, synonyms, and related keywords
2. Vary acronyms vs. full names (e.g., "API" vs "Application Programming Interface")
3. Include domain-specific terminology and jargon variations
4. Use different phrasing that changes the keywords substantially
5. Each query must be standalone and include necessary context
6. Preserve all technical accuracy - don't invent terms
7. Focus on LEXICAL DIVERSITY - queries should have minimal keyword overlap

Strategies:
- Query 1: Use primary technical terms and acronyms
- Query 2: Expand acronyms and use formal terminology
- Query 3: Use related concepts and synonymous technical terms

Example:
Original: "How to configure Kubernetes autoscaling?"

Query 1: How to configure Kubernetes autoscaling?
Query 2: How to set up horizontal pod autoscaler HPA in K8s?
Query 3: Kubernetes automatic scaling configuration pods workloads?

DO NOT answer the question. Output ONLY {num_queries} queries, one per line, without numbering."""

    MULTIQUERY_PROMPT_DENSE = """You are a helpful assistant that generates diverse search queries for SEMANTIC SEARCH (Dense Embeddings).

Your task: Generate {num_queries} different standalone queries that capture DIFFERENT SEMANTIC ASPECTS of the information need.

CRITICAL RULES FOR SEMANTIC SEARCH:
1. Each query should explore a different facet or angle of the question
2. Vary the semantic focus (e.g., what/why/how, concepts vs. procedures, problems vs. solutions)
3. Use natural language with different framings
4. Each query must be standalone and include necessary context
5. Preserve technical terms exactly - don't paraphrase technical keywords
6. Focus on SEMANTIC DIVERSITY - different perspectives on the same topic

Strategies:
- Query 1: Direct question (what/how/why)
- Query 2: Problem-solution framing
- Query 3: Alternative perspective or related aspect

Example:
Original: "How does Kubernetes autoscaling work?"

Query 1: How does Kubernetes autoscaling work?
Query 2: What mechanisms does Kubernetes use to automatically scale applications?
Query 3: Kubernetes horizontal pod autoscaler behavior and scaling decisions?

DO NOT answer the question. Output ONLY {num_queries} queries, one per line, without numbering."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        temperature: float = 0.0,
        max_rewrites: int = 1,
        max_tokens: int = 100,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,
        max_model_len: int = 8192,
        retrieval_type: str = "dense"
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_rewrites = max_rewrites
        self.max_tokens = max_tokens
        self.retrieval_type = retrieval_type.lower()
        
        # Select appropriate system prompt based on retrieval type and task
        if max_rewrites == 1:
            # Condensation task
            if self.retrieval_type in ["sparse", "bm25", "splade"]:
                self.system_prompt = self.CONDENSATION_PROMPT_SPARSE
                logger.info("Using SPARSE-optimized condensation prompt for keyword search")
            else:
                self.system_prompt = self.CONDENSATION_PROMPT_DENSE
                logger.info("Using DENSE-optimized condensation prompt for semantic search")
        else:
            # Multi-query task
            if self.retrieval_type in ["sparse", "bm25", "splade"]:
                self.system_prompt = self.MULTIQUERY_PROMPT_SPARSE
                logger.info("Using SPARSE-optimized multi-query prompt for keyword search")
            else:
                self.system_prompt = self.MULTIQUERY_PROMPT_DENSE
                logger.info("Using DENSE-optimized multi-query prompt for semantic search")
        
        if not VLLM_AVAILABLE:
            logger.warning("vLLM not installed. Install with: pip install vllm")
            self.llm = None
        else:
            try:
                # Initialize vLLM
                self.llm = LLM(
                    model=model_name,
                    gpu_memory_utilization=gpu_memory_utilization,
                    trust_remote_code=True,
                    enforce_eager=True, # Often helps with memory
                    tensor_parallel_size=tensor_parallel_size,
                    quantization=quantization,
                    max_model_len=max_model_len
                )
                self.tokenizer = self.llm.get_tokenizer()
                self.sampling_params = SamplingParams(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=["<|eot_id|>"]
                )
            except Exception as e:
                logger.error(f"Failed to initialize vLLM: {e}")
                self.llm = None
    
    def rewrite(self, query: str, context: Optional[List[str]] = None) -> List[str]:
        """Generate query variations using LLM (single)."""
        return self.batch_rewrite([(query, context)])[0]
        
    def batch_rewrite(self, queries: List[tuple[str, Optional[List[str]]]]) -> List[List[str]]:
        """Batch rewrite multiple queries."""
        if not self.llm:
            return [[q] for q, c in queries]
        
        results = [None] * len(queries)
        prompts = []
        indices_to_process = []
        
        for idx, (query, context) in enumerate(queries):
            # Check cache
            cache_data = {
                "type": "vllm_rewriter",
                "model": self.model_name,
                "temp": self.temperature,
                "max_rewrites": self.max_rewrites,
                "retrieval_type": self.retrieval_type,
                "query": query,
                "context": context,
                "system_prompt": self.system_prompt
            }
            cache_key = get_cache_key("vllm", cache_data)
            cache_file = CACHE_DIR / f"{cache_key}.json"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        results[idx] = json.load(f)
                    continue
                except Exception:
                    pass
            
            # Prepare prompt
            user_prompt = self._build_user_prompt(query, context)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(full_prompt)
            indices_to_process.append((idx, cache_file))
            
        if prompts:
            logger.info(f"Running vLLM batch generation for {len(prompts)} queries...")
            outputs = self.llm.generate(prompts, self.sampling_params)
            
            for (idx, cache_file), output in zip(indices_to_process, outputs):
                generated_text = output.outputs[0].text.strip()
                result = []
                if self.max_rewrites == 1:
                    result = [generated_text]
                else:
                    lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
                    result = lines[:self.max_rewrites] if lines else [generated_text]
                
                results[idx] = result
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(result, f)
                except Exception as e:
                    logger.warning(f"Failed to write cache {cache_file}: {e}")
                    
        return results

    def _build_user_prompt(self, query: str, context: Optional[List[str]] = None) -> str:
        conversation_history = ""
        if context and len(context) > 0:
            conversation_history = "\n".join(context[-6:])
        else:
            conversation_history = "(No prior conversation)"
        
        if self.max_rewrites == 1:
            # Condensation
            return f"""Conversation history:
{conversation_history}

Current user question (LAST TURN):
{query}

Rewrite the LAST TURN into a standalone query by replacing pronouns with explicit entities from history.
DO NOT summarize or omit any details from the last turn. Preserve all error messages, technical terms, and specific problems."""
        else:
            # Multi-query generation
            return f"""Conversation history:
{conversation_history}

Current user question:
{query}

Generate {self.max_rewrites} diverse standalone queries following the strategies above.

Each query must be standalone and suitable for document retrieval."""


class QueryDecomposer(QueryRewriter):
    """
    Decompose complex queries into simpler sub-queries.
    
    Useful for multi-hop questions that require multiple retrieval steps.
    
    WARNING: This rewriter is NOT implemented. If you set rewriter_type='decompose'
    in your config, the system will silently return the original query unchanged.
    This class exists as a placeholder for future implementation.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        logger.warning(
            "QueryDecomposer is NOT IMPLEMENTED. "
            "Using this rewriter will return the original query unchanged. "
            "Consider using 'contextual', 'llm', or 'hyde' instead."
        )
        
    def rewrite(self, query: str, context: Optional[List[str]] = None) -> List[str]:
        """
        Decompose complex query into sub-queries.
        
        WARNING: NOT IMPLEMENTED - returns original query unchanged.
        """
        logger.warning(
            f"QueryDecomposer.rewrite() called but not implemented. "
            f"Returning original query: '{query[:50]}...'"
        )
        return [query]


class ContextualRewriter(QueryRewriter):
    """
    Add conversational context to standalone queries (Query Condensation).
    
    Condenses the conversation history and the current query into a single standalone query.
    """
    
    SYSTEM_PROMPT = """You are a helpful assistant that formulates standalone search queries.
Your task is to condense the conversation history and the latest user question into a single, standalone question that can be understood without the chat history.

Follow these rules strictly:
- The standalone question must fully represent the user's intent.
- Incorporate necessary context from previous turns to make the question self-contained.
- Do NOT answer the question.
- Do NOT explain your reasoning.
- Output ONLY the condensed, standalone question as a single sentence."""
    
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
            # Format conversation history with clear role labels
            # Assume context alternates: [user1, assistant1, user2, assistant2, ...]
            formatted_context = []
            recent_turns = context[-self.context_turns * 2:]
            
            for i, turn in enumerate(recent_turns):
                # Alternate between User and Assistant based on position
                role = "User" if i % 2 == 0 else "Assistant"
                formatted_context.append(f"{role}: {turn}")
            
            conversation_history = "\n".join(formatted_context) if formatted_context else "(No prior conversation)"
            
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
        'vllm': VLLMRewriter,
        'decompose': QueryDecomposer,
        'contextual': ContextualRewriter,
        'template': TemplateRewriter,
        'hyde': HyDERewriter,
    }
    
    if rewriter_type not in rewriters:
        raise ValueError(f"Unknown rewriter type: {rewriter_type}. Available: {list(rewriters.keys())}")
    
    return rewriters[rewriter_type](**kwargs)
