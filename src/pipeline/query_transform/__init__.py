"""
Query transformation module.

Provides query rewriting, expansion, and decomposition strategies.
"""

from .rewriters import (
    QueryRewriter,
    IdentityRewriter,
    LLMRewriter,
    VLLMRewriter,
    QueryDecomposer,
    ContextualRewriter,
    TemplateRewriter,
    HyDERewriter,
    get_rewriter,
)

from .expansion import (
    QueryExpander,
    IdentityExpander,
    SynonymExpander,
    PRFExpander,
    MultiQueryExpander,
    DomainExpander,
    BackTranslationExpander,
    get_expander,
)

__all__ = [
    # Rewriters
    'QueryRewriter',
    'IdentityRewriter',
    'LLMRewriter',
    'QueryDecomposer',
    'ContextualRewriter',
    'TemplateRewriter',
    'HyDERewriter',
    'get_rewriter',
    # Expanders
    'QueryExpander',
    'IdentityExpander',
    'SynonymExpander',
    'PRFExpander',
    'MultiQueryExpander',
    'DomainExpander',
    'BackTranslationExpander',
    'get_expander',
]
