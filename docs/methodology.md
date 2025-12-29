# Methodology

## Overview

This document describes the experimental methodology for our multi-domain RAG evaluation framework.

## Research Questions

1. **RQ1**: How do different retrieval methods (sparse, dense, hybrid) perform across diverse domains?
2. **RQ2**: What is the impact of query rewriting on retrieval effectiveness?
3. **RQ3**: How much improvement does reranking provide over initial retrieval?
4. **RQ4**: Can domain-specific fine-tuning improve cross-domain generalization?

## Experimental Design

### Datasets

We evaluate on four domains:
- **ClapNQ**: Conversational Q&A with contextual dependencies
- **FiQA**: Financial domain questions
- **Government**: Policy and regulation queries
- **Cloud**: Technical documentation Q&A

### Data Splits

All experiments use fixed 80/10/10 splits:
- **Train (80%)**: Used for fine-tuning and parameter selection
- **Val (10%)**: Used for hyperparameter tuning
- **Test (10%)**: Held-out for final evaluation (pseudo-test)

**Critical**: Conversation-level splits for ClapNQ to prevent leakage across dialogue turns.

### Baseline Systems

#### A0: Sparse Baseline
- Methods: BM25, ELSER, SPLADE
- No query processing
- Establishes lexical retrieval lower bound

#### A1: Dense Baseline
- Method: BGE-M3
- No query processing
- Establishes semantic retrieval lower bound

### Query Processing Experiments

#### A2: Single Query Rewrite
- LLM-based query expansion
- Single rewrite variant per query
- Retrieval: SPLADE

#### A3: Multi-Variant Rewriting
- Generate N query variants
- Fusion: Reciprocal Rank Fusion (RRF)
- Retrieval: BM25

### Hybrid Retrieval Experiments

#### A5: Sparse-Dense Hybrid
- Sparse: SPLADE
- Dense: BGE-M3
- Fusion: RRF with k=60

#### A6: Hybrid + Reranking
- Initial retrieval: A5
- Reranker: Cross-encoder (ms-marco-MiniLM-L-6-v2)
- Top-k reranking: 100 → 10

#### A7: Domain-Gated Pipeline
- Based on A6
- Domain-specific rules and parameters
- Configured per domain in `configs/domains/`

### Advanced Experiments

#### A8: Iterative Refinement (Optional)
- Multi-round query refinement
- Uses initial results to improve query

#### A9: ColBERT Reranking (Optional)
- Late interaction reranking
- Token-level matching

#### A10-A11: Fine-tuned Models
- Domain-adapted reranker (A10)
- Domain-adapted SPLADE (A11)

## Configuration System

### Hierarchical Configs

Experiments load configurations in order:
1. `configs/base.yaml` - Global defaults
2. `configs/domains/{domain}.yaml` - Domain overrides
3. `configs/experiments/{category}/{experiment}.yaml` - Experiment settings

### Reproducibility

All configs specify:
- Random seeds
- Model versions
- Hyperparameters
- Processing steps

## Evaluation Metrics

### Retrieval Metrics
- **Recall@k** (k=5,10,20,100): Coverage of relevant documents
- **MRR**: Mean Reciprocal Rank
- **NDCG@k** (k=5,10,20): Ranking quality
- **MAP**: Mean Average Precision

### Generation Metrics (if applicable)
- **ROUGE-L**: Token overlap with reference
- **BERTScore**: Semantic similarity
- **Exact Match**: Strict answer matching

### Efficiency Metrics
- **Latency**: End-to-end query time (ms)
- **Throughput**: Queries per second

## Statistical Testing

- Paired t-tests between experiment pairs
- Bonferroni correction for multiple comparisons
- Significance threshold: p < 0.05

## Domain-Specific Parameters

Parameters are fixed a priori per domain, not tuned on test:

**ClapNQ** (Conversational):
- Higher context window for multi-turn
- Error code detection enabled
- Sparse weight: 0.6, Dense: 0.4

**FiQA** (Financial):
- Entity boosting for financial terms
- Sparse weight: 0.5, Dense: 0.5

**Government** (Policy):
- Section-aware chunking
- Sparse weight: 0.7, Dense: 0.3

**Cloud** (Technical):
- Code-aware processing
- Sparse weight: 0.6, Dense: 0.4

## Ablation Studies

For key experiments (A6, A7), we conduct ablations:
1. Remove query rewriting
2. Remove fusion (sparse only)
3. Remove reranking
4. Remove domain-specific rules (A7)

## Reproducibility Checklist

- [ ] Fixed random seeds
- [ ] Deterministic GPU operations
- [ ] Versioned dependencies
- [ ] Saved configurations
- [ ] Logged hyperparameters
- [ ] Documented data splits

## References

[To be filled with relevant papers]
