# Experiment Configurations

Organized by experiment category for better maintainability.

## 📁 Directory Structure

```
experiments/
├── baselines/          # A0-A1: Lower bound experiments
├── query/              # A2-A4: Query processing variations
├── hybrid/             # A5-A7: Hybrid retrieval strategies
├── rerank/             # A9: Reranking approaches
├── finetune/           # A10-A11: Fine-tuned models
└── iterative/          # A8: Multi-round refinement
```

## 📋 Configuration Categories

### Baselines (`baselines/`)
Establish lower bounds without advanced techniques:
- **A0_baseline_sparse**: BM25/ELSER/SPLADE only
- **A1_dense_baseline**: BGE-M3 dense retrieval only

### Query Processing (`query/`)
Evaluate query transformation strategies:
- **A2_rewrite_splade**: Single query rewrite + SPLADE
- **A3_rewrite_multi**: Multi-variant rewriting + RRF fusion
- **A4_rewrite_splade**: Enhanced rewrite + SPLADE

### Hybrid Retrieval (`hybrid/`)
Combine sparse and dense methods:
- **A5_hybrid_sparse_dense**: SPLADE + BGE-M3 with RRF
- **A6_hybrid_rerank**: Hybrid + cross-encoder reranker
- **A7_domain_gated**: A6 + domain-specific rules

### Reranking (`rerank/`)
Advanced reranking techniques:
- **A9_colbert_rerank**: ColBERT late-interaction reranker

### Fine-tuned Models (`finetune/`)
Domain-adapted models:
- **A10_finetuned_reranker**: Domain fine-tuned cross-encoder
- **A11_finetuned_splade**: Domain fine-tuned SPLADE

### Iterative Refinement (`iterative/`)
Multi-round query processing:
- **A8_iterative_refinement**: Query refinement based on initial results

## 🔧 Usage

Configs are automatically loaded by experiment name:

```python
# Script automatically finds: configs/experiments/baselines/A0_baseline_sparse.yaml
python scripts/run_experiment.py --experiment A0_baseline_sparse --domain clapnq
```

## 📝 Configuration Format

Each config file specifies:
- Which modules to activate/deactivate
- Model selections
- Hyperparameters
- Processing strategies

Example:
```yaml
# A6_hybrid_rerank.yaml
modules:
  query_rewrite: true
  sparse_retrieval: splade
  dense_retrieval: bge-m3
  fusion: rrf
  reranking: cross-encoder

parameters:
  sparse_weight: 0.6
  dense_weight: 0.4
  rerank_top_k: 10
```

## 🔗 Related

- **configs/base.yaml**: Global defaults
- **configs/domains/**: Domain-specific overrides
- **experiments/**: Results organized by experiment ID
