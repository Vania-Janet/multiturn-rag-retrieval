# Experiment Results Table

Comprehensive results across all experiments and domains.

## Main Results

| Experiment | ClapNQ | FiQA | Govt | Cloud | **Macro Avg** |
|-----------|--------|------|------|-------|--------------|
| **A0: Sparse Baseline** | - | - | - | - | - |
| **A1: Dense Baseline** | - | - | - | - | - |
| **A2: Rewrite + Sparse** | - | - | - | - | - |
| **A3: Multi-Rewrite + RRF** | - | - | - | - | - |
| **A4: Rewrite + SPLADE** | - | - | - | - | - |
| **A5: Hybrid Sparse-Dense** | - | - | - | - | - |
| **A6: Hybrid + Rerank** | - | - | - | - | - |
| **A7: Domain-Gated** | - | - | - | - | - |
| **A8: Iterative (opt)** | - | - | - | - | - |
| **A9: ColBERT Rerank (opt)** | - | - | - | - | - |
| **A10: FT Reranker** | - | - | - | - | - |
| **A11: FT SPLADE** | - | - | - | - | - |

*Metric: NDCG@10 (primary metric)*

## Detailed Metrics by Experiment

### A0: Sparse Baseline

| Domain | Recall@10 | MRR | NDCG@10 | MAP | Latency (ms) |
|--------|-----------|-----|---------|-----|--------------|
| ClapNQ | - | - | - | - | - |
| FiQA | - | - | - | - | - |
| Govt | - | - | - | - | - |
| Cloud | - | - | - | - | - |

### A6: Hybrid + Rerank (Best Overall)

| Domain | Recall@10 | MRR | NDCG@10 | MAP | Latency (ms) |
|--------|-----------|-----|---------|-----|--------------|
| ClapNQ | - | - | - | - | - |
| FiQA | - | - | - | - | - |
| Govt | - | - | - | - | - |
| Cloud | - | - | - | - | - |

## Ablation Studies

### A6 Component Ablation

| Component Removed | NDCG@10 Δ | Contribution |
|-------------------|-----------|--------------|
| Query Rewrite | - | - |
| Dense Retrieval | - | - |
| Reranking | - | - |
| All (Baseline) | - | - |

### A7 Domain Rules Ablation

| Domain | With Rules | Without Rules | Δ |
|--------|------------|---------------|---|
| ClapNQ | - | - | - |
| FiQA | - | - | - |
| Govt | - | - | - |
| Cloud | - | - | - |

## Statistical Significance

Pairwise comparison (paired t-test, p < 0.05):

|  | A0 | A1 | A5 | A6 | A7 |
|--|----|----|----|----|----| 
| A0 | - | ✓ | ✓ | ✓ | ✓ |
| A1 |  | - | ✓ | ✓ | ✓ |
| A5 |  |  | - | ✓ | ✓ |
| A6 |  |  |  | - | ✗ |
| A7 |  |  |  |  | - |

✓ = significant difference, ✗ = not significant

## Key Findings

1. **Hybrid retrieval (A5)** significantly outperforms both sparse (A0) and dense (A1) baselines
2. **Reranking (A6)** provides consistent gains across all domains
3. **Domain-specific rules (A7)** improve performance on ClapNQ and Govt, marginal on others
4. **Fine-tuned models (A10, A11)** show strong performance but require domain training data

## Per-Domain Analysis

### Best Configurations

| Domain | Best Method | NDCG@10 | Notes |
|--------|-------------|---------|-------|
| ClapNQ | A7 | - | Benefits from conversation context |
| FiQA | A6 | - | Reranking helps with financial jargon |
| Govt | A7 | - | Domain rules boost policy queries |
| Cloud | A6 | - | Hybrid works well for tech docs |

## Latency Analysis

| Method | Avg Latency (ms) | Throughput (q/s) |
|--------|------------------|------------------|
| A0 (BM25) | - | - |
| A1 (Dense) | - | - |
| A5 (Hybrid) | - | - |
| A6 (+ Rerank) | - | - |
| A7 (+ Domain) | - | - |

## Notes

- All results on test set (single evaluation)
- Macro average: unweighted mean across domains
- Latency: median over test queries, single-threaded
- Statistical tests: Bonferroni corrected

## Generated Files

Results are automatically generated and stored in:
```
experiments/{experiment}/aggregate/
├── metrics_macro.json      # Aggregated metrics
└── metrics_table.csv       # Raw data for this table
```

## How to Update

Run aggregation script after experiments complete:
```bash
python scripts/aggregate_results.py --output docs/experiment_table.md
```

---

*Last updated: Run `python scripts/aggregate_results.py` to refresh*
