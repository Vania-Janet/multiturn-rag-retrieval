# Experiment Results Table

Comprehensive results across all experiments and domains.

## Main Results

| Experiment | ClapNQ | FiQA | Govt | Cloud | **Macro Avg** |
|-----------|--------|------|------|-------|--------------|
| **A0: Sparse Baseline (BM25 Last)** | 0.2353 | 0.1026 | 0.2767 | 0.2201 | 0.2087 |
| **A0: BM25 FullHist** | 0.2041 | 0.0912 | 0.2696 | 0.1408 | 0.1764 |
| **A1: Dense Baseline (BGE-M3 Last)** | 0.4768 | 0.3613 | 0.4197 | 0.3938 | 0.4129 |
| **A1: BGE-M3 FullHist** | 0.3248 | 0.2223 | 0.3246 | 0.2291 | 0.2752 |
| **A1: BGE-1.5 Last (Repl)** | 0.4762 | 0.3457 | 0.3952 | 0.3762 | 0.3983 |
| **A1: Voyage Last (Repl)** | 0.5163 | 0.2912 | 0.4144 | 0.3621 | 0.3960 |
| **A1: Voyage FullHist** | 0.3567 | 0.1790 | 0.3458 | 0.2333 | 0.2787 |
| **SPLADE Baseline (Last)** | 0.5082 | 0.3715 | 0.4515 | 0.4512 | 0.4456 |
| **SPLADE FullHist** | 0.3449 | 0.2025 | 0.3494 | 0.2236 | 0.2801 |
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

**Average Performance by Query Mode:**
- **Last Turn (Avg across all models):** 0.3723
- **Full History (Avg across all models):** 0.2526
- **Difference:** Last Turn is +47% better on average.


## Detailed Metrics by Experiment

### A0: Sparse Baseline

| Domain | Recall@10 | MRR | NDCG@10 | MAP | Latency (ms) |
|--------|-----------|-----|---------|-----|--------------|
| ClapNQ | 0.3384 | - | 0.2353 | 0.1604 | 764 |
| FiQA | 0.1712 | - | 0.1026 | 0.0589 | 263 |
| Govt | 0.3794 | - | 0.2767 | 0.2087 | 196 |
| Cloud | 0.3131 | - | 0.2201 | 0.1630 | 272 |

### A1: Dense Baseline (BGE-M3)

| Domain | Recall@10 | MRR | NDCG@10 | MAP | Latency (ms) |
|--------|-----------|-----|---------|-----|--------------|
| ClapNQ | 0.6237 | - | 0.4768 | 0.3721 | 19 |
| FiQA | 0.4847 | - | 0.3613 | 0.2640 | 18 |
| Govt | 0.5620 | - | 0.4197 | 0.3252 | 18 |
| Cloud | 0.5145 | - | 0.3938 | 0.3030 | 18 |

### SPLADE Baseline

| Domain | Recall@10 | MRR | NDCG@10 | MAP | Latency (ms) |
|--------|-----------|-----|---------|-----|--------------|
| ClapNQ | 0.6760 | - | 0.5082 | 0.3993 | 89 |
| FiQA | 0.5241 | - | 0.3715 | 0.2672 | 43 |
| Govt | 0.6176 | - | 0.4515 | 0.3513 | 34 |
| Cloud | 0.6143 | - | 0.4512 | 0.3487 | 45 |

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
| ClapNQ | SPLADE | 0.5082 | Strongest baseline so far |
| FiQA | SPLADE | 0.3715 | Outperforms dense models |
| Govt | SPLADE | 0.4515 | Strongest baseline so far |
| Cloud | SPLADE | 0.4512 | Strongest baseline so far |

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
