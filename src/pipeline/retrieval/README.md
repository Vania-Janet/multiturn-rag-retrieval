# Retrieval Pipeline

This package contains the core retrieval logic for the MT-RAG benchmark.

## Features

- **Dense Retrieval**: BGE 1.5 (`BAAI/bge-large-en-v1.5`) with instruction tuning. Optimized for A100 (FP16).
- **Sparse Retrieval**: BM25 and ELSER (Elasticsearch).
- **Hybrid Retrieval**: Weighted fusion of dense and sparse scores.
- **Reproducibility**: Strict seeding of random, numpy, and torch generators.
- **Statistical Analysis**:
    - Wilcoxon Signed-Rank Test
    - Bootstrap Confidence Intervals (95%)
    - Bonferroni Correction for multiple comparisons
- **Robustness Analysis**:
    - Latency Monitoring (P95, P99)
    - Hard Failure Analysis (NDCG=0)
    - Late Turn Performance Analysis

## Usage

The main entry point is `src/pipeline/run.py`.

```python
from pipeline.run import run_pipeline

config = load_config(...)
run_pipeline(config, output_dir, domain="clapnq")
```

## Analysis Tools

The `analysis` module provides tools for deep error analysis:

```python
from pipeline.retrieval.analysis import analyze_hard_failures, analyze_performance_by_turn

# Identify queries with 0 score
failures = analyze_hard_failures(df_results)

# Check performance degradation over turns
turn_stats = analyze_performance_by_turn(df_results)
```
