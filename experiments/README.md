# Experiments Directory Structure

This directory contains all experimental results organized by experiment type.

## Directory Structure

```
experiments/
├── 0-baselines/          # Baseline experiments (no query transformation)
├── 01-query/             # Query transformation experiments (rewrite, HyDE, etc.)
├── 02-hybrid/            # Hybrid retrieval experiments (sparse + dense)
├── 03-rerank/            # Reranking experiments
├── 04-iterative/         # Iterative retrieval experiments (future)
├── 05-finetune/          # Fine-tuned model experiments (future)
└── [legacy folders]      # Original flat structure (to be cleaned up)
```

## Experiment Categories

### 0-baselines/
Baseline experiments without query transformation:
- `A0_baseline_bm25_fullhist` - BM25 with full conversation history
- `A0_baseline_splade_fullhist` - SPLADE with full conversation history
- `A1_baseline_bgem3_fullhist` - BGE-M3 with full conversation history
- `A1_baseline_voyage_fullhist` - Voyage with full conversation history
- `replication_bm25` - BM25 with last turn (paper baseline)
- `replication_bge15` - BGE-1.5 with last turn
- `replication_bgem3` - BGE-M3 with last turn
- `replication_splade` - SPLADE with last turn
- `replication_voyage` - Voyage with last turn

### 01-query/
Query transformation experiments:
- `*_r1_condensation` - R1 query condensation with LLM
- `*_r2_multi` - R2 multi-diverse query generation (k=3)
- `*_r3_hyde` - R3 HyDE (Hypothetical Document Embeddings)

### 02-hybrid/
Hybrid retrieval combining sparse and dense methods:
- `hybrid_splade_*` - SPLADE + dense retriever
- `hybrid_cohere_*` - Cohere + another retriever

### 03-rerank/
Experiments with reranking:
- `bgem3_cohere_norewrite` - BGE-M3 retrieval + Cohere reranking

## Important Findings

### Full History vs Last Turn

Our experiments show that **using only the last turn consistently outperforms using full conversation history** for retrieval:

| Domain | Full History nDCG@100 | Last Turn nDCG@100 | Improvement |
|--------|----------------------|-------------------|-------------|
| clapnq | 0.4229 | 0.5492 | +29.9% |
| cloud  | 0.2812 | 0.4064 | +44.5% |
| fiqa   | 0.2213 | 0.3446 | +55.7% |
| govt   | 0.4024 | 0.4505 | +12.0% |

**Why Last Turn is Better:**
1. **More specific**: The final user question is what actually needs answering
2. **Less noise**: Previous turns may contain irrelevant context
3. **Better embeddings**: Dense retrieval models work better with focused, short queries
4. **Clearer intent**: The last question represents the user's current information need

**Implication**: For conversational retrieval, using the last user turn as the query is the recommended baseline approach. Full history can be valuable for query rewriting/condensation by an LLM, but not for direct embedding.

## Naming Convention

Experiments follow this naming pattern:
- `{model}_{technique}_{variant}`
  - Model: `bm25`, `splade`, `bgem3`, `voyage`, etc.
  - Technique: `r1`, `r2`, `r3`, `hybrid`, etc.
  - Variant: `condensation`, `multi`, `norewrite`, etc.

Baselines use:
- `A{N}_baseline_{model}_{mode}` for new baselines
- `replication_{model}` for paper replication baselines

## Results Location

Each experiment directory contains subdirectories for each domain:
```
experiment_name/
├── clapnq/
│   ├── metrics.json           # Evaluation metrics
│   ├── retrieval_results.jsonl # Raw retrieval results
│   └── analysis_report.json   # Detailed analysis
├── cloud/
├── fiqa/
├── govt/
└── config_resolved.yaml       # Full resolved configuration
```

## Running Experiments

To run an experiment:
```bash
python scripts/run_experiment.py -e experiment_name -d domain
```

To run on all domains:
```bash
for domain in clapnq cloud fiqa govt; do
    python scripts/run_experiment.py -e experiment_name -d $domain
done
```

## Next Steps

1. Clean up legacy flat structure directories
2. Add missing experiments to appropriate categories
3. Run baseline experiments on all domains
4. Document best performing configurations per category
