# Scripts

Main execution scripts for the RAG benchmark evaluation framework.

## 🎯 Core Scripts

### 1. [`run_experiment.py`](run_experiment.py) ⭐
**Main script for running experiments**

Run any experiment configuration on any domain(s).

```bash
# Single experiment, single domain
python scripts/run_experiment.py --experiment A6_hybrid_rerank --domain fiqa

# Single experiment, all domains
python scripts/run_experiment.py --experiment A6_hybrid_rerank --domain all

# All experiments, single domain
python scripts/run_experiment.py --experiment all --domain clapnq

# Dry run (validate configs only)
python scripts/run_experiment.py --experiment A6_hybrid_rerank --domain fiqa --dry-run

# Force overwrite existing results
python scripts/run_experiment.py --experiment A6_hybrid_rerank --domain fiqa --force
```

### 2. [`build_indices.py`](build_indices.py)
**Build search indices for all retrieval models**

Creates indices from processed corpora for BM25, ELSER, SPLADE, BGE-M3, ColBERT.

```bash
# Build all indices for all domains
python scripts/build_indices.py --all

# Build specific model for specific domain
python scripts/build_indices.py --domain clapnq --model splade

# Build all models for one domain
python scripts/build_indices.py --domain fiqa --model all

# Rebuild existing indices
python scripts/build_indices.py --all --force
```

### 3. [`reproduce_baselines.py`](reproduce_baselines.py)
**One-command baseline reproduction**

Runs canonical baseline experiments (A0-A1) for paper results.

```bash
# Run all baselines on all domains
python scripts/reproduce_baselines.py

# Run specific baseline
python scripts/reproduce_baselines.py --baseline A0_baseline_sparse

# Run on specific domains only
python scripts/reproduce_baselines.py --domain clapnq fiqa

# Dry run
python scripts/reproduce_baselines.py --dry-run
```

## 📁 Additional Scripts

### Training Scripts ([`training/`](training/))
Fine-tuning scripts for domain-adapted models:
- `finetune_reranker.py`: Train cross-encoder reranker
- `finetune_splade.py`: Train SPLADE model
- `prepare_training_data.py`: Generate training pairs

See [`training/README.md`](training/README.md) for details.

## 🔄 Complete Workflow

### 1. Preprocessing
```bash
# Validate raw corpus
python preprocessing/corpus_sanity.py --corpus data/raw/clapnq/corpus.jsonl

# Process corpus with chunking
python preprocessing/build_processed_corpus.py --domain clapnq

# Or process all domains
python preprocessing/build_processed_corpus.py --all
```

### 2. Index Building
```bash
# Build all indices
python scripts/build_indices.py --all

# Or build specific indices as needed
python scripts/build_indices.py --domain clapnq --model splade
```

### 3. Run Baselines
```bash
# Establish lower bounds
python scripts/reproduce_baselines.py
```

### 4. Run Main Experiments
```bash
# Run key experiments
python scripts/run_experiment.py --experiment A6_hybrid_rerank --domain all
python scripts/run_experiment.py --experiment A7_domain_gated --domain all
```

### 5. Fine-tuning (Optional)
```bash
# Prepare training data
python scripts/training/prepare_training_data.py --domain clapnq

# Fine-tune models
python scripts/training/finetune_reranker.py --domain clapnq
python scripts/training/finetune_splade.py --domain clapnq

# Run fine-tuned experiments
python scripts/run_experiment.py --experiment A10_finetuned_reranker --domain all
python scripts/run_experiment.py --experiment A11_finetuned_splade --domain all
```

### 6. Analysis
```bash
# Analyze results in notebooks
jupyter notebook notebooks/02_results_analysis.ipynb
```

## 🎨 Script Design Principles

1. **Single responsibility**: Each script does one thing well
2. **Composable**: Scripts can be chained together
3. **Reproducible**: All parameters explicit, results deterministic
4. **Fail-safe**: Validates inputs, checks prerequisites
5. **Informative**: Clear logging and progress reporting

## 📊 Output Locations

- **Processed corpora**: `data/processed/{domain}/`
- **Indices**: `indices/{domain}/{model}/`
- **Experiment results**: `experiments/{experiment}/{domain}/`
- **Logs**: `experiments/{experiment}/logs/`
- **Aggregated results**: `experiments/{experiment}/aggregate/`

## 🔧 Common Options

Most scripts support:
- `--verbose` / `-v`: Detailed logging
- `--force`: Overwrite existing outputs
- `--dry-run`: Validate without execution
- `--help` / `-h`: Show full usage

## ⚡ Performance Tips

- **Parallel execution**: Run multiple domains in parallel
  ```bash
  for domain in clapnq fiqa govt cloud; do
    python scripts/run_experiment.py --experiment A6_hybrid_rerank --domain $domain &
  done
  wait
  ```

- **Index reuse**: Indices are built once and reused across experiments
- **Caching**: Embeddings and LLM calls are cached in `.cache/`

## 🐛 Debugging

Enable verbose mode for detailed logs:
```bash
python scripts/run_experiment.py --experiment A6_hybrid_rerank --domain fiqa --verbose
```

Check logs in:
- `experiments/{experiment}/logs/run.log`
- Console output

## 📚 Further Reading

- [Experiment definitions](../experiments/README.md)
- [Configuration system](../configs/README.md)
- [Pipeline architecture](../src/pipeline/README.md)
