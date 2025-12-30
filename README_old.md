# MT-RAG Benchmark - Vania Evals

## ✅ Project Status

**Estructura completa configurada** con:

- ✅ **Configuraciones jerárquicas**: `configs/base.yaml` → domains → experiments
- ✅ **12 experimentos** organizados: baselines, query, hybrid, rerank, finetune, iterative
- ✅ **Pipeline modular**: `src/pipeline/retrieval/` + `reranking/`
- ✅ **Scripts canónicos**: `run_experiment.py`, `build_indices.py`, `reproduce_baselines.py`
- ✅ **Prevención de leakage**: splits fijos en `splits/`, política documentada en `docs/leakage_policy.md`
- ✅ **Documentación para revisores**: `docs/methodology.md`, `experiment_table.md`
- ✅ **Gestión de artifacts**: `artifacts/` con política git/cloud en README
- ✅ **Validación de submissions**: `scripts/make_submission.py` + `validate_submission.py`
- ✅ **Reproducibilidad**: seeds, determinismo, manifests en `base.yaml` y `manifest.example.json`

---

## 🎯 Overview

This project implements a comprehensive evaluation framework for multi-domain RAG systems, with emphasis on reproducibility, leakage prevention, and reviewer transparency.

### Key Features

- **Multi-domain evaluation**: ClapNQ (conversational), FiQA (financial), Government (policy), Cloud (technical)
- **Modular pipeline**: Retrieval (sparse/dense/hybrid) → Reranking → Evaluation
- **12 experiments**: From baselines (A0-A1) to fine-tuned models (A10-A11)
- **Leakage prevention**: Conversation-level splits, fixed hyperparameters, audit trails
- **Reproducibility**: Fixed seeds, deterministic training, preprocessing manifests
- **Experiment tracking**: MLflow and Weights & Biases integration

## 📁 Project Structure

```
vania-evals/
├── configs/
│   ├── base.yaml                    # Global defaults (seed, batch sizes, reproducibility)
│   ├── domains/                     # Domain-specific configs
│   │   ├── clapnq.yaml
│   │   ├── fiqa.yaml
│   │   ├── govt.yaml
│   │   └── cloud.yaml
│   └── experiments/                 # Experiment configs organized by category
│       ├── baselines/               # A0, A1
│       ├── query/                   # A2, A3, A4
│       ├── hybrid/                  # A5, A6, A7
│       ├── rerank/                  # A9
│       ├── finetune/                # A10, A11
│       └── iterative/               # A8
│
├── data/
│   ├── raw/                         # Original datasets (gitignored)
│   ├── processed/                   # Chunked corpus with manifests (gitignored)
│   │   └── manifest.example.json   # Template for tracking preprocessing
│   └── responses-10.jsonl           # Ground truth with contexts and targets
│
├── splits/
│   ├── clapnq.yaml                  # Conversation-level splits (prevents turn leakage)
│   ├── fiqa.yaml                    # Query-level splits with stratification
│   ├── govt.yaml
│   └── cloud.yaml
│
├── src/
│   └── pipeline/
│       ├── retrieval/               # Modular retrieval implementations
│       │   ├── sparse.py            # BM25, ELSER, SPLADE
│       │   ├── dense.py             # BGE-M3, Sentence Transformers
│       │   ├── hybrid.py            # Combines sparse + dense
│       │   └── fusion.py            # RRF, linear combination
│       └── reranking/               # Modular reranking implementations
│           ├── cross_encoder.py     # CrossEncoder, DomainAdapted
│           └── colbert.py           # ColBERT, ColBERTv2
│
├── scripts/
│   ├── run_experiment.py            # Main: run any experiment on any domain
│   ├── build_indices.py             # Build all retrieval indices
│   ├── reproduce_baselines.py       # One-command baseline reproduction
│   ├── make_submission.py           # Create Task A submission JSONL
│   └── validate_submission.py       # Comprehensive format validation
│
├── preprocessing/
│   ├── corpus_sanity.py             # Validation checks
│   ├── chunking/
│   │   ├── sliding_window.py        # Fixed-size with overlap
│   │   └── hierarchical.py          # Semantic/structural chunking
│   └── build_processed_corpus.py    # Full preprocessing pipeline
│
├── experiments/                     # Results storage (gitignored)
│   └── A0_baseline_sparse/          # Example experiment
│       ├── config_resolved.yaml     # Merged base + domain + experiment
│       ├── clapnq/
│       │   ├── queries/
│       │   ├── retrieval/
│       │   ├── reranking/
│       │   └── eval/
│       ├── aggregate/
│       └── logs/
│
├── artifacts/                       # Large files (managed outside git)
│   ├── models/                      # Fine-tuned checkpoints → HuggingFace Hub
│   ├── embeddings/                  # Cached embeddings → regenerable
│   ├── logs/                        # Detailed execution logs → W&B
│   └── README.md                    # Policy: what to commit, where to store
│
├── docs/
│   ├── methodology.md               # Experimental design, metrics, reproducibility
│   ├── leakage_policy.md            # 10-point data leakage prevention policy
│   └── experiment_table.md          # Results table template for paper
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md (this file)
```

## 🚀 Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Elasticsearch credentials, API keys, etc.

# Prepare data splits (prevents leakage)
python scripts/prepare_splits.py
```

### 2. Preprocess Corpus

```bash
# Build processed corpus with explicit manifest tracking
python preprocessing/build_processed_corpus.py \
  --domain clapnq \
  --strategy sliding_window \
  --chunk-size 512 \
  --overlap 128

# Check manifest to verify what was built
cat data/processed/clapnq/manifest.json
```

### 3. Build Indices

```bash
# Build all retrieval indices (BM25, ELSER, SPLADE, BGE-M3, ColBERT)
python scripts/build_indices.py --domain clapnq --models all

# Or build specific index
python scripts/build_indices.py --domain clapnq --models bm25
```

### 4. Run Experiments

```bash
# Run single experiment on single domain
python scripts/run_experiment.py --experiment A0 --domain clapnq

# Run experiment on all domains
python scripts/run_experiment.py --experiment A0 --domain all

# Reproduce all baselines at once
python scripts/reproduce_baselines.py --domains all
```

### 5. Create Submission

```bash
# Generate submission file from experiment results
python scripts/make_submission.py \
  --experiment A0 \
  --domain clapnq \
  --output submissions/A0_clapnq.jsonl \
  --run-name "vania_bm25_baseline"

# Validate submission format before submitting
python scripts/validate_submission.py \
  --file submissions/A0_clapnq.jsonl \
  --expected-queries 120
```

---

## 🔬 Experiments Overview

| ID | Category | Description |
|----|----------|-------------|
| **A0** | Baseline | Sparse retrieval only (BM25/ELSER) |
| **A1** | Baseline | Dense retrieval only (BGE-M3) |
| **A2** | Query | Query rewriting with LLM |
| **A3** | Query | Query expansion with PRF |
| **A4** | Query | Conversation history concatenation |
| **A5** | Hybrid | Hybrid retrieval (sparse + dense, RRF fusion) |
| **A6** | Hybrid | A5 + query rewriting |
| **A7** | Hybrid | A6 + domain-specific rules |
| **A8** | Iterative | Iterative refinement with feedback |
| **A9** | Reranking | A7 + ColBERT reranking |
| **A10** | Fine-tune | Fine-tuned SPLADE |
| **A11** | Fine-tune | Fine-tuned reranker |

---

## 🔒 Reproducibility Guarantees

### Fixed Seeds
All random operations use **seed=42** (configured in [configs/base.yaml](configs/base.yaml)):
- Python `random.seed(42)`
- NumPy `np.random.seed(42)`
- PyTorch `torch.manual_seed(42)`
- PYTHONHASHSEED=42

### Deterministic Settings
```yaml
# configs/base.yaml
torch:
  use_deterministic_algorithms: true
  backends:
    cudnn:
      deterministic: true
      benchmark: false
```

### Data Leakage Prevention
- ✅ **Fixed splits** in [splits/](splits/) (version-controlled)
- ✅ **Conversation-level splits** for ClapNQ (prevents turn leakage)
- ✅ **Hyperparameter tuning** only on validation set
- ✅ **Domain parameters** fixed a priori (not empirically tuned on test)
- ✅ **Audit trail** in `experiments/{exp}/logs/audit.json`

See [docs/leakage_policy.md](docs/leakage_policy.md) for complete 10-point policy.

### Preprocessing Manifests
Each processed corpus has a `manifest.json` tracking:
- Chunking strategy and parameters
- Raw corpus hash (detects changes)
- Build date and statistics
- Output file paths

Template: [data/processed/manifest.example.json](data/processed/manifest.example.json)

---

## ⚙️ Configuration System

Hierarchical YAML configs enable parametric experimentation:

```
base.yaml (global defaults)
  ↓
domains/clapnq.yaml (domain overrides)
  ↓
experiments/baselines/A0_baseline_sparse.yaml (experiment specifics)
```

**Run with merged config:**
```bash
python scripts/run_experiment.py --experiment A0 --domain clapnq
# Loads: base.yaml + clapnq.yaml + A0_baseline_sparse.yaml
```

**Key configuration sections** in [configs/base.yaml](configs/base.yaml):
- `seed`, `deterministic`: Reproducibility settings
- `batch_sizes`: Indexing, inference, reranking batch sizes
- `retrieval`: Sparse/dense/hybrid defaults
- `evaluation.metrics`: Recall@k, MRR, NDCG, MAP, latency
- `training`: Fine-tuning hyperparameters with early stopping

---

## 📊 Evaluation Metrics

- **Recall@k** (k=5, 10, 20, 100): Fraction of relevant docs in top-k
- **MRR**: Mean reciprocal rank of first relevant doc
- **NDCG@10**: Normalized discounted cumulative gain
- **MAP**: Mean average precision
- **Latency**: Query processing time (ms)

**Statistical Testing:**
- Paired t-tests between experiments
- Bonferroni correction for multiple comparisons
- p < 0.05 threshold

See [docs/methodology.md](docs/methodology.md) for complete methodology.

---

## 🌐 Domain-Specific Parameters

Each domain has optimized settings in `configs/domains/`:

- **ClapNQ** (conversational): `sparse_weight: 0.6` (favor BM25 for code)
- **FiQA** (financial): `sparse_weight: 0.4` (favor dense for semantics)
- **Government**: `chunk_size: 1024` (long policy documents)
- **Cloud**: `top_k: 100` (technical queries need broad recall)

Rationales documented in [docs/methodology.md](docs/methodology.md#domain-specific-parameters).

---

## 🛠️ Development Workflow

### Adding a New Experiment

1. Create config in `configs/experiments/{category}/A12_new_experiment.yaml`
2. Update `EXPERIMENT_DIRS` mapping in [scripts/run_experiment.py](scripts/run_experiment.py)
3. Run experiment: `python scripts/run_experiment.py --experiment A12 --domain clapnq`
4. Validate submission: `python scripts/validate_submission.py --file submissions/A12_clapnq.jsonl`

### Debugging

```bash
# Dry run (shows merged config, no execution)
python scripts/run_experiment.py --experiment A0 --domain clapnq --dry-run

# Check logs
tail -f experiments/A0_baseline_sparse/logs/run.log

# Validate data splits have no overlap
python scripts/validate_splits.py --domain clapnq
```

### Common Issues

**Q: "ModuleNotFoundError: No module named 'splade'"**
→ Install from git: `pip install git+https://github.com/naver/splade.git`

**Q: "Elasticsearch connection refused"**
→ Check `.env` has correct `ELASTICSEARCH_HOST` and cluster is running

**Q: "CUDA out of memory"**
→ Reduce `batch_sizes.inference` in [configs/base.yaml](configs/base.yaml) or use `device: "cpu"`

---

## 📄 Documentation

- [docs/methodology.md](docs/methodology.md): Research questions, experimental design, metrics, reproducibility
- [docs/leakage_policy.md](docs/leakage_policy.md): 10-point data leakage prevention policy
- [docs/experiment_table.md](docs/experiment_table.md): Results table template for paper
- [artifacts/README.md](artifacts/README.md): Policy for managing large files (models, embeddings, logs)

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys and settings
nano .env
```

### 3. Run Experiments

```bash
# Run a specific experiment on a domain
python scripts/run_experiment.py --domain clapnq --experiment A2_rewrite_splade

# Run all experiments for a domain
python scripts/run_experiment.py --domain fiqa --experiment all

# Run baseline across all domains
python scripts/run_experiment.py --domain all --experiment A0_baseline
```

## 🔧 Configuration System

### Hierarchical Loading

Configurations are loaded in this order:
1. `base.yaml` - Global defaults
2. `domains/{domain}.yaml` - Domain-specific parameters
3. `experiments/{experiment}.yaml` - Experiment-specific modules

Example:
```bash
run --domain clapnq --experiment A2_rewrite_splade
```
Loads: `base.yaml` → `domains/clapnq.yaml` → `experiments/A2_rewrite_splade.yaml`

### Configuration Files

#### Base Configuration (`configs/base.yaml`)
```yaml
retrieval:
  top_k: 100
  sparse_weight: 0.5
  dense_weight: 0.5

reranking:
  enabled: false
  top_k: 10
```

#### Domain Configuration (`configs/domains/clapnq.yaml`)
```yaml
query_processing:
  rewrite_variants: 5
  enable_error_code_regex: true

retrieval:
  sparse_weight: 0.6
  dense_weight: 0.4
```

#### Experiment Configuration (`configs/experiments/A2_rewrite_splade.yaml`)
```yaml
modules:
  query_rewrite: true
  sparse_retrieval: splade
  dense_retrieval: false
  reranking: false
```

## 🧪 Experiments

| ID | Name | Query Rewrite | Retrieval | Reranking |
|----|------|---------------|-----------|-----------|
| A0 | Baseline | ❌ | BM25 | ❌ |
| A1 | Rewrite | ✅ | BM25 | ❌ |
| A2 | Rewrite + SPLADE | ✅ | SPLADE | ❌ |
| A3 | Hybrid | ✅ | BM25 + BGE-M3 | ❌ |
| A4 | Rerank | ✅ | BM25 + BGE-M3 | ✅ |

## 📊 Evaluation Metrics

- **Recall@k**: Top-k retrieval accuracy
- **MRR**: Mean Reciprocal Rank
- **NDCG@k**: Normalized Discounted Cumulative Gain
- **Precision@k**: Precision at k
- **Latency**: End-to-end query time

## 🔍 Indices

Pre-built indices are stored in `indices/{domain}/{model}/` for reusability across experiments:

```
indices/
├── clapnq/
│   ├── bm25/          # Elasticsearch BM25 index
│   ├── elser/         # Elastic Learned Sparse Encoder
│   ├── splade/        # SPLADE sparse vectors
│   ├── bge-m3/        # BGE-M3 dense embeddings
│   └── colbert/       # ColBERT multi-vector
```

## 📝 Development

### Running Tests

```bash
pytest tests/
pytest tests/ --cov=src  # with coverage
```

### Code Quality

```bash
# Format code
black src/ scripts/

# Lint
flake8 src/ scripts/

# Type checking
mypy src/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## 📈 Results

Experiment results are stored with timestamps:

```
experiments/
├── 2025-01-15_A0_baseline/
│   ├── clapnq/
│   │   ├── metrics.json
│   │   ├── predictions.jsonl
│   │   └── config.yaml
│   └── aggregate/
│       └── summary.csv
```

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Run tests and linting
4. Submit a pull request

## 📄 License

MIT License

## 🙏 Acknowledgments

- Built on [mt-rag-benchmark](https://github.com/original/repo)
- Elasticsearch for search infrastructure
- Hugging Face for model hosting

## 📧 Contact

- Author: Vania Janet
- Repository: https://github.com/Vania-Janet/rag-comp3
