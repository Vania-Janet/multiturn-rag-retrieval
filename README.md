# MT-RAG Benchmark: Task A - Retrieval

[![Paper](https://img.shields.io/badge/Paper-ACL%202024-blue)](https://arxiv.org/placeholder)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Official implementation of the retrieval experiments from the MT-RAG Benchmark paper (ACL 2024).

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Reproducibility](#reproducibility)
- [Experiments](#experiments)
- [Results](#results)
- [Citation](#citation)

## 🎯 Overview

This repository contains code for evaluating retrieval methods on multi-turn conversational queries across four domains:
- **ClapNQ**: Conversational QA
- **Cloud**: Cloud computing documentation
- **FiQA**: Financial QA
- **Govt**: Government documents

We evaluate:
- **Sparse Retrieval**: BM25, ELSER
- **Dense Retrieval**: BGE-1.5, BGE-M3
- **Query Strategies**: Last Turn, Full History

## 🚀 Quick Start

### Option 1: Docker (Recommended for Reproducibility)

Docker setup provides complete environment reproducibility with GPU support.

#### Prerequisites
- Docker 20.10+ and Docker Compose 2.0+
- NVIDIA Docker runtime (`nvidia-docker2` package)
- NVIDIA GPU with CUDA 12.1+ support

```bash
# Clone repository
git clone https://github.com/Vania-Janet/multiturn-rag-retrieval.git
cd mt-rag-benchmark/task_a_retrieval

# Create .env file with API keys (optional, for ELSER/Cohere/Voyage)
cat > .env << EOF
COHERE_API_KEY=your_key_here
VOYAGE_API_KEY=your_key_here
HUGGINGFACE_TOKEN=your_token_here
EOF

# Build Docker image
docker-compose build

# Start container with GPU support
docker-compose up -d

# Enter container
docker-compose exec mtrag-retrieval bash

# Inside container: Build indices
python src/pipeline/indexing/build_indices.py --models bge bge-m3 bm25 --domains all

# Run experiments
python scripts/run_experiment.py --config configs/experiments/0-baselines/replication_bm25.yaml
```

#### Docker Volumes
Persistent data is mounted from host:
- `./data` → Container data directory
- `./experiments` → Experiment results
- `./indices` → Built indices (FAISS, BM25)
- `./cache` → Model cache (HuggingFace, Transformers)

#### Useful Docker Commands
```bash
# View logs
docker-compose logs -f

# Stop container
docker-compose down

# Rebuild after code changes
docker-compose build --no-cache

# Run single command without entering container
docker-compose run --rm mtrag-retrieval python scripts/run_experiment.py --help

# Check GPU availability in container
docker-compose exec mtrag-retrieval nvidia-smi
```

### Option 2: Local Setup (Linux/Ubuntu 22.04)

```bash
# Run automated setup
./setup.sh

# Activate environment
source .venv/bin/activate

# Build indices
python src/pipeline/indexing/build_indices.py --models bge bge-m3 bm25 --domains clapnq cloud fiqa govt

# Run experiments
python scripts/run_experiment.py --config configs/experiments/0-baselines/replication_bm25.yaml
```

### Option 3: Manual Setup

```bash
# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies (pinned versions for reproducibility)
pip install -r requirements.txt

# Install FAISS GPU
pip install faiss-gpu

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

## 🔬 Reproducibility

This repository follows ACL 2024 reproducibility guidelines with full Docker support.

### Docker Environment (Recommended)
- **Base Image**: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- **Python**: 3.10
- **PyTorch**: 2.0+ with CUDA 12.1
- **GPU Support**: NVIDIA Docker runtime with all GPUs accessible
- **Deterministic**: All random seeds fixed at 42
- **Cached Models**: HuggingFace/Transformers cache persisted in volumes

All experiment configurations in `configs/experiments/` use deterministic settings.

### Local Environment
- **Python**: 3.10+
- **CUDA**: 12.1
- **PyTorch**: 2.0+ (see `requirements.txt` for exact versions)
- **Random Seeds**: Fixed at 42 across all experiments
- **Hardware**: Tested on NVIDIA A100 (40GB) and RTX 4090

### Data
Download the MT-RAG benchmark data:
```bash
# TODO: Add data download instructions
# wget <data-url>
# unzip data.zip
```

Expected directory structure:
```
data/
├── passage_level_processed/
│   ├── clapnq/corpus.jsonl
│   ├── cloud/corpus.jsonl
│   ├── fiqa/corpus.jsonl
│   └── govt/corpus.jsonl
└── retrieval_tasks/
    ├── clapnq/
    │   ├── clapnq_lastturn.jsonl
    │   ├── clapnq_questions.jsonl
    │   └── qrels/dev.tsv
    └── ...
```

### Deterministic Configuration
All experiments use deterministic settings:
- `PYTHONHASHSEED=0`
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- `torch.use_deterministic_algorithms(True)`
- Fixed batch sizes and worker counts

## 🧪 Experiments

### Paper Replication (Last Turn)
Replicates baseline results from the original paper:
```bash
# BM25 Baseline
python scripts/run_experiment.py --experiment replication_bm25 --domain all

# BGE 1.5 Baseline
python scripts/run_experiment.py --experiment replication_bge15 --domain all

# ELSER Baseline (requires Elasticsearch)
python scripts/run_experiment.py --experiment replication_elser --domain all
```

### Advanced Baselines (Full History)
Uses full conversation history:
```bash
# BM25 with Full History
python scripts/run_experiment.py --experiment A0_baseline_bm25_fullhist --domain all

# BGE-M3 with Full History
python scripts/run_experiment.py --experiment A1_baseline_bgem3_fullhist --domain all

# ELSER with Full History
python scripts/run_experiment.py --experiment A0_baseline_elser_fullhist --domain all
```

### Running All Baselines
```bash
# Run all 6 baseline experiments across 4 domains (24 total runs)
for exp in replication_bm25 replication_bge15 replication_elser A0_baseline_bm25_fullhist A1_baseline_bgem3_fullhist A0_baseline_elser_fullhist; do
    python scripts/run_experiment.py --experiment $exp --domain all
done
```

### Parallel Execution (Multi-GPU)
See [SETUP_A100_GUIDE.md](configs/experiments/0-baselines/SETUP_A100_GUIDE.md) for parallel execution on multi-GPU servers.

## 📊 Results

Results are saved in `experiments/{experiment_name}/{domain}/`:
- `retrieval_results.jsonl`: Retrieved documents with scores
- `metrics.json`: Evaluation metrics (NDCG@10, Recall@10, etc.)
- `analysis_report.json`: Statistical analysis and robustness metrics

### Expected Performance (NDCG@10)

| Method | ClapNQ | Cloud | FiQA | Govt | Avg |
|--------|--------|-------|------|------|-----|
| BM25 (Last) | X.XX | X.XX | X.XX | X.XX | X.XX |
| BGE-1.5 (Last) | X.XX | X.XX | X.XX | X.XX | X.XX |
| BGE-M3 (Full) | X.XX | X.XX | X.XX | X.XX | X.XX |

*Fill in with actual results from your experiments*

## 📁 Repository Structure

```
task_a_retrieval/
├── configs/                       # Experiment configurations
│   ├── base.yaml                 # Global settings
│   ├── domains/                  # Domain-specific configs
│   └── experiments/              # Experiment-specific configs
├── data/                         # Data files
│   ├── passage_level_processed/  # Corpus documents
│   ├── retrieval_tasks/          # Queries and qrels
│   └── submissions/              # Test submissions (gitignored)
├── src/
│   ├── pipeline/
│   │   ├── indexing/            # Index building (FAISS, BM25)
│   │   ├── retrieval/           # Retrieval models
│   │   └── evaluation/          # Evaluation metrics
│   └── utils/                   # Utilities
├── scripts/                      # Organized utility scripts
│   ├── run_experiment.py        # Main experiment runner
│   ├── make_submission.py       # Generate test submissions
│   ├── extract_test_queries.py  # Test query extraction
│   ├── run_test_submission.py   # Run test retrieval
│   └── ...                      # Other utilities
├── experiments/                  # Experiment results (gitignored)
├── indices/                      # Built indices (gitignored)
├── cache/                        # Model cache (gitignored)
├── logs/                         # Execution logs
├── docs/                         # Documentation
│   ├── VALIDACION_ESTADISTICA_COMPLETA.md
│   ├── RESUMEN_PARA_PROFESORA.md
│   └── ...
├── Dockerfile                    # Production container
├── docker-compose.yml            # Multi-service orchestration  
├── .dockerignore                 # Docker build exclusions
├── setup.sh                      # Automated setup script
└── requirements.txt              # Python dependencies
```

## 🛠️ Troubleshooting

### GPU not detected
```bash
# Check NVIDIA driver
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Elasticsearch connection failed
```bash
# Check Elasticsearch status
curl http://localhost:9200/_cluster/health

# Restart Elasticsearch
docker-compose restart elasticsearch
```

### Out of memory
- Reduce batch size in `configs/base.yaml`
- Use smaller model (BGE-base instead of BGE-large)
- Use single GPU with `CUDA_VISIBLE_DEVICES=0`

## 📝 Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{mtrag2024,
  title={MT-RAG: Multi-Turn Retrieval-Augmented Generation Benchmark},
  author={Your Name et al.},
  booktitle={Proceedings of ACL 2024},
  year={2024}
}
```

## 📄 License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

This is research code. For questions or issues, please open a GitHub issue.

## 🙏 Acknowledgments

- IBM Research for the MT-RAG benchmark dataset
- HuggingFace for BGE models
- Elastic for ELSER model
