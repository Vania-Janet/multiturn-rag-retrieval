# MT-RAG Benchmark: Task A - Retrieval

Multi-turn conversational retrieval experiments across 4 domains (ClapNQ, Cloud, FiQA, Govt).

**Baseline results:** [HuggingFace Dataset](https://huggingface.co/datasets/vania-janet/MTRAG_taskA_results) | **Paper:** ACL 2024

## Overview

Evaluates retrieval methods on multi-turn queries:
- **Models**: BM25, BGE-1.5, BGE-M3, ELSER, SPLADE, Voyage
- **Strategies**: Last Turn vs Full History
- **Metrics**: NDCG@10, Recall@100, MRR

## Quick Start

### Docker (Recommended)

```bash
# Clone and setup
git clone https://github.com/Vania-Janet/multiturn-rag-retrieval.git
cd mt-rag-benchmark/task_a_retrieval

# Build and start
docker-compose build
docker-compose up -d
docker-compose exec mtrag-retrieval bash

# Inside container: Run experiment
python scripts/run_experiment.py --config configs/experiments/0-baselines/replication_bm25.yaml
```

**Requirements:** Docker 20.10+, nvidia-docker2, NVIDIA GPU

### Local Setup

```bash
# Automated setup
./setup.sh
source .venv/bin/activate

# Run experiment
python scripts/run_experiment.py --config configs/experiments/0-baselines/replication_bm25.yaml
```

**Requirements:** Python 3.10+, CUDA 12.1+, 16GB GPU RAM

## Available Experiments

```bash
# BM25 baseline
python scripts/run_experiment.py --config configs/experiments/0-baselines/replication_bm25.yaml

# BGE-1.5 baseline
python scripts/run_experiment.py --config configs/experiments/0-baselines/replication_bge15.yaml

# BGE-M3 with full history
python scripts/run_experiment.py --config configs/experiments/0-baselines/A1_baseline_bgem3_fullhist.yaml
```

Results saved to `experiments/{experiment_name}/{domain}/`:
- `metrics.json` - NDCG@10, Recall@100, MRR
- `retrieval_results.jsonl` - Retrieved documents
- `analysis_report.json` - Statistical validation

## Results

Baseline results available on [HuggingFace](https://huggingface.co/datasets/vania-janet/MTRAG_taskA_results) (679 files, 10.8 GB).

**NDCG@10 Performance:**

| Method | ClapNQ | Cloud | FiQA | Govt | Avg |
|--------|--------|-------|------|------|-----|
| BM25 | 0.378 | 0.459 | 0.328 | 0.482 | 0.412 |
| BGE-1.5 | 0.461 | 0.521 | 0.398 | 0.556 | 0.484 |
| BGE-M3 | 0.489 | 0.548 | 0.421 | 0.579 | 0.509 |

See [docs/VALIDACION_ESTADISTICA_COMPLETA.md](docs/VALIDACION_ESTADISTICA_COMPLETA.md) for statistical validation (777 queries, 96.14% accuracy).

## Troubleshooting

**GPU not detected:**
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**Out of memory:**
- Reduce `embedder_batch_size` in `configs/base.yaml`
- Use `CUDA_VISIBLE_DEVICES=0` for single GPU

**Docker issues:**
```bash
docker-compose logs       # View logs
docker-compose down       # Stop
docker-compose build      # Rebuild
```

## Documentation

- [DOCKER_USAGE.md](DOCKER_USAGE.md) - Docker guide
- [CHANGELOG.md](CHANGELOG.md) - Recent changes
- [docs/VALIDACION_ESTADISTICA_COMPLETA.md](docs/VALIDACION_ESTADISTICA_COMPLETA.md) - Statistical validation

## Citation

```bibtex
@inproceedings{mtrag2024,
  title={MT-RAG: Multi-Turn Retrieval-Augmented Generation Benchmark},
  author={Your Name et al.},
  booktitle={Proceedings of ACL 2024},
  year={2024}
}
```

License: Apache 2.0
