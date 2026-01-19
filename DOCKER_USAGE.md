# Docker Usage Guide

Simple guide for running experiments in Docker with GPU support.

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA GPU with CUDA 12.1+
- nvidia-docker2 runtime

**Install NVIDIA Docker:**
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/Vania-Janet/multiturn-rag-retrieval.git
cd mt-rag-benchmark/task_a_retrieval

# 2. Build image (~10 minutes)
docker-compose build

# 3. Start container
docker-compose up -d

# 4. Check status
docker-compose ps

# 5. Enter container
docker-compose exec mtrag-retrieval bash
```

**Inside container:**
```bash
# Run experiment
python scripts/run_experiment.py --config configs/experiments/0-baselines/replication_bm25.yaml

# Check results
ls experiments/replication_bm25/*/metrics.json
```

## Common Commands

**Container management:**
```bash
docker-compose up -d              # Start
docker-compose down               # Stop
docker-compose restart            # Restart
docker-compose logs -f            # View logs
docker-compose ps                 # Check status
```

**Execute commands:**
```bash
# Enter shell
docker-compose exec mtrag-retrieval bash

# Run single command
docker-compose run --rm mtrag-retrieval python scripts/run_experiment.py --help

# Check GPU
docker-compose exec mtrag-retrieval nvidia-smi
```

**Rebuild:**
```bash
docker-compose build              # Normal rebuild
docker-compose build --no-cache   # Full rebuild
```

## Data Persistence

Directories mounted from host (changes persist):
- `./data` - Corpus and queries
- `./experiments` - Results
- `./indices` - FAISS/BM25 indices
- `./cache` - Model cache
- `./logs` - Execution logs

**Backup results:**
```bash
# Archive experiments
tar -czf results_$(date +%Y%m%d).tar.gz experiments/

# Copy from container
docker cp mtrag-retrieval:/workspace/mt-rag-benchmark/task_a_retrieval/experiments ./backup/
```

## Running Experiments

**Single experiment:**
```bash
docker-compose exec mtrag-retrieval bash
python scripts/run_experiment.py --config configs/experiments/0-baselines/replication_bm25.yaml
```

**Multiple experiments:**
```bash
# All baselines
for config in configs/experiments/0-baselines/*.yaml; do
  python scripts/run_experiment.py --config $config
done
```

**Parallel (multi-GPU):**
```bash
# GPU 0: ClapNQ
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py --config ... --domains clapnq &

# GPU 1: Cloud
CUDA_VISIBLE_DEVICES=1 python scripts/run_experiment.py --config ... --domains cloud &

wait
```

## Troubleshooting

**Container won't start:**
```bash
docker-compose logs              # Check errors
docker system prune              # Clean up
docker-compose up                # Foreground mode to see output
```

**GPU not detected:**
```bash
# Test NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Check inside container
docker-compose exec mtrag-retrieval nvidia-smi
```

**Out of memory:**
```bash
# Reduce batch size in configs/base.yaml
embedder_batch_size: 16  # Change from 32

# Use single GPU
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py ...
```

**Slow downloads:**
```bash
# Pre-download models on host
export HF_HOME=/workspace/cache
python -c "from transformers import AutoModel; AutoModel.from_pretrained('BAAI/bge-base-en-v1.5')"
```

## Additional Resources

- [README.md](README.md) - Main documentation
- [CHANGELOG.md](CHANGELOG.md) - Recent changes  
- [HuggingFace Dataset](https://huggingface.co/datasets/vania-janet/MTRAG_taskA_results) - Baseline results
