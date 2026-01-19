# Docker Usage Guide - MT-RAG Retrieval

Complete guide for using Docker to ensure reproducibility of experiments.

## 🐳 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 22.04+ recommended)
- **Docker**: 20.10 or later
- **Docker Compose**: 2.0 or later
- **GPU**: NVIDIA GPU with CUDA 12.1+ support
- **VRAM**: 16GB+ recommended for BGE-M3 models

### Install NVIDIA Docker Runtime

```bash
# Add NVIDIA Docker repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker daemon
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## 📦 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/Vania-Janet/multiturn-rag-retrieval.git
cd mt-rag-benchmark/task_a_retrieval
```

### 2. Configure Environment (Optional)

Create `.env` file for API keys (only needed for Cohere/Voyage/ELSER experiments):

```bash
cat > .env << EOF
COHERE_API_KEY=your_cohere_key
VOYAGE_API_KEY=your_voyage_key
HUGGINGFACE_TOKEN=your_hf_token
EOF
```

### 3. Build and Run

```bash
# Build Docker image (first time only, ~10-15 minutes)
docker-compose build

# Start container in background
docker-compose up -d

# Verify container is running
docker-compose ps

# Enter container shell
docker-compose exec mtrag-retrieval bash
```

### 4. Inside Container: Run Experiments

```bash
# Build FAISS indices for BGE models (first time only)
python src/pipeline/indexing/build_indices.py --models bge bge-m3 --domains all

# Run baseline experiments
python scripts/run_experiment.py --config configs/experiments/0-baselines/replication_bm25.yaml
python scripts/run_experiment.py --config configs/experiments/0-baselines/replication_bge15.yaml
python scripts/run_experiment.py --config configs/experiments/0-baselines/A1_baseline_bgem3_fullhist.yaml

# Check results
ls -lh experiments/*/*/metrics.json
```

## 🔧 Docker Commands Reference

### Container Management

```bash
# Start container (background)
docker-compose up -d

# Start container (foreground with logs)
docker-compose up

# Stop container
docker-compose down

# Restart container
docker-compose restart

# View logs
docker-compose logs -f

# Remove container and volumes (WARNING: deletes data!)
docker-compose down -v
```

### Execute Commands

```bash
# Enter interactive bash shell
docker-compose exec mtrag-retrieval bash

# Run single command without entering shell
docker-compose run --rm mtrag-retrieval python scripts/run_experiment.py --help

# Run command as specific user
docker-compose exec -u root mtrag-retrieval apt-get update
```

### Debugging

```bash
# Check GPU availability
docker-compose exec mtrag-retrieval nvidia-smi

# Check Python/PyTorch CUDA
docker-compose exec mtrag-retrieval python -c "import torch; print(torch.cuda.is_available())"

# Check disk usage
docker-compose exec mtrag-retrieval df -h

# Check running processes
docker-compose exec mtrag-retrieval ps aux
```

### Image Management

```bash
# Rebuild image after code changes
docker-compose build --no-cache

# Pull latest base image
docker pull nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# View image size
docker images | grep mtrag

# Remove old/dangling images
docker image prune
```

## 📂 Data Persistence

### Volume Mounts

The following host directories are mounted into the container:

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./data` | `/workspace/mt-rag-benchmark/task_a_retrieval/data` | Corpus and queries |
| `./experiments` | `/workspace/.../experiments` | Experiment results |
| `./indices` | `/workspace/.../indices` | FAISS/BM25 indices |
| `./cache` | `/workspace/cache` | Model cache (HF/Transformers) |
| `./logs` | `/workspace/.../logs` | Execution logs |
| `./.env` | `/workspace/.../.env` | API keys (read-only) |

**Important**: Changes made inside these mounted directories persist on the host.

### Backing Up Results

```bash
# From host: Copy experiments to backup
cp -r experiments experiments_backup_$(date +%Y%m%d)

# From host: Archive results
tar -czf results_$(date +%Y%m%d).tar.gz experiments/ logs/

# From container: Copy specific results out
docker cp mtrag-retrieval:/workspace/mt-rag-benchmark/task_a_retrieval/experiments/replication_bm25 ./backup/
```

## 🚀 Running Experiments in Docker

### Example: Full Baseline Pipeline

```bash
# Start container
docker-compose up -d && docker-compose exec mtrag-retrieval bash

# Inside container:

# 1. Build all indices (one-time, ~30 minutes)
python src/pipeline/indexing/build_indices.py --models bge bge-m3 bm25 --domains all

# 2. Run BM25 baselines (Last Turn)
python scripts/run_experiment.py --config configs/experiments/0-baselines/replication_bm25.yaml

# 3. Run BGE-1.5 baselines (Last Turn)
python scripts/run_experiment.py --config configs/experiments/0-baselines/replication_bge15.yaml

# 4. Run BGE-M3 baselines (Full History)
python scripts/run_experiment.py --config configs/experiments/0-baselines/A1_baseline_bgem3_fullhist.yaml

# 5. View results
python scripts/summarize_results.py
```

### Example: Single Domain

```bash
docker-compose exec mtrag-retrieval bash

# Run ClapNQ only
python scripts/run_experiment.py \
  --config configs/experiments/0-baselines/replication_bm25.yaml \
  --domains clapnq
```

### Example: Parallel Multi-GPU

```bash
# GPU 0: ClapNQ
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py --config ... --domains clapnq &

# GPU 1: Cloud
CUDA_VISIBLE_DEVICES=1 python scripts/run_experiment.py --config ... --domains cloud &

# Wait for both
wait
```

## 🐛 Troubleshooting

### Issue: Container won't start

```bash
# Check Docker logs
docker-compose logs

# Check if ports are in use
sudo lsof -i :8888  # Jupyter (if enabled)

# Remove old containers
docker-compose down
docker system prune
```

### Issue: GPU not detected

```bash
# Verify nvidia-docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Check docker-compose.yml has deploy.resources.reservations.devices

# Restart Docker daemon
sudo systemctl restart docker
```

### Issue: Out of memory

```bash
# Check GPU memory
docker-compose exec mtrag-retrieval nvidia-smi

# Reduce batch size in configs/base.yaml:
# embedder_batch_size: 32  →  16

# Use single GPU
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py ...
```

### Issue: Slow model downloads

```bash
# Pre-download models on host
cd cache/huggingface
git clone https://huggingface.co/BAAI/bge-base-en-v1.5

# Restart container to pick up cache
docker-compose restart
```

### Issue: Permission denied

```bash
# Enter container as root
docker-compose exec -u root mtrag-retrieval bash

# Fix permissions
chown -R $(id -u):$(id -g) /workspace/mt-rag-benchmark/task_a_retrieval
```

## 🔐 Security Best Practices

1. **API Keys**: Never commit `.env` to Git
2. **Volumes**: Use read-only mounts when possible (`:ro` suffix)
3. **Network**: Don't expose container ports unless needed
4. **Updates**: Regularly update base image for security patches

```bash
# Update base image
docker pull nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
docker-compose build --no-cache
```

## 📊 Performance Optimization

### Cache Model Weights

Pre-populate cache on host to avoid repeated downloads:

```bash
# On host
export HF_HOME=/workspace/cache
python -c "from transformers import AutoModel; AutoModel.from_pretrained('BAAI/bge-base-en-v1.5')"
```

### Shared Memory

Container uses 16GB shared memory (`shm_size: '16gb'` in docker-compose.yml) for PyTorch DataLoader. Adjust if needed.

### GPU Memory

Monitor GPU usage:
```bash
watch -n 1 nvidia-smi
```

## 📝 Development Workflow

### Edit Code on Host → Run in Container

```bash
# On host: Edit code with your IDE
vim src/pipeline/retrieval/dense_retriever.py

# In container: Run without rebuilding
docker-compose exec mtrag-retrieval python scripts/run_experiment.py ...
```

### Rebuild After Dependency Changes

```bash
# If requirements.txt changed
docker-compose build --no-cache

# Restart container
docker-compose up -d
```

## 🎯 Next Steps

1. **Read**: [README.md](README.md) for experiment details
2. **Explore**: [configs/experiments/](configs/experiments/) for available experiments
3. **Validate**: [docs/VALIDACION_ESTADISTICA_COMPLETA.md](docs/VALIDACION_ESTADISTICA_COMPLETA.md) for statistical analysis
4. **Results**: [Hugging Face Dataset](https://huggingface.co/datasets/vania-janet/MTRAG_taskA_results) for baseline results

## 📧 Support

For issues:
1. Check troubleshooting section above
2. Review container logs: `docker-compose logs`
3. Open GitHub issue with error details
