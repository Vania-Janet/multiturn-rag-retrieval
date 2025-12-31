# Complete Deployment Guide - RTX 4090 Server

**Time-optimized, copy-paste ready commands for running all experiments.**

---

## 🔌 STEP 0: Connect to Remote Server (Vast.ai via VSCode)

### Option A: Quick SSH Connection
```bash
# From your local terminal, copy SSH command from Vast.ai instance page
# Example format: ssh -p PORT_NUMBER root@ssh.vast.ai -L 8080:localhost:8080

# Replace with your actual values:
ssh -p YOUR_PORT root@YOUR_HOST.vast.ai
```

### Option B: VSCode Remote SSH (Recommended)

1. **Install VSCode Remote-SSH Extension**
   - Open VSCode
   - Go to Extensions (Cmd+Shift+X / Ctrl+Shift+X)
   - Search for "Remote - SSH"
   - Install the extension by Microsoft

2. **Get SSH Connection Details from Vast.ai**
   - Log into your Vast.ai dashboard
   - Find your running instance with 2x RTX 4090
   - Click "Connect" button
   - Copy the SSH command (looks like: `ssh -p 12345 root@ssh1.vast.ai`)

3. **Add Host to VSCode SSH Config**
   - Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
   - Type "Remote-SSH: Open SSH Configuration File"
   - Select your SSH config file (usually `~/.ssh/config`)
   - Add the following:
   ```
   Host vast-rtx4090
       HostName ssh1.vast.ai
       User root
       Port 12345
       ServerAliveInterval 60
       ServerAliveCountMax 3
   ```
   - Replace `ssh1.vast.ai` and `12345` with your actual host and port from Vast.ai

4. **Connect to Remote Server**
   - Press `Cmd+Shift+P` / `Ctrl+Shift+P`
   - Type "Remote-SSH: Connect to Host"
   - Select "vast-rtx4090" (or your chosen name)
   - VSCode will open a new window connected to your server

5. **Open Project Directory**
   - In the remote VSCode window: File → Open Folder
   - Navigate to: `/root/mt-rag-benchmark/task_a_retrieval` (or wherever you clone the repo)
   - Click OK

6. **Open Integrated Terminal**
   - Press `` Ctrl+` `` or Terminal → New Terminal
   - You're now ready to run commands directly on the remote server!

### Verify Connection
```bash
# In the VSCode terminal on remote server
hostname  # Should show vast.ai hostname
nvidia-smi  # Should show 2x RTX 4090
pwd  # Verify you're in the project directory
```

---

## ⚡ Quick Start (All Commands)

Copy this entire block and paste it into your RTX 4090 server terminal:

```bash
# ============================================================================
# STEP 1: INITIAL SETUP & HARDWARE VERIFICATION
# ============================================================================

# Navigate to project directory
cd /path/to/mt-rag-benchmark/task_a_retrieval

# Verify ALL GPUs are available (should show 2x RTX 4090)
nvidia-smi
echo "Expected: 2x RTX 4090 GPUs"

# Check CPU cores (for optimal thread settings)
nproc
echo "CPU cores detected"

# Verify .env file exists with API keys
cat .env | grep -E "OPENAI_API_KEY|VOYAGE_API_KEY|COHERE_API_KEY"

# ============================================================================
# STEP 2: START ELASTICSEARCH (Required for ELSER)
# ============================================================================

docker-compose up -d elasticsearch

# Wait for Elasticsearch to be healthy (30-60 seconds)
echo "Waiting for Elasticsearch to start..."
sleep 60

# Verify Elasticsearch is running
curl http://localhost:9200

# ============================================================================
# STEP 3: BUILD ALL INDICES (BM25, BGE-M3, ELSER)
# ============================================================================

# Start the main application container with GPU support
docker-compose up -d mt-rag-app

# Enter the container
docker exec -it mt-rag-app bash

# Inside container, run:
# ----------------------------

# VERIFY MULTI-GPU SETUP INSIDE CONTAINER
nvidia-smi
python -c "import torch; print(f'PyTorch detects {torch.cuda.device_count()} GPUs')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install dependencies (if not already in image)
pip install nltk rank-bm25
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# OPTIMIZE FOR 2x RTX 4090: Set environment variables for maximum performance
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export CUDA_VISIBLE_DEVICES=0,1  # Use both GPUs
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Optimize memory allocation

# Build BM25 indices for all domains (~5 minutes, CPU-bound, uses all cores)
python scripts/build_indices.py --domain all --model bm25 --corpus-dir data/passage_level_processed

# Build BGE-M3 indices for all domains (~15-20 minutes on 2x RTX 4090 with multi-GPU)
# Note: FAISS will automatically use both GPUs during index building
python scripts/build_indices.py --domain all --model bge-m3 --corpus-dir data/passage_level_processed

# Build ELSER indices for all domains (requires Elasticsearch, ~45 minutes)
python scripts/build_indices.py --domain all --model elser --corpus-dir data/passage_level_processed

# Verify all indices were built successfully
ls -lh indices/clapnq/
ls -lh indices/cloud/
ls -lh indices/fiqa/
ls -lh indices/govt/

# Exit container
exit

# ============================================================================
# RE-APPLY PERFORMANCE OPTIMIZATIONS (if you exited container)
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export CUDA_VISIBLE_DEVICES=0,1  # Use both RTX 4090 GPUs
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Verify multi-GPU is active
python -c "import torch; print(f'Using {torch.cuda.device_count()} GPUs for experiments')"

# STEP 4: RUN ALL EXPERIMENTS
# ============================================================================

# Re-enter container
docker exec -it mt-rag-app bash

# Inside container:
# ----------------------------

# Set PYTHONPATH
export PYTHONPATH=/workspace/src:$PYTHONPATH

# Create logs directory
mkdir -p logs

# ============================================================================
# RUN BASELINES (No query rewriting)
# ============================================================================

# BM25 Baseline - All Domains
for domain in clapnq cloud fiqa govt; do
    echo "Running BM25 baseline on $domain"
    python scripts/run_experiment.py \
        --config configs/experiments/0-baselines/replication_bm25.yaml \
        --domain $domain \
        --output experiments/baseline_bm25/$domain
done

# BGE-M3 Baseline - All Domains
for domain in clapnq cloud fiqa govt; do
    echo "Running BGE-M3 baseline on $domain"
    python scripts/run_experiment.py \
        --config configs/experiments/0-baselines/replication_bgem3.yaml \
        --domain $domain \
        --output experiments/baseline_bgem3/$domain
done

# ELSER Baseline - All Domains
for domain in clapnq cloud fiqa govt; do
    echo "Running ELSER baseline on $domain"
    python scripts/run_experiment.py \
        --config configs/experiments/0-baselines/replication_elser.yaml \
        --domain $domain \
        --output experiments/baseline_elser/$domain
done

# ============================================================================
# RUN QUERY REWRITING ABLATIONS (coref, multi, hyde)
# ============================================================================

# BGE-M3 + Coref Rewriting (R1)
for domain in clapnq cloud fiqa govt; do
    echo "Running BGE-M3 + Coref on $domain"
    python scripts/run_experiment.py \
        --config configs/experiments/01-query/bgem3_r1_coref.yaml \
        --domain $domain \
        --output experiments/bgem3_r1_coref/$domain
done

# BGE-M3 + Multi-Query Expansion (R2)
for domain in clapnq cloud fiqa govt; do
    echo "Running BGE-M3 + Multi on $domain"
    python scripts/run_experiment.py \
        --config configs/experiments/01-query/bgem3_r2_multi.yaml \
        --domain $domain \
        --output experiments/bgem3_r2_multi/$domain
done

# BM25 + Coref Rewriting (R1)
for domain in clapnq cloud fiqa govt; do
    echo "Running BM25 + Coref on $domain"
    python scripts/run_experiment.py \
        --config configs/experiments/01-query/bm25_r1_coref.yaml \
        --domain $domain \
        --output experiments/bm25_r1_coref/$domain
done

# BM25 + Multi-Query Expansion (R2)
for domain in clapnq cloud fiqa govt; do
    echo "Running BM25 + Multi on $domain"
    python scripts/run_experiment.py \
        --config configs/experiments/01-query/bm25_r2_multi.yaml \
        --domain $domain \
        --output experiments/bm25_r2_multi/$domain
done

# ELSER + Coref Rewriting (R1)
for domain in clapnq cloud fiqa govt; do
    echo "Running ELSER + Coref on $domain"
    python scripts/run_experiment.py \
        --config configs/experiments/01-query/elser_r1_coref.yaml \
        --domain $domain \
        --output experiments/elser_r1_coref/$domain
done

# ELSER + HyDE (R3)
for domain in clapnq cloud fiqa govt; do
    echo "Running ELSER + HyDE on $domain"
    python scripts/run_experiment.py \
        --config configs/experiments/01-query/elser_r3_hyde.yaml \
        --domain $domain \
        --output experiments/elser_r3_hyde/$domain
done

# ============================================================================
# RUN HYBRID RETRIEVAL EXPERIMENTS
# ============================================================================

# Hybrid ELSER + BGE-M3 (No Rewrite)
for domain in clapnq cloud fiqa govt; do
    echo "Running Hybrid ELSER+BGE-M3 (no rewrite) on $domain"
    python scripts/run_experiment.py \
        --config configs/experiments/02-hybrid/hybrid_elser_bgem3_norewrite.yaml \
        --domain $domain \
        --output experiments/hybrid_elser_bgem3_norewrite/$domain
done

# Hybrid ELSER + BGE-M3 + R1 Coref
for domain in clapnq cloud fiqa govt; do
    echo "Running Hybrid ELSER+BGE-M3 + Coref on $domain"
    python scripts/run_experiment.py \
        --config configs/experiments/02-hybrid/hybrid_elser_bgem3_r1.yaml \
        --domain $domain \
        --output experiments/hybrid_elser_bgem3_r1/$domain
done

# ============================================================================
# STEP 5: VERIFY RESULTS
# ============================================================================

# Check that all experiments produced results
echo "Verifying experiment outputs..."

find experiments/ -name "metrics.json" | wc -l
# Should see 60+ results files (15 experiments × 4 domains)

# Display sample results
echo "Sample results:"
cat experiments/baseline_bgem3/clapnq/metrics.json | jq '.ndcg_at_10'

# ============================================================================
# STEP 6: COLLECT ALL RESULTS
# ============================================================================

# Create results summary
python scripts/summarize_results.py --experiments experiments/ --output results_summary.csv

# Exit container
exit

# ============================================================================
# STEP 7: COPY RESULTS TO HOST (if needed)
# ============================================================================

# Copy results from container to host
docker cp mt-rag-app:/workspace/experiments ./experiments_results
docker cp mt-rag-app:/workspace/logs ./logs_backup

# Stop containers (optional)
docker-compose down

echo "✅ All experiments completed! Results are in ./experiments/ directory"

# ============================================================================
# STEP 8: SYNC ARTIFACTS TO HUGGING FACE (Optional)
# ============================================================================

# If you need to upload indices, models, or results to Hugging Face Hub
# This script handles large files (indices, artifacts) automatically.
# It uses the configuration in scripts/hf_sync.py to identify large directories.

# 1. Login to Hugging Face (if not already logged in)
huggingface-cli login

# 2. Run the sync script
# This will upload:
# - indices/ (FAISS, BM25, etc.)
# - artifacts/ (Fine-tuned models)
# - experiments/ (Results and logs)
python scripts/hf_sync.py

```

---

## � Checkpointing & Crash Recovery

**Your experiments are safe!** The pipeline has multiple checkpoint mechanisms:

### ✅ Indexing Checkpoints
- **Checkpoint Files**: Each index creates intermediate files (`embeddings.npy`, `doc_ids.json`)
- **Success Flag**: `_SUCCESS` marker file indicates completed indices
- **Recovery**: If indexing fails, it will automatically resume from the last checkpoint
- **Location**: `indices/{domain}/{model}/_SUCCESS`

### ✅ Experiment Results Auto-Saved
- **Per-Domain Saves**: Results are saved immediately after EACH domain completes
- **Files Created**:
  - `retrieval_results.jsonl` - Raw retrieval results (saved per query)
  - `metrics.json` - Evaluation metrics (saved after domain completion)
  - `analysis.json` - Performance analysis
  - `config_resolved.yaml` - Exact config used
- **Location**: `experiments/{experiment_name}/{domain}/`

### ✅ Comprehensive Logging
- **Log Files**:
  - `logs/build_indices.log` - Indexing progress and errors
  - `logs/experiment.log` - Experiment execution details
  - `logs/{experiment_name}_{timestamp}.log` - Per-experiment logs
- **Real-time Updates**: Logs are flushed immediately (no buffering)
- **Recovery Info**: Logs show exactly which queries were processed

### 🔄 How to Resume After Internet Loss

If your SSH connection drops:

1. **Reconnect to server**:
   ```bash
   ssh -p YOUR_PORT root@YOUR_HOST.vast.ai
   cd /path/to/mt-rag-benchmark/task_a_retrieval
   ```

2. **Check what completed**:
   ```bash
   # Check indexing status
   find indices/ -name "_SUCCESS"
   
   # Check experiment results
   find experiments/ -name "metrics.json"
   
   # View last log entries
   tail -n 50 logs/build_indices.log
   ```

3. **Resume from where you left off**:
   ```bash
   # If indexing was interrupted (indices will skip completed ones)
   docker exec -it mt-rag-app bash
   python scripts/build_indices.py --domain all --model bge-m3 --corpus-dir data/passage_level_processed
   
   # If experiments were interrupted (manually re-run failed domains)
   # Example: If clapnq completed but cloud failed
   python scripts/run_experiment.py \
       --config configs/experiments/0-baselines/replication_bgem3.yaml \
       --domain cloud \
       --output experiments/baseline_bgem3/cloud
   ```

### 🛡️ Best Practices for Long Runs

1. **Use `tmux` or `screen`** (already in guide) - keeps processes alive after SSH disconnect
2. **Monitor progress**: Check log files periodically
3. **Verify after each phase**:
   ```bash
   # After indexing
   ls -lh indices/clapnq/*/_SUCCESS
   
   # After experiments
   find experiments/ -name "metrics.json" | wc -l
   ```

---

## �📋 Pre-Flight Checklist

Before running the above commands, verify:

- [ ] **GPU Available**: `nvidia-smi` shows RTX 4090
- [ ] **Docker GPU Runtime**: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi` works
- [ ] **API Keys Set**: `.env` file contains:
  - `OPENAI_API_KEY=sk-...`
  - `VOYAGE_API_KEY=pa-...` (optional, only for Voyage experiments)
  - `COHERE_API_KEY=...` (optional, only for reranking experiments)
- [ ] **Disk Space**: At 2x RTX 4090 + Multi-Core CPU)

| Phase | Time | Details |
|-------|------|---------|
| **Index Building** | **60 min** | BM25 (5m, all CPU cores) + BGE-M3 (15-20m, 2 GPUs) + ELSER (35-40m, 2 GPUs) |
| **Baseline Experiments** | **30 min** | 3 methods × 4 domains × ~2-3 min/run (GPU parallel) |
| **Query Rewriting** | **60 min** | 6 configs × 4 domains × ~2-3 min/run (GPU parallel) |
| **Hybrid Experiments** | **20 min** | 2 configs × 4 domains × ~2-3 min/run (GPU parallel) |
| **TOTAL** | **~3 hours** | For all 60+ experiment runs |

**Performance Optimizations Applied:**
- ✅ **2x RTX 4090 GPUs**: FAISS indices distributed across both GPUs during search
- ✅ **Multi-Core CPU**: BM25 indexing uses all available CPU threads
- ✅ **FP16 Precision**: Embedding models run in half precision (2x speedup)
- ✅ **Large Batch Sizes**: 1024 for indexing, 128 for retrieval
- ✅ **Memory Optimization**: `PYTORCH_CUDA_ALLOC_CONF` prevents fragmentation

*Note: Times assume warm cache, no API rate limits, and proper multi-GPU detection
| Phase | Time | Details |
|-------|------|---------|
| **Index Building** | 80 min | BM25 (5m) + BGE-M3 (30m) + ELSER (45m) |
| **Baseline Experiments** | 45 min | 3 methods × 4 domains × ~3-4 min/run |
| **Query Rewriting** | 90 min | 6 configs × 4 domains × ~3-4 min/run |
| **Hybrid Experiments** | 30 min | 2 configs × 4 domains × ~3-4 min/run |
| **TOTAL** | **~4 hours** | For all 60+ experiment runs |

*Note: Times assume warm cache and no API rate limits.*

---

## 🔧 Troubleshooting

### If Elasticsearch fails to start:
```bash
docker-compose logs elasticsearch
# Check if port 9200 is already in use
sudo lsof -i :9200
```

### If GPU is nBOTH GPUs during experiments:
```bash
# In a separate terminal/tmux pane
watch -n 1 nvidia-smi

# Or check current GPU utilization
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv

# Verify both GPUs are being used
nvidia-smi dmon -s u
```

### If only 1 GPU is being used:
```bash
# Verify CUDA_VISIBLE_DEVICES is set correctly
echo $CUDA_VISIBLE_DEVICES
# Should output: 0,1

# Force PyTorch to see both GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Verify
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"iner Toolkit is installed
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### If indexing fails with NLTK errors:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### If experiments fail with import errors:
```bash
export PYTHONPATH=/workspace/src:$PYTHONPATH
```

### To monitor GPU usage during experiments:
```bash for Maximum Performance

1. **Use `tmux` with split panes** to monitor GPUs while experiments run:
   ```bash
   tmux new -s rag-experiments
   
   # Split pane horizontally (Ctrl+B, then ")
   # Top pane: Run experiments
   # Bottom pane: watch -n 1 nvidia-smi
   
   # Detach: Ctrl+B, then D
   # Reattach: tmux attach -t rag-experiments
   ```

2. **Verify BOTH GPUs are at >80% utilization**:
   - During BGE-M3 indexing: Both GPUs should show ~90%+ usage
   - During retrieval: Both GPUs should share FAISS index queries
   - If only 1 GPU is active, check `CUDA_VISIBLE_DEVICES=0,1`

3. **Monitor CPU during BM25 indexing**: Should see all cores at 100%
   ```bash
   htop  # or top
   ```

4. **Batch experiments efficiently**: 
   - Run CPU-bound (BM25) and GPU-bound (BGE-M3) experiments in parallel
   - Example: BM25 experiment in one terminal, BGE-M3 in another

5. **Save intermediate results**: The script saves results after each domain, so you can interrupt safely.

6. **Cost optimization**: 
   - If GPU utilization is <50%, investigate bottlenecks (API rate limits, disk I/O)
   - Consider running experiments for 1 domain first to validate performance
# Example: BGE-M3 baseline on CLAPNQ
python scripts/run_experiment.py \
    --config configs/experiments/0-baselines/replication_bgem3.yaml \
    --domain clapnq \
    --output experiments/baseline_bgem3/clapnq

# Check the output
cat experiments/baseline_bgem3/clapnq/metrics.json
```

---

## 🚀 Production Tips

1. **Use `tmux` or `screen`** to keep experiments running if SSH disconnects:
   ```bash
   tmux new -s rag-experiments
   # Run all commands
   # Detach: Ctrl+B, then D
   # Reattach: tmux attach -t rag-experiments
   ```

2. **Monitor costs**: Check GPU usage with `nvidia-smi` and stop immediately if utilization is unexpectedly low.

3. **Batch experiments**: If you only need specific ablations (e.g., only coref + multi), comment out the hyde experiments.

4. **Save intermediate results**: The script saves results after each domain, so you can interrupt safely.

---

**Estimated Total Cost**: ~3 hours × your instance hourly rate **(25% faster with 2 GPUs!)**

**Hardware Utilization Summary:**
- 🔥 **2x RTX 4090**: ~90%+ utilization during indexing and retrieval
- 💻 **All CPU Cores**: 100% utilization during BM25 indexing
- 🧠 **Memory**: ~24GB VRAM per GPU, ~32GB System RAM
- 📊 **Expected Throughput**: ~150 documents/sec encoding, ~5000 queries/sec retrieval

**Pro tip**: Run during off-peak hours if your provider has variable pricing.
