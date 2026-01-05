# Complete Deployment Guide - RTX 4090 Server

**Time-optimized, copy-paste ready commands for running all experiments.**

---

## ☁️ OPTION A: RunPod Deployment (Quick Start)

**Critical path to get your environment up and running on RunPod.**

### Step 1: Configure Instance
1. Create account on **RunPod.io** and load $10-20 USD.
2. Go to **Pods > Deploy**.
3. **Select GPU**: Search for **RTX 4090**. Choose "2-GPU" (recommended) or "1-GPU".
4. **Template (Docker Image)**: Select **RunPod Pytorch 2.1** (or newer). Includes CUDA, Python, Torch.
5. **Customize Deployment**:
   - **Container Disk**: 20 GB (for libraries).
   - **Volume Disk**: 50 GB or 100 GB (Persistent storage for indices/datasets).
   - *Note: Only `/workspace` persists if pod is turned off.*

### Step 2: Connection
Once Pod is "Running":
1. Click **Connect**.
2. **Option A (Easy)**: Click **Jupyter Lab** (web browser).
3. **Option B (Pro)**: Copy SSH command (`ssh root@... -p ...`) and use in VS Code (Remote - SSH).

### Step 3: Upload Data & Scripts
Using Jupyter Lab upload or SFTP (FileZilla):
1. Upload project folder to `/workspace/mt-rag`.
2. Upload `data/` folder to `/workspace/mt-rag/data`.

### Step 4: Install Dependencies
Open terminal in Pod:
```bash
cd /workspace/mt-rag

# Basic tools
apt-get update && apt-get install -y git htop screen

# Python libraries
# (Torch is usually pre-installed in the template)
pip install sentence-transformers faiss-gpu rank_bm25 elasticsearch python-dotenv openai pytrec_eval pandas numpy tqdm pyyaml huggingface_hub
```

**Elasticsearch (ELSER) Note:**
- **Option A (Easy)**: Skip ELSER experiments for now (comment out in configs).
- **Option B (Full)**: Install Java & Elasticsearch manually in the pod.
  ```bash
  wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.11.1-linux-x86_64.tar.gz
  tar -xzf elasticsearch-8.11.1-linux-x86_64.tar.gz
  # Configure and run in background...
  ```

### Step 5: Run Experiment
Use `screen` or `tmux` to keep sessions alive.

```bash
screen -S experiment1

# Inside screen session
export OPENAI_API_KEY="sk-..."
export HF_TOKEN="hf_..."

# Validate
python scripts/validate_code.py

# Run
python scripts/run_experiment.py -e replication_bm25 -d fiqa

# Detach: Ctrl+A, then D
```

---

## 🔌 OPTION B: Connect to Remote Server (Vast.ai via VSCode)

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

## ⚡ Quick Start (All Commands - Bare Metal)

**IMPORTANT**: RunPod/Vast.ai instances are already Docker containers. We run everything directly on the instance (no Docker-in-Docker).

Copy this entire block and paste it into your RTX 4090 server terminal:

```bash
# ============================================================================
# STEP 0: INITIAL SETUP (BARE METAL)
# ============================================================================

# Navigate to project directory (adjust path if needed)
cd /workspace/mt-rag-benchmark/task_a_retrieval  # RunPod uses /workspace
# OR: cd /root/mt-rag-benchmark/task_a_retrieval  # Vast.ai uses /root

# Verify ALL GPUs are available (should show 2x RTX 4090)
nvidia-smi
echo "Expected: 2x RTX 4090 GPUs"

# Check CPU cores (for optimal thread settings)
nproc
echo "CPU cores detected"

# Install system dependencies
apt-get update && apt-get install -y git htop screen default-jdk curl

# Install Python dependencies
pip install sentence-transformers faiss-gpu rank_bm25 elasticsearch \
    python-dotenv openai pytrec_eval pandas numpy tqdm pyyaml \
    huggingface_hub cohere backoff

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Set environment variables for 2x RTX 4090 optimization
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export CUDA_VISIBLE_DEVICES=0,1  # Use both GPUs
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Verify GPU setup
python -c "import torch; print(f'PyTorch detects {torch.cuda.device_count()} GPUs')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Verify .env file exists with API keys
cat .env | grep -E "OPENAI_API_KEY|VOYAGE_API_KEY|COHERE_API_KEY"

# ============================================================================
# STEP 1: START ELASTICSEARCH (Required for ELSER)
# ============================================================================

# Elasticsearch cannot run as root. Create a dedicated user.
useradd -m elasticsearch_user
chown -R elasticsearch_user:elasticsearch_user /workspace

# Download and extract Elasticsearch
cd /workspace
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.11.1-linux-x86_64.tar.gz
tar -xzf elasticsearch-8.11.1-linux-x86_64.tar.gz
chown -R elasticsearch_user:elasticsearch_user elasticsearch-8.11.1

# Start Elasticsearch in background as non-root user
su - elasticsearch_user -c "/workspace/elasticsearch-8.11.1/bin/elasticsearch -d -p /workspace/elasticsearch.pid"

# Wait for Elasticsearch to be healthy
echo "Waiting for Elasticsearch to start..."
sleep 60

# Verify Elasticsearch is running
curl http://localhost:9200

# Return to project directory
cd /workspace/mt-rag-benchmark/task_a_retrieval

# ============================================================================
# STEP 2: BUILD ALL INDICES (BM25, BGE-M3, ELSER)
# ============================================================================

# Use 'screen' to keep the process running if SSH disconnects
screen -S indexing

# Build BM25 indices for all domains (~5 minutes, CPU-bound, uses all cores)
python scripts/build_indices.py --domains clapnq cloud fiqa govt \
    --models bm25 --data_dir data/passage_level_processed

# Build BGE-M3 indices for all domains (~15-20 minutes on 2x RTX 4090)
# Note: Will automatically use both GPUs via multi-process encoding
python scripts/build_indices.py --domains clapnq cloud fiqa govt \
    --models bge-m3 --data_dir data/passage_level_processed

# Build ELSER indices for all domains (requires Elasticsearch, ~45 minutes)
python scripts/build_indices.py --domains clapnq cloud fiqa govt \
    --models elser --data_dir data/passage_level_processed

# Verify all indices were built successfully
ls -lh indices/clapnq/
ls -lh indices/cloud/
ls -lh indices/fiqa/
ls -lh indices/govt/

# Detach from screen: Ctrl+A, then D
# To reattach later: screen -r indexing

# ============================================================================
# STEP 3: RUN ALL EXPERIMENTS
# ============================================================================

# Create new screen session for experiments
screen -S experiments

# Set PYTHONPATH
export PYTHONPATH=/workspace/mt-rag-benchmark/task_a_retrieval/src:$PYTHONPATH

# Create logs directory
mkdir -p logs

# Validate code before starting
python scripts/validate_code.py

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
# RUN RERANKING EXPERIMENTS (Cohere v4.0-pro)
# ============================================================================

# Hybrid + Cohere Rerank (No Rewrite)
for domain in clapnq cloud fiqa govt; do
    echo "Running Hybrid + Cohere Rerank (no rewrite) on $domain"
    python scripts/run_experiment.py \
        --config configs/experiments/03-rerank/hybrid_cohere_norewrite.yaml \
        --domain $domain \
        --output experiments/hybrid_cohere_norewrite/$domain
done

# Hybrid + Cohere Rerank + R1 Coref
for domain in clapnq cloud fiqa govt; do
    echo "Running Hybrid + Cohere Rerank + Coref on $domain"
    python scripts/run_experiment.py \
        --config configs/experiments/03-rerank/hybrid_cohere_r1.yaml \
        --domain $domain \
        --output experiments/hybrid_cohere_r1/$domain
done

# Detach from screen: Ctrl+A, then D
# To reattach later: screen -r experiments

# ============================================================================
# STEP 4: VERIFY RESULTS
# ============================================================================

# Check that all experiments produced results
echo "Verifying experiment outputs..."

find experiments/ -name "metrics.json" | wc -l
# Should see 60+ results files (15 experiments × 4 domains)

# Display sample results
echo "Sample results:"
cat experiments/baseline_bgem3/clapnq/metrics.json | jq '.ndcg_at_10'

# ============================================================================
# STEP 5: SYNC ARTIFACTS TO HUGGING FACE (Optional)
# ============================================================================

# If you need to upload indices, models, or results to Hugging Face Hub
# This script handles large files (indices, artifacts) automatically.

# 1. Login to Hugging Face (if not already logged in)
huggingface-cli login

# 2. Run the sync script
# This will upload:
# - indices/ (FAISS, BM25, etc.)
# - artifacts/ (Fine-tuned models)
# - experiments/ (Results and logs)
python scripts/hf_sync.py

# ============================================================================
# STEP 6: CLEANUP (Optional)
# ============================================================================

# Stop Elasticsearch to free up resources
pkill -F /workspace/elasticsearch.pid

echo "✅ All experiments completed! Results are in ./experiments/ directory"

```

---

## 🔄 Checkpointing & Crash Recovery

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

1. **Reattach to screen session**:
   ```bash
   # Reconnect to server
   ssh -p YOUR_PORT root@YOUR_HOST.vast.ai  # Or vast.ai hostname
   cd /workspace/mt-rag-benchmark/task_a_retrieval

   # List available screen sessions
   screen -ls
   
   # Reattach to your session
   screen -r indexing   # Or: screen -r experiments
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
   # If indexing was interrupted (script will skip completed indices)
   python scripts/build_indices.py --domains clapnq cloud fiqa govt \
       --models bge-m3 --data_dir data/passage_level_processed
   
   # If experiments were interrupted (manually re-run failed domains)
   # Example: If clapnq completed but cloud failed
   python scripts/run_experiment.py \
       --config configs/experiments/0-baselines/replication_bgem3.yaml \
       --domain cloud \
       --output experiments/baseline_bgem3/cloud
   ```

### 🛡️ Best Practices for Long Runs

1. **Use `screen`** (already in guide) - keeps processes alive after SSH disconnect:
   ```bash
   # Create session
   screen -S my_experiment
   
   # Run your commands...
   
   # Detach: Ctrl+A, then D
   # Reattach later: screen -r my_experiment
   ```

2. **Monitor progress**: Check log files periodically
   ```bash
   # Watch logs in real-time
   tail -f logs/build_indices.log
   ```

3. **Verify after each phase**:
   ```bash
   # After indexing
   ls -lh indices/clapnq/*/_SUCCESS
   
   # After experiments
   find experiments/ -name "metrics.json" | wc -l
   ```

4. **Save work to persistent storage**:
   - **RunPod**: Only `/workspace` persists across restarts
   - **Vast.ai**: `/root` persists, but verify with your provider
   - Always keep backups on Hugging Face Hub (see Step 5 in Quick Start)

---

## 📋 Pre-Flight Checklist

Before running the above commands, verify:

- [ ] **GPU Available**: `nvidia-smi` shows 2x RTX 4090
- [ ] **Python Environment**: `python --version` shows Python 3.8+
- [ ] **PyTorch with CUDA**: `python -c "import torch; print(torch.cuda.is_available())"` returns `True`
- [ ] **API Keys Set**: `.env` file contains:
  - `OPENAI_API_KEY=sk-...` (required for query rewriting)
  - `COHERE_API_KEY=...` (required for reranking experiments)
  - `HF_TOKEN=hf_...` (optional, for Hugging Face uploads)
- [ ] **Disk Space**: At least 100 GB free
  ```bash
  df -h /workspace  # RunPod
  # OR
  df -h /root       # Vast.ai
  ```
- [ ] **Data Files Present**:
  ```bash
  ls data/passage_level_processed/
  # Should show: clapnq/ cloud/ fiqa/ govt/
  ```

---

## ⏱️ Expected Runtime (2x RTX 4090 + Multi-Core CPU)

| Phase | Time | Details |
|-------|------|---------|
| **Index Building** | **60 min** | BM25 (5m, all CPU cores) + BGE-M3 (15-20m, 2 GPUs) + ELSER (35-40m) |
| **Baseline Experiments** | **30 min** | 3 methods × 4 domains × ~2-3 min/run |
| **Query Rewriting** | **60 min** | 6 configs × 4 domains × ~2-3 min/run |
| **Hybrid Experiments** | **20 min** | 2 configs × 4 domains × ~2-3 min/run |
| **Reranking Experiments** | **20 min** | 2 configs × 4 domains × ~2-3 min/run (Cohere API) |
| **TOTAL** | **~3 hours** | For all 70+ experiment runs |

**Performance Optimizations Applied:**
- ✅ **2x RTX 4090 GPUs**: Multi-process encoding uses both GPUs simultaneously
- ✅ **Multi-Core CPU**: BM25 indexing uses all available CPU threads
- ✅ **FP16 Precision**: Embedding models run in half precision (2x speedup)
- ✅ **Optimized Batch Sizes**: 256 for indexing, 128 for retrieval (from `configs/base.yaml`)
- ✅ **Memory Optimization**: `PYTORCH_CUDA_ALLOC_CONF` prevents fragmentation

*Note: Times assume warm cache, no API rate limits, and proper multi-GPU detection.*

---

## 🔧 Troubleshooting

### If Elasticsearch fails to start:
```bash
# Check Elasticsearch logs
tail -f /workspace/elasticsearch-8.11.1/logs/elasticsearch.log

# Check if port 9200 is already in use
sudo lsof -i :9200

# Kill and restart Elasticsearch
pkill -F /workspace/elasticsearch.pid
su - elasticsearch_user -c "/workspace/elasticsearch-8.11.1/bin/elasticsearch -d -p /workspace/elasticsearch.pid"
```

### If Elasticsearch permission errors:
```bash
# Ensure the elasticsearch user owns the directory
chown -R elasticsearch_user:elasticsearch_user /workspace/elasticsearch-8.11.1

# Restart Elasticsearch
su - elasticsearch_user -c "/workspace/elasticsearch-8.11.1/bin/elasticsearch -d -p /workspace/elasticsearch.pid"
```

### If GPU is not detected:
```bash
# Check GPU visibility
nvidia-smi

# Verify PyTorch can see GPUs
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPUs detected: {torch.cuda.device_count()}')"

# Set CUDA_VISIBLE_DEVICES if needed
export CUDA_VISIBLE_DEVICES=0,1
```

### If only 1 GPU is being used:
```bash
# Verify CUDA_VISIBLE_DEVICES is set correctly
echo $CUDA_VISIBLE_DEVICES
# Should output: 0,1

# Check GPU utilization during indexing
watch -n 1 nvidia-smi

# Verify multi-GPU encoding is enabled
python -c "from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer('BAAI/bge-m3'); \
    pool = model.start_multi_process_pool(); \
    print('Multi-GPU pool started successfully')"
```

### If indexing fails with NLTK errors:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### If experiments fail with import errors:
```bash
# Set PYTHONPATH to include src directory
export PYTHONPATH=/workspace/mt-rag-benchmark/task_a_retrieval/src:$PYTHONPATH

# Verify imports work
python -c "from pipeline.retrieval import DenseRetriever; print('Import successful')"
```

### If API rate limits are hit:
```bash
# OpenAI rate limits (query rewriting experiments)
# - Solution: The code already has exponential backoff. Just wait.
# - Alternative: Use a higher tier API key with increased limits.

# Cohere rate limits (reranking experiments)
# - Solution: The code uses backoff retry logic (max 3 retries).
# - Alternative: Contact Cohere for increased rate limits.
```

### To monitor GPU usage during experiments:
```bash
# Option 1: Use watch command
watch -n 1 nvidia-smi

# Option 2: Use nvidia-smi in monitoring mode
nvidia-smi dmon -s u

# Option 3: Check GPU memory and utilization
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
```

---

## 💡 Tips for Maximum Performance

1. **Use `screen` with split monitoring** to watch GPUs while experiments run:
   ```bash
   screen -S rag-experiments
   
   # In screen, you can create a monitoring pane
   # Ctrl+A, then | (vertical split) or " (horizontal split)
   # In one pane: run experiments
   # In another pane: watch -n 1 nvidia-smi
   
   # Detach: Ctrl+A, then D
   # Reattach: screen -r rag-experiments
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
   - All experiments in the Quick Start guide run sequentially
   - If you want to parallelize, run different domains in separate screen sessions
   - Example:
     ```bash
     screen -S clapnq
     python scripts/run_experiment.py --config ... --domain clapnq
     # Ctrl+A, D (detach)
     
     screen -S cloud
     python scripts/run_experiment.py --config ... --domain cloud
     # Ctrl+A, D (detach)
     ```

5. **Save intermediate results**: The script saves results after each domain, so you can interrupt safely.

6. **Cost optimization**: 
   - If GPU utilization is <50%, investigate bottlenecks (API rate limits, disk I/O)
   - Consider running experiments for 1 domain first to validate performance
   - Monitor costs: Check GPU usage with `nvidia-smi` and stop if utilization is low

---

## 🚀 Production Tips

1. **Persistent Sessions**: Use `screen` (already in guide) to keep experiments running if SSH disconnects.

2. **Data Persistence**:
   - **RunPod**: Only `/workspace` survives pod restarts. Keep everything there.
   - **Vast.ai**: `/root` persists, but verify with your provider.
   - **Best Practice**: Upload critical results to Hugging Face Hub immediately after completion.

3. **Monitoring Dashboard**:
   ```bash
   # Create a monitoring script
   cat > /workspace/monitor.sh << 'EOF'
   #!/bin/bash
   while true; do
     clear
     echo "=== GPU Status ==="
     nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
     echo ""
     echo "=== Elasticsearch Status ==="
     curl -s http://localhost:9200/_cluster/health | jq -r '.status'
     echo ""
     echo "=== Completed Experiments ==="
     find experiments/ -name "metrics.json" | wc -l
     echo ""
     sleep 5
   done
   EOF
   chmod +x /workspace/monitor.sh
   
   # Run in a separate screen session
   screen -S monitor
   /workspace/monitor.sh
   ```

4. **Backup Strategy**:
   ```bash
   # After each major phase, sync to HF Hub
   python scripts/hf_sync.py
   
   # Or manually backup critical files
   tar -czf results_backup_$(date +%Y%m%d_%H%M%S).tar.gz experiments/ indices/
   ```

---

**Estimated Total Cost**: ~3 hours × your instance hourly rate

**Hardware Utilization Summary:**
- 🔥 **2x RTX 4090**: ~90%+ utilization during indexing and retrieval
- 💻 **All CPU Cores**: 100% utilization during BM25 indexing
- 🧠 **Memory**: ~20-24GB VRAM per GPU, ~32-48GB System RAM
- 📊 **Expected Throughput**: ~1000 docs/sec encoding (multi-GPU), ~5000 queries/sec retrieval

**Pro tip**: Run during off-peak hours if your provider has variable pricing.


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
