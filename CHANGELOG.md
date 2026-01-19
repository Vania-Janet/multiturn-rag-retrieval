# Changelog - MT-RAG Retrieval

## [2024-01-19] - Repository Cleanup and Docker Setup

### 🧹 Scripts Organization

#### Deleted Files (Empty/Duplicates)
- `migrate_to_split.py` (0 bytes, empty file)
- `scripts/run_experiment.py` (0 bytes, empty file - use main script)
- `scripts/summarize_metrics.py` (0 bytes, empty file)
- `generate_submission.py` (duplicate of `scripts/make_submission.py`)

#### Moved to scripts/
Organized utility scripts into `scripts/` directory:
- `extract_test_queries.py` → `scripts/extract_test_queries.py`
- `run_simple_test_retrieval.py` → `scripts/run_simple_test_retrieval.py`
- `run_test_submission.py` → `scripts/run_test_submission.py`
- `run_vllm.py` → `scripts/run_vllm.py`
- `test_finetuned_integration.py` → `scripts/test_finetuned_integration.py`

#### Analysis Document
Created `SCRIPTS_CLEANUP_ANALYSIS.md` documenting:
- Full inventory of 45 Python files in project
- Categorization by purpose (main experiments, preprocessing, utilities)
- Identification of empty files and duplicates
- Recommendations for cleanup

### 🐳 Docker Setup

#### Files Created/Updated
- ✅ `Dockerfile`: Production-ready container with NVIDIA CUDA 12.1, Python 3.10
- ✅ `docker-compose.yml`: Multi-service orchestration with GPU support
- ✅ `.dockerignore`: Build exclusions (cache, experiments, indices, etc.)
- ✅ `DOCKER_USAGE.md`: Complete Docker usage guide

#### Key Features
- **GPU Support**: NVIDIA Docker runtime with all GPUs accessible
- **Persistent Volumes**: data, experiments, indices, cache, logs
- **Environment Variables**: API keys via .env file
- **Reproducibility**: Fixed CUDA version, pinned dependencies, deterministic settings
- **Shared Memory**: 16GB shm_size for PyTorch DataLoader
- **Health Checks**: Automatic container health monitoring

### 📝 Documentation Updates

#### README.md
- ✅ Expanded Docker Quick Start with detailed instructions
- ✅ Added Docker prerequisites and volume mount documentation
- ✅ Updated reproducibility section with Docker environment details
- ✅ Updated repository structure with organized scripts/ directory
- ✅ Added useful Docker commands reference

#### New Guides
- ✅ `DOCKER_USAGE.md`: Comprehensive 300+ line Docker guide
  * Prerequisites and NVIDIA runtime installation
  * Quick start and common workflows
  * Complete command reference (build, run, debug)
  * Volume persistence and data backup
  * Troubleshooting section
  * Performance optimization tips
  * Development workflow

### 🗂️ Git History Cleanup

#### Large Files Removed
- Removed `data/submissions/` from entire Git history (files >100MB)
- Repository size reduced from several GB → **12 MB**
- Commands used:
  ```bash
  git filter-branch --force --index-filter 'git rm -rf --cached --ignore-unmatch data/submissions/'
  git reflog expire --expire=now --all
  git gc --prune=now --aggressive
  git push --force
  ```

#### .gitignore Updates
Added to `.gitignore`:
- `data/submissions/`
- `experiments/`
- `indices/`
- `cache/`

### 📊 Current Repository State

#### Root Directory Structure
```
task_a_retrieval/
├── Dockerfile                          # ✅ Production container
├── docker-compose.yml                  # ✅ GPU orchestration
├── .dockerignore                       # ✅ Build exclusions
├── DOCKER_USAGE.md                     # ✅ NEW: Docker guide
├── SCRIPTS_CLEANUP_ANALYSIS.md         # ✅ NEW: Scripts audit
├── CHANGELOG.md                        # ✅ NEW: This file
├── README.md                           # ✅ UPDATED: Docker docs
├── setup.sh                            # Local setup script
├── requirements.txt                    # Dependencies
├── configs/                            # Experiment configs
├── data/                               # Data files (gitignored)
├── scripts/                            # ✅ ORGANIZED: All utilities
├── src/                                # Source code
├── docs/                               # Documentation
├── experiments/                        # Results (gitignored)
├── indices/                            # Indices (gitignored)
├── cache/                              # Model cache (gitignored)
└── logs/                               # Execution logs
```

#### Scripts Organization (scripts/)
Now contains all utility scripts:
- `run_experiment.py` - Main experiment runner
- `make_submission.py` - Generate test submissions
- `extract_test_queries.py` - Test query extraction
- `run_test_submission.py` - Run test retrieval
- `run_vllm.py` - vLLM inference
- `test_finetuned_integration.py` - Test fine-tuned models
- `summarize_results.py` - Results aggregation
- ... (total 9 organized scripts)

### 🎯 Reproducibility Enhancements

#### Docker-based Reproducibility
- **Environment**: NVIDIA CUDA 12.1 + Python 3.10 (fixed versions)
- **Dependencies**: requirements.txt with specific versions
- **Determinism**: 
  * Fixed random seeds (42)
  * PYTHONHASHSEED=0
  * CUBLAS_WORKSPACE_CONFIG=:4096:8
  * torch.use_deterministic_algorithms(True)
- **Hardware**: Tested on NVIDIA A100 40GB
- **Caching**: Persistent model cache prevents re-downloads

#### Command Consistency
All experiments now use config files:
```bash
# Old (inconsistent)
python scripts/run_experiment.py --experiment replication_bm25 --domain all

# New (recommended)
python scripts/run_experiment.py --config configs/experiments/0-baselines/replication_bm25.yaml
```

### 🚀 Next Steps

For users:
1. Read [DOCKER_USAGE.md](DOCKER_USAGE.md) for Docker setup
2. Read [README.md](README.md) for experiment overview
3. Run `docker-compose build` to start

For developers:
1. Use Docker for consistent environment
2. All new scripts go in `scripts/` directory
3. Update configs/ rather than hardcoding parameters
4. Test changes in container before committing

### 📦 Data Availability

- **Training Data**: In repository (`data/passage_level_processed/`, `data/retrieval_tasks/`)
- **Baseline Results**: [Hugging Face Dataset](https://huggingface.co/datasets/vania-janet/MTRAG_taskA_results)
  * 679 files, 10.8 GB
  * Includes: experiments/, data/, docs/, configs/
- **Model Cache**: Auto-downloaded to `cache/` on first run

---

**Summary**: This release focuses on repository cleanup, Docker-based reproducibility, and improved documentation. The codebase is now leaner (12 MB), better organized (scripts/ directory), and fully containerized for consistent execution across machines.
