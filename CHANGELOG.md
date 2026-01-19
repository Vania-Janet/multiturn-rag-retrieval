# Changelog

## [2024-01-19] - Repository Cleanup and Docker Setup

### Scripts Organization

**Deleted files:**
- `migrate_to_split.py`, `scripts/run_experiment.py`, `scripts/summarize_metrics.py` (empty)
- `generate_submission.py` (duplicate)

**Moved to scripts/:**
- `extract_test_queries.py`
- `run_simple_test_retrieval.py`
- `run_test_submission.py`
- `run_vllm.py`
- `test_finetuned_integration.py`

**Analysis:** See [SCRIPTS_CLEANUP_ANALYSIS.md](SCRIPTS_CLEANUP_ANALYSIS.md)

### Docker Setup

**Files:**
- `Dockerfile` - NVIDIA CUDA 12.1 + Python 3.10
- `docker-compose.yml` - GPU support, volume mounts
- `.dockerignore` - Build optimization
- `DOCKER_USAGE.md` - Usage guide

**Features:**
- GPU support (NVIDIA Docker runtime)
- Persistent volumes (data, experiments, indices, cache)
- Deterministic settings (seeds, CUDA config)
- 16GB shared memory for PyTorch

### Documentation

**Updated:**
- `README.md` - Simplified quick start, added results table
- Reproducibility section with Docker details

**New:**
- `DOCKER_USAGE.md` - Complete Docker guide
- `CHANGELOG.md` - This file

### Git Cleanup

- Removed `data/submissions/` from history (files >100MB)
- Repository size: **12 MB** (previously several GB)
- Updated `.gitignore`

### Data

**Available:**
- Training data: In repository
- Baseline results: [HuggingFace](https://huggingface.co/datasets/vania-janet/MTRAG_taskA_results) (679 files, 10.8 GB)
- Model cache: Auto-downloaded

---

**Summary:** Repository cleanup, Docker setup for reproducibility, improved documentation.
