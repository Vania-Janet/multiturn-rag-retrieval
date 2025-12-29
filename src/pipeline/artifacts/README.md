# Artifacts Directory

Large files and outputs not suitable for git versioning.

## 📁 Structure

```
artifacts/
├── models/           # Fine-tuned models, checkpoints
├── embeddings/       # Cached embeddings
└── logs/             # Detailed execution logs
```

## 🎯 Purpose

This directory stores large artifacts that:
- Are too large for git (>100MB)
- Can be regenerated from code + data
- Should be stored in cloud storage or HuggingFace Hub

## 📦 Contents

### `models/`
Fine-tuned model checkpoints:
- Domain-adapted rerankers (A10)
- Fine-tuned SPLADE models (A11)
- Intermediate training checkpoints

**Storage:** Upload to HuggingFace Hub for sharing

### `embeddings/`
Pre-computed embeddings cache:
- Query embeddings (by model)
- Document embeddings (by model)
- Cached for faster experimentation

**Note:** These can be regenerated but take time

### `logs/`
Detailed execution logs:
- Training logs with metrics per epoch
- Indexing logs with progress
- Full experiment traces

## 🚫 Git Policy

**DO NOT commit to git:**
- Model checkpoints (.pt, .pth, .bin, .safetensors)
- Embedding caches (.npy, .pkl, .h5)
- Large log files (>10MB)

**DO commit:**
- Model configurations (.json, .yaml)
- Training scripts
- README files

## ☁️ Cloud Storage

For collaboration, upload large artifacts to:
- **HuggingFace Hub**: Models and embeddings
- **Weights & Biases**: Training logs and metrics
- **Git LFS** (optional): For critical reproducibility files

## 📝 Manifest Files

Each subdirectory should contain a `manifest.json`:
```json
{
  "created_at": "2025-01-15T10:30:00Z",
  "model_name": "domain-reranker-clapnq",
  "base_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "training_config": "configs/training/reranker.yaml",
  "metrics": {
    "ndcg@10": 0.645
  },
  "size_mb": 127,
  "huggingface_url": "https://huggingface.co/user/model"
}
```

## 🔗 Related

- `.gitignore`: Exclusion rules
- `models/`: Lightweight model configs (git-tracked)
- `experiments/`: Experiment results (git-tracked)
