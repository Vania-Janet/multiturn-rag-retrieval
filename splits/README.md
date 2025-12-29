# Dataset Splits

This directory contains train/validation/test split definitions for all domains.

## 🎯 Purpose

Explicit split definitions ensure:
- **No data leakage** between train/val/test
- **Reproducibility** of all experiments
- **Transparency** for reviewers and future work

## 📋 Split Strategy by Domain

| Domain | Strategy | Rationale |
|--------|----------|-----------|
| **ClapNQ** | Conversation-level | Prevents leakage across dialogue turns |
| **FiQA** | Query-level | Independent Q&A pairs |
| **Govt** | Query-level | Independent policy queries |
| **Cloud** | Query-level + stratified | Balanced across service types |

## 📊 Split Ratios

All domains use **80/10/10** split:
- **Train (80%)**: Used for any training/tuning (fine-tuning models, etc.)
- **Val (10%)**: Used for hyperparameter selection and model selection
- **Test (10%)**: **Pseudo-test** - final evaluation only, never used for tuning

## ⚠️ Important Rules

1. **Test set is held out**: No hyperparameter tuning on test
2. **Conversation integrity**: For ClapNQ, all turns from a conversation stay in same split
3. **Temporal consistency**: If applicable, older queries go to train
4. **No query/document overlap**: Ensure independence between splits

## 🔄 Usage

Scripts automatically load splits from these files:

```python
from utils.data_loader import load_split

train_queries = load_split('clapnq', 'train')
val_queries = load_split('clapnq', 'val')
test_queries = load_split('clapnq', 'test')
```

## 📝 Format

Each YAML file contains:
```yaml
train_queries: [...]  # or train_conversations
val_queries: [...]
test_queries: [...]

split_strategy: query_level | conversation_level
split_ratio: [0.8, 0.1, 0.1]
```

## 🔗 Related

- **data/raw/{domain}/**: Original datasets
- **data/processed/{domain}/**: Processed corpora
- **experiments/**: All experiments respect these splits
