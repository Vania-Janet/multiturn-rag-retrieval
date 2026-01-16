# Fine-tuning Experiments

This directory contains configurations for experiments using the fine-tuned BGE reranker model.

## Model Details

**Model**: `pedrovo9/bge-reranker-v2-m3-multirag-finetuned`

Fine-tuned version of BAAI/bge-reranker-v2-m3 for multi-domain conversational RAG.

### Training Details

- **Base model**: BAAI/bge-reranker-v2-m3
- **Training strategy**: Pairwise learning (1:2 ratio positive:negative)
- **Hard negatives**: BM25-retrieved documents
- **Epochs**: 3
- **Domains**: ClapNQ, Cloud, FiQA, Govt

### Data Splits

The model repository includes proper train/test/val splits to prevent data leakage:
- `data/train.jsonl` - Training data
- `data/test.jsonl` - Test data
- `data/val.jsonl` - Validation data

These splits are conversation-level splits ensuring no information leakage across splits.

## Experiments

### A10_finetuned_reranker

Hybrid retrieval (SPLADE + BGE-M3) with fine-tuned reranker:
- Query rewriting with vLLM (Qwen2.5-7B)
- Hybrid retrieval (RRF fusion)
- Fine-tuned BGE reranking

```bash
python scripts/run_experiment.py -e A10_finetuned_reranker -d all
```

### finetune_bge_splade_bge15_rewrite

Hybrid retrieval (SPLADE + BGE-1.5) with fine-tuned reranker:
- Query rewriting with vLLM
- SPLADE (sparse) + BGE-base-en-v1.5 (dense)
- Fine-tuned BGE reranking

```bash
python scripts/run_experiment.py -e finetune_bge_splade_bge15_rewrite -d all
```

### finetune_bge_splade_voyage_rewrite

Hybrid retrieval (SPLADE + Voyage-3) with fine-tuned reranker:
- Query rewriting with vLLM
- SPLADE (sparse) + Voyage-3-large (dense)
- Fine-tuned BGE reranking

```bash
python scripts/run_experiment.py -e finetune_bge_splade_voyage_rewrite -d all
```

## Usage

### Load Model Directly

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("pedrovo9/bge-reranker-v2-m3-multirag-finetuned")
model = AutoModelForSequenceClassification.from_pretrained("pedrovo9/bge-reranker-v2-m3-multirag-finetuned")
```

### Use in Pipeline

The model is automatically loaded when you run experiments with `reranker_type: finetuned_bge`:

```yaml
reranking:
  enabled: true
  reranker_type: "finetuned_bge"
  model_name: "pedrovo9/bge-reranker-v2-m3-multirag-finetuned"
  top_k: 100
  batch_size: 32
  use_fp16: true
```

## Expected Results

The fine-tuned model should show improvements over the base BGE reranker, especially on:
- Multi-turn conversational queries
- Domain-specific terminology
- Hard negative examples that appear relevant but aren't

Baseline comparisons:
- **SPLADE baseline**: nDCG@10 ~0.44
- **Hybrid (SPLADE + Voyage)**: nDCG@10 ~0.48-0.52
- **+ Fine-tuned reranking**: Expected nDCG@10 ~0.50-0.55

## Run All Fine-tuning Experiments

```bash
# Run all fine-tuning experiments across all domains
./run_finetuned_experiments.sh
```

Or run individually:

```bash
# A10 (Hybrid + Fine-tuned Reranker)
python scripts/run_experiment.py -e A10_finetuned_reranker -d clapnq
python scripts/run_experiment.py -e A10_finetuned_reranker -d cloud
python scripts/run_experiment.py -e A10_finetuned_reranker -d fiqa
python scripts/run_experiment.py -e A10_finetuned_reranker -d govt
```

## Notes

- The model was trained on the same domains as the test set, but with proper train/test splits
- Data leakage prevention: conversation-level splits ensure no dialogue from training appears in test
- The model repository on Hugging Face contains the complete training data splits for transparency
- All training was done with hard negatives from BM25 to improve discrimination capability

### Step 2: Update Experiment Configs

Once training completes, Cohere will provide a model ID like: `abc123-ft-rerank-v4-pro`

Update the configs:
```bash
# Replace REPLACE_WITH_FINETUNED_MODEL_ID with your actual model ID
sed -i 's/REPLACE_WITH_FINETUNED_MODEL_ID/abc123-ft-rerank-v4-pro/g' \
  configs/experiments/05-finetune/finetune_cohere_*.yaml
```

### Step 3: Run Experiments

```bash
# Run fine-tuned experiments
cd /workspace/mt-rag-benchmark/task_a_retrieval

# CLAPNQ + GOVT (with Voyage embeddings)
python -m pipeline.run \
  --config configs/experiments/05-finetune/finetune_cohere_splade_voyage_rewrite.yaml \
  --domain clapnq

python -m pipeline.run \
  --config configs/experiments/05-finetune/finetune_cohere_splade_voyage_rewrite.yaml \
  --domain govt

# CLOUD + FIQA (with BGE embeddings)
python -m pipeline.run \
  --config configs/experiments/05-finetune/finetune_cohere_splade_bge15_rewrite.yaml \
  --domain cloud

python -m pipeline.run \
  --config configs/experiments/05-finetune/finetune_cohere_splade_bge15_rewrite.yaml \
  --domain fiqa
```

## Expected Improvements

**Baseline (no reranking)**:
- Hybrid SPLADE + Voyage/BGE with RRF: 0.486 nDCG@10

**Base rerank-v4-pro (without fine-tuning)**:
- Tested with v3.5, failed with -20.9% (0.384 nDCG@10)
- Reason: Generic semantic similarity doesn't capture multi-turn context

**Fine-tuned rerank-v4-pro (expected)**:
- Target: +5-15% improvement over baseline (0.51-0.56 nDCG@10)
- Trained on exact task structure (multi-turn conversations)
- Hard negatives teach discrimination of "similar but wrong" docs
- Should understand conversation context better

## Files Generated

```
experiments/05-finetune/
├── cohere_rerank_data/
│   ├── train.jsonl                    # 620 training examples
│   └── validation.jsonl               # 157 validation examples
├── finetune_cohere_splade_voyage_rewrite/
│   ├── clapnq/
│   │   ├── retrieval_results.jsonl
│   │   └── metrics.json
│   └── govt/
│       ├── retrieval_results.jsonl
│       └── metrics.json
└── finetune_cohere_splade_bge15_rewrite/
    ├── cloud/
    │   ├── retrieval_results.jsonl
    │   └── metrics.json
    └── fiqa/
        ├── retrieval_results.jsonl
        └── metrics.json
```

## Troubleshooting

**If fine-tuned model still underperforms:**
1. Check if training converged (view training metrics in Cohere dashboard)
2. Try increasing training epochs
3. Consider using more hard negatives (currently ~5, can go up to 10)
4. Experiment with learning rate hyperparameters
5. Ensure no data leakage between train/validation splits

**Alternative approaches if fine-tuning doesn't help:**
- Stick with hybrid baseline (0.486 is already excellent)
- Try query expansion instead of reranking
- Experiment with different RRF k values
- Use ColBERT for late interaction reranking
