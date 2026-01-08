# Cohere Fine-tuning Experiments

This directory contains configurations and training data for fine-tuning Cohere's rerank model on the multi-turn RAG dataset.

## Training Data

**Location**: `experiments/05-finetune/cohere_rerank_data/`

- `train.jsonl`: 620 training examples (80% split, stratified by domain)
- `validation.jsonl`: 157 validation examples (20% split, stratified by domain)

**Data Statistics**:
- Total queries: 777 (from 842 tasks with ground truth)
- Avg relevant passages per query: 2.73
- Avg hard negatives per query: 4.99
- Domains: CLAPNQ (208), GOVT (201), CLOUD (188), FIQA (180)

**Triplet Structure**:
```json
{
  "query": "Rewritten multi-turn query",
  "relevant_passages": ["Ground truth relevant passage 1", "..."],
  "hard_negatives": ["Top-10 doc that looks relevant but isn't", "..."]
}
```

**Hard Negatives Strategy**:
- Extracted from hybrid baseline (SPLADE + Voyage/BGE with RRF)
- Top-10 retrieved documents that are NOT ground truth
- Teaches model to distinguish between "seems relevant" vs "actually relevant"

## How to Fine-tune

### Step 1: Upload Training Data to Cohere

```bash
# Install Cohere SDK
pip install cohere

# Upload training data
python scripts/upload_cohere_finetune.py
```

Or manually via Cohere Dashboard:
1. Go to https://dashboard.cohere.com/fine-tuning
2. Click "Create Fine-tuned Model"
3. Upload `train.jsonl` and `validation.jsonl`
4. Select base model: `rerank-v4-pro`
5. Start training (takes ~2-4 hours)

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
