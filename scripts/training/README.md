# Fine-tuning scripts

Scripts for domain-specific model fine-tuning.

## Reranker Fine-tuning

Train cross-encoder reranker on domain query-document pairs:

```bash
python scripts/training/finetune_reranker.py \
  --domain clapnq \
  --base_model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --output_dir models/finetuned/reranker/clapnq \
  --epochs 3 \
  --batch_size 16
```

## SPLADE Fine-tuning

Train SPLADE model on domain-specific corpus:

```bash
python scripts/training/finetune_splade.py \
  --domain fiqa \
  --base_model naver/splade-cocondenser-ensembledistil \
  --corpus data/processed/fiqa/corpus.jsonl \
  --output_dir models/finetuned/splade/fiqa \
  --epochs 5 \
  --batch_size 32
```

## Training Data Preparation

Generate training pairs from queries and relevance judgments:

```bash
python scripts/training/prepare_training_data.py \
  --domain clapnq \
  --output data/training/clapnq
```
