# Submissions Directory

This directory contains formatted submission files for the MT-RAG-Benchmark competition.

## Structure

```
submissions/
├── val/                    # Validation set submissions (for testing)
│   ├── A6_hybrid_rerank.jsonl
│   └── A10_finetuned.jsonl
└── test/                   # Test set submissions (for final evaluation)
    └── final_submission.jsonl
```

## File Format

Each submission file is in JSONL format with one entry per query:

```json
{
  "query_id": "clapnq_123",
  "retrieved_passages": [
    "doc1_chunk0",
    "doc2_chunk3",
    "doc5_chunk1",
    ...
  ],
  "top_k": 10
}
```

## Generating Submissions

Use the `make_submission.py` script:

```bash
# Generate validation submission
python scripts/make_submission.py \
  --experiment A6_hybrid_rerank \
  --split val \
  --output data/submissions/val/A6_hybrid_rerank.jsonl

# Generate test submission (final)
python scripts/make_submission.py \
  --experiment A6_hybrid_rerank \
  --split test \
  --output data/submissions/test/final_submission.jsonl
```

## Validating Submissions

Always validate before uploading:

```bash
python scripts/validate_submission.py \
  --submission data/submissions/test/final_submission.jsonl \
  --strict
```

## Submission Checklist

Before submitting to the competition:

- [ ] Run validation script with `--strict` flag
- [ ] Verify all domains covered (clapnq, cloud, fiqa, govt)
- [ ] Check file size is reasonable (< 10MB)
- [ ] Ensure exactly 10 passages per query
- [ ] No duplicate query_ids
- [ ] File is valid JSONL (one JSON per line)

## Naming Convention

Use descriptive names that include:
- Experiment ID (e.g., A6)
- Key technique (e.g., hybrid_rerank)
- Split (val or test)

Example: `A6_hybrid_rerank_val.jsonl`

## Git Policy

⚠️ **Never commit test set submissions to git!**

Only validation submissions can be tracked for experimentation.
Add test submissions to `.gitignore`.
