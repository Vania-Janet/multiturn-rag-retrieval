# Chunking Strategies

Document chunking approaches for corpus preprocessing.

## Available strategies:

### 1. Sliding Window ([sliding_window.py](sliding_window.py))
- Fixed-size chunks with overlap
- Best for: Dense retrieval, uniform document structure
- Parameters: chunk_size, overlap

### 2. Hierarchical ([hierarchical.py](hierarchical.py))
- Multi-level nested chunks
- Best for: Long documents with clear structure
- Preserves document hierarchy

## Choosing a strategy:

| Domain | Recommended | Rationale |
|--------|-------------|-----------|
| ClapNQ | Sliding window | Code/error logs are sequential |
| FiQA | Hierarchical | Financial docs have clear sections |
| Govt | Hierarchical | Policy documents are structured |
| Cloud | Sliding window | Technical docs vary in structure |

## Usage patterns:

```bash
# Sliding window (default)
python preprocessing/chunking/sliding_window.py \
  --input data/raw/{domain}/corpus.jsonl \
  --output data/processed/{domain}/corpus.jsonl \
  --chunk_size 512 \
  --overlap 128

# Hierarchical (structured docs)
python preprocessing/chunking/hierarchical.py \
  --input data/raw/{domain}/corpus.jsonl \
  --output data/processed/{domain}/corpus.jsonl \
  --strategy semantic
```

## Output format:

All strategies produce JSONL with:
```json
{
  "id": "doc_123_chunk_0",
  "text": "chunk content...",
  "metadata": {
    "source_doc_id": "doc_123",
    "chunk_index": 0,
    "strategy": "sliding_window"
  }
}
```
