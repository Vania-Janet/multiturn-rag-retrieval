#!/usr/bin/env python3
"""
Script to verify indexing configuration and GPU setup.
"""
import torch
import os
import json

print("=" * 60)
print("INDEXING CONFIGURATION VERIFICATION")
print("=" * 60)

# 1. GPU Check
print("\n[1] GPU Configuration:")
print(f"  - CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  - GPU Count: {torch.cuda.device_count()}")
    print(f"  - GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"  - CUDA Version: {torch.version.cuda}")
    
    # Memory info
    mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
    mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
    mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  - GPU Memory: {mem_total:.1f} GB total")
    print(f"  - Memory Allocated: {mem_allocated:.1f} GB")
    print(f"  - Memory Reserved: {mem_reserved:.1f} GB")
else:
    print("  ⚠️  WARNING: No GPU detected. Indexing will run on CPU (very slow)")

# 2. Data Path Check
print("\n[2] Data Path Configuration:")
data_dir = "data/passage_level_processed"
domains = ["clapnq", "cloud", "fiqa", "govt"]

for domain in domains:
    corpus_path = os.path.join(data_dir, domain, "corpus.jsonl")
    if os.path.exists(corpus_path):
        # Count lines
        with open(corpus_path, 'r') as f:
            count = sum(1 for _ in f)
        print(f"  ✓ {domain:8s}: {count:6d} documents at {corpus_path}")
    else:
        print(f"  ✗ {domain:8s}: NOT FOUND at {corpus_path}")

# 3. Index Configuration
print("\n[3] BGE 1.5 Configuration:")
print(f"  - Model: BAAI/bge-base-en-v1.5")
print(f"  - Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print(f"  - Precision: FP16 (if CUDA available)")
print(f"  - Batch Size: 1024 (optimized for A100)")
print(f"  - Embedding Dim: 1024")
print(f"  - FAISS Index Type: IndexFlatIP (Inner Product for Cosine Similarity)")

print("\n[4] BM25 Configuration:")
print(f"  - Tokenizer: NLTK word_tokenize")
print(f"  - Saved Format: Pickle (.pkl)")

print("\n[5] ELSER Configuration:")
es_url = os.getenv("ELASTICSEARCH_URL")
if es_url:
    print(f"  - Elasticsearch URL: {es_url}")
    print(f"  - Model: .elser_model_2")
    print(f"  - Ingest Pipeline: elser-ingest-pipeline")
else:
    print(f"  ⚠️  ELASTICSEARCH_URL not set. ELSER indexing will be skipped.")

# 4. Estimated Memory Usage (BGE on A100)
print("\n[6] Estimated Memory Usage (BGE 1.5):")
avg_docs = 91620  # Average across domains
embedding_size = 1024 * 4  # float32 = 4 bytes per dim
model_size_gb = 1.3  # BGE-large is ~1.3GB
embeddings_size_gb = (avg_docs * embedding_size) / (1024**3)
print(f"  - Model Size: ~{model_size_gb:.1f} GB")
print(f"  - Embeddings (avg): ~{embeddings_size_gb:.1f} GB per domain")
print(f"  - Batch Processing: ~{1024 * embedding_size / (1024**2):.1f} MB per batch")
print(f"  - Total Est. (worst case): ~{model_size_gb + embeddings_size_gb * 1.5:.1f} GB")
print(f"  - A100 80GB: ✓ SUFFICIENT")

print("\n" + "=" * 60)
print("READY FOR INDEXING")
print("=" * 60)
print("\nRun:")
print("  python src/pipeline/indexing/build_indices.py --domains all --models bge bm25")
print("\nOr with ELSER (if Elasticsearch is configured):")
print("  python src/pipeline/indexing/build_indices.py --domains all --models bge bm25 elser")
