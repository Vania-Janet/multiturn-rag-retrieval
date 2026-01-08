#!/bin/bash
# Run Cohere reranking experiments SEQUENTIALLY (Trial key has 40 req/min limit)
# Running in parallel causes rate limit errors

cd /workspace/mt-rag-benchmark/task_a_retrieval
export PYTHONPATH=/workspace/mt-rag-benchmark/task_a_retrieval/src:$PYTHONPATH

echo "⚠️  Cohere Trial Key: 40 API calls/min limit"
echo "→ Ejecutando experimentos secuencialmente para evitar rate limit"
echo ""

# Run on GPU 0 one at a time
echo "[1/4] CLAPNQ (GPU 0)..."
CUDA_VISIBLE_DEVICES=0 python -m pipeline.run \
  --config configs/experiments/03-rerank/rerank_cohere_splade_voyage_rewrite.yaml \
  --domain clapnq \
  2>&1 | tee /tmp/cohere_clapnq.log

echo "[2/4] GOVT (GPU 0)..."
CUDA_VISIBLE_DEVICES=0 python -m pipeline.run \
  --config configs/experiments/03-rerank/rerank_cohere_splade_voyage_rewrite.yaml \
  --domain govt \
  2>&1 | tee /tmp/cohere_govt.log

echo "[3/4] CLOUD (GPU 1)..."
CUDA_VISIBLE_DEVICES=1 python -m pipeline.run \
  --config configs/experiments/03-rerank/rerank_cohere_splade_bge15_rewrite.yaml \
  --domain cloud \
  2>&1 | tee /tmp/cohere_cloud.log

echo "[4/4] FIQA (GPU 1)..."
CUDA_VISIBLE_DEVICES=1 python -m pipeline.run \
  --config configs/experiments/03-rerank/rerank_cohere_splade_bge15_rewrite.yaml \
  --domain fiqa \
  2>&1 | tee /tmp/cohere_fiqa.log

echo "✅ 4 experimentos Cohere completados"
