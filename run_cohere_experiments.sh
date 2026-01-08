#!/bin/bash
# Run Cohere reranking experiments across all domains
# This replaces BGE with Cohere which should perform better on multi-turn conversations

cd /workspace/mt-rag-benchmark/task_a_retrieval

# Set Python path for relative imports
export PYTHONPATH=/workspace/mt-rag-benchmark/task_a_retrieval/src:$PYTHONPATH

# GPU 0: CLAPNQ + GOVT (with Voyage embeddings)
CUDA_VISIBLE_DEVICES=0 python -m pipeline.run \
  --config configs/experiments/03-rerank/rerank_cohere_splade_voyage_rewrite.yaml \
  --domain clapnq \
  > /tmp/cohere_clapnq.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python -m pipeline.run \
  --config configs/experiments/03-rerank/rerank_cohere_splade_voyage_rewrite.yaml \
  --domain govt \
  > /tmp/cohere_govt.log 2>&1 &

# GPU 1: CLOUD + FIQA (with BGE embeddings)
CUDA_VISIBLE_DEVICES=1 python -m pipeline.run \
  --config configs/experiments/03-rerank/rerank_cohere_splade_bge15_rewrite.yaml \
  --domain cloud \
  > /tmp/cohere_cloud.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -m pipeline.run \
  --config configs/experiments/03-rerank/rerank_cohere_splade_bge15_rewrite.yaml \
  --domain fiqa \
  > /tmp/cohere_fiqa.log 2>&1 &

echo "✓ 4 experimentos Cohere iniciados"
echo "  GPU 0: CLAPNQ, GOVT (logs: /tmp/cohere_clapnq.log, /tmp/cohere_govt.log)"
echo "  GPU 1: CLOUD, FIQA (logs: /tmp/cohere_cloud.log, /tmp/cohere_fiqa.log)"
echo ""
echo "Monitorear progreso:"
echo "  watch -n 10 'tail -5 /tmp/cohere_*.log'"
