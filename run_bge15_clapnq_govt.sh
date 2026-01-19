#!/bin/bash
# Ejecutar experimentos BGE-1.5 para clapnq y govt

set -e

export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "Experimentos SPLADE + BGE-1.5"
echo "Dominios: CLAPNQ y GOVT"
echo "=========================================="
date
echo ""

# ============================================
# Ground Truth Rewrites
# ============================================
echo "1️⃣  SPLADE + BGE-1.5 (ground truth rewrites)"
echo "=========================================="

for domain in clapnq govt; do
    echo ""
    echo ">>> Procesando: $domain"
    echo "----------------------------------------"
    
    python -m src.pipeline.run \
        --config configs/experiments/02-hybrid/hybrid_splade_bge15_rewrite.yaml \
        --domain $domain \
        --force
    
    echo "✓ Completado: $domain"
    echo ""
done

# ============================================
# Cohere Rewrites
# ============================================
echo "2️⃣  SPLADE + BGE-1.5 (Cohere rewrites propios)"
echo "=========================================="

for domain in clapnq govt; do
    echo ""
    echo ">>> Procesando: $domain"
    echo "----------------------------------------"
    
    python -m src.pipeline.run \
        --config configs/experiments/02-hybrid/hybrid_splade_bge15_rewrite_own.yaml \
        --domain $domain \
        --force
    
    echo "✓ Completado: $domain"
    echo ""
done

# ============================================
# Resumen
# ============================================
echo ""
echo "=========================================="
echo "✅ EXPERIMENTOS COMPLETADOS"
echo "=========================================="
date
echo ""

echo "📊 Métricas generadas:"
for domain in clapnq govt; do
    metrics_file="experiments/02-hybrid/hybrid_splade_bge15_rewrite/$domain/metrics.json"
    if [ -f "$metrics_file" ]; then
        ndcg10=$(python3 -c "import json; m=json.load(open('$metrics_file')); print(f'{m[\"nDCG\"][3]:.5f}')" 2>/dev/null || echo "N/A")
        echo "   • $domain (GT rewrite):     NDCG@10 = $ndcg10"
    fi
    
    metrics_file="experiments/02-hybrid/hybrid_splade_bge15_rewrite_own/$domain/metrics.json"
    if [ -f "$metrics_file" ]; then
        ndcg10=$(python3 -c "import json; m=json.load(open('$metrics_file')); print(f'{m[\"nDCG\"][3]:.5f}')" 2>/dev/null || echo "N/A")
        echo "   • $domain (Cohere rewrite): NDCG@10 = $ndcg10"
    fi
done

echo ""
