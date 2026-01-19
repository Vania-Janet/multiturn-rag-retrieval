#!/bin/bash
# Re-ejecutar experimentos baseline con k_values completo [1,3,5,10,20,100]

set -e

export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "Re-ejecutando Baselines Híbridos"
echo "Con k_values = [1, 3, 5, 10, 20, 100]"
echo "=========================================="
date
echo ""

# ============================================
# SPLADE + Voyage-3 (ground truth rewrites)
# ============================================
echo "1️⃣  SPLADE + Voyage-3 (ground truth rewrites)"
echo "=========================================="

for domain in clapnq govt; do
    echo ""
    echo ">>> Procesando: $domain"
    echo "----------------------------------------"
    
    python -m src.pipeline.run \
        --config configs/experiments/02-hybrid/hybrid_splade_voyage_rewrite.yaml \
        --domain $domain \
        --force
    
    echo "✓ Completado: $domain"
    echo ""
done

# ============================================
# SPLADE + BGE-1.5 (ground truth rewrites)
# ============================================
echo "2️⃣  SPLADE + BGE-1.5 (ground truth rewrites)"
echo "=========================================="

for domain in cloud fiqa; do
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
# Resumen Final
# ============================================
echo ""
echo "=========================================="
echo "✅ BASELINES ACTUALIZADOS"
echo "=========================================="
date
echo ""

echo "📊 Nuevas métricas (k=[1,3,5,10,20,100]):"
for domain in clapnq govt; do
    metrics_file="experiments/02-hybrid/hybrid_splade_voyage_rewrite/$domain/metrics.json"
    if [ -f "$metrics_file" ]; then
        ndcg10=$(python3 -c "import json; m=json.load(open('$metrics_file')); print(f'{m[\"nDCG\"][3]:.5f}')")
        echo "   • $domain (SPLADE+Voyage GT): NDCG@10 = $ndcg10"
    fi
done

for domain in cloud fiqa; do
    metrics_file="experiments/02-hybrid/hybrid_splade_bge15_rewrite/$domain/metrics.json"
    if [ -f "$metrics_file" ]; then
        ndcg10=$(python3 -c "import json; m=json.load(open('$metrics_file')); print(f'{m[\"nDCG\"][3]:.5f}')")
        echo "   • $domain (SPLADE+BGE15 GT): NDCG@10 = $ndcg10"
    fi
done

echo ""
