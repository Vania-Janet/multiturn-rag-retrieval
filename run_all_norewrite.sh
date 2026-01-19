#!/bin/bash
# Ejecutar experimentos sin rewrites (norewrite) para todos los dominios

set -e

export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "Experimentos Híbridos SIN REWRITES"
echo "Todos los dominios: clapnq, govt, cloud, fiqa"
echo "=========================================="
date
echo ""

DOMAINS="clapnq govt cloud fiqa"

# ============================================
# SPLADE + Voyage-3 (No Rewrite)
# ============================================
echo "1️⃣  SPLADE + Voyage-3 (sin rewrites - last turn)"
echo "=========================================="

for domain in $DOMAINS; do
    echo ""
    echo ">>> Procesando: $domain"
    echo "----------------------------------------"
    
    python -m src.pipeline.run \
        --config configs/experiments/02-hybrid/hybrid_splade_voyage_norewrite.yaml \
        --domain $domain \
        --force
    
    echo "✓ Completado: $domain"
    echo ""
done

# ============================================
# SPLADE + BGE-1.5 (No Rewrite)
# ============================================
echo "2️⃣  SPLADE + BGE-1.5 (sin rewrites - last turn)"
echo "=========================================="

for domain in $DOMAINS; do
    echo ""
    echo ">>> Procesando: $domain"
    echo "----------------------------------------"
    
    python -m src.pipeline.run \
        --config configs/experiments/02-hybrid/hybrid_splade_bge15_norewrite.yaml \
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
echo "✅ EXPERIMENTOS SIN REWRITES COMPLETADOS"
echo "=========================================="
date
echo ""

echo "📊 Métricas NDCG@10 generadas:"
echo ""
echo "SPLADE + Voyage-3 (No Rewrite):"
for domain in $DOMAINS; do
    metrics_file="experiments/02-hybrid/hybrid_splade_voyage_norewrite/$domain/metrics.json"
    if [ -f "$metrics_file" ]; then
        ndcg10=$(python3 -c "import json; m=json.load(open('$metrics_file')); idx=3 if len(m['nDCG'])>=6 else min(1,len(m['nDCG'])-1); print(f'{m[\"nDCG\"][idx]:.5f}')" 2>/dev/null || echo "N/A")
        echo "   • $domain: NDCG@10 = $ndcg10"
    fi
done

echo ""
echo "SPLADE + BGE-1.5 (No Rewrite):"
for domain in $DOMAINS; do
    metrics_file="experiments/02-hybrid/hybrid_splade_bge15_norewrite/$domain/metrics.json"
    if [ -f "$metrics_file" ]; then
        ndcg10=$(python3 -c "import json; m=json.load(open('$metrics_file')); idx=3 if len(m['nDCG'])>=6 else min(1,len(m['nDCG'])-1); print(f'{m[\"nDCG\"][idx]:.5f}')" 2>/dev/null || echo "N/A")
        echo "   • $domain: NDCG@10 = $ndcg10"
    fi
done

echo ""
