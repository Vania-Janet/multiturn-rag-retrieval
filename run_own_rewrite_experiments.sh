#!/bin/bash
# Ejecutar experimentos híbridos con rewrites propios (Cohere command-r)
# Optimizado para RTX 4090 (24GB VRAM)

set -e

export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "Experimentos con Rewrites Propios (Cohere)"
echo "=========================================="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader -i 0)"
echo ""
date
echo ""

# Verificar que existan los archivos de rewrite
echo "Verificando archivos de rewrite..."
for domain in clapnq cloud fiqa govt; do
    file="data/rewrite_cohere/${domain}_command-r-rewrite.txt"
    if [ ! -f "$file" ]; then
        echo "❌ ERROR: No se encuentra $file"
        exit 1
    fi
    queries=$(wc -l < "$file")
    echo "✓ $domain: $queries queries"
done
echo ""

# ============================================
# Experimento 1: SPLADE + Voyage-3 
# Para: CLAPNQ y GOVT (dominios "fuertes")
# ============================================
echo "=========================================="
echo "1️⃣  SPLADE + Voyage-3 (con rewrites propios)"
echo "=========================================="

for domain in clapnq govt; do
    echo ""
    echo ">>> Procesando: $domain"
    echo "----------------------------------------"
    
    python -m src.pipeline.run \
        --config configs/experiments/02-hybrid/hybrid_splade_voyage_rewrite_own.yaml \
        --domain $domain
    
    echo "✓ Completado: $domain"
    echo ""
done

# ============================================
# Experimento 2: SPLADE + BGE-1.5
# Para: CLOUD y FIQA (dominios "débiles")
# ============================================
echo "=========================================="
echo "2️⃣  SPLADE + BGE-1.5 (con rewrites propios)"
echo "=========================================="

for domain in cloud fiqa; do
    echo ""
    echo ">>> Procesando: $domain"
    echo "----------------------------------------"
    
    python -m src.pipeline.run \
        --config configs/experiments/02-hybrid/hybrid_splade_bge15_rewrite_own.yaml \
        --domain $domain
    
    echo "✓ Completado: $domain"
    echo ""
done

# ============================================
# Resumen Final
# ============================================
echo ""
echo "=========================================="
echo "✅ TODOS LOS EXPERIMENTOS COMPLETADOS"
echo "=========================================="
date
echo ""

echo "📊 Resultados guardados en:"
echo "   experiments/02-hybrid/hybrid_splade_voyage_rewrite_own/"
echo "   experiments/02-hybrid/hybrid_splade_bge15_rewrite_own/"
echo ""

echo "📈 Ver métricas:"
for domain in clapnq govt; do
    metrics_file="experiments/02-hybrid/hybrid_splade_voyage_rewrite_own/$domain/metrics.json"
    if [ -f "$metrics_file" ]; then
        ndcg10=$(python3 -c "import json; m=json.load(open('$metrics_file')); print(f'{m[\"nDCG\"][3]:.5f}')")
        echo "   • $domain (SPLADE+Voyage): NDCG@10 = $ndcg10"
    fi
done

for domain in cloud fiqa; do
    metrics_file="experiments/02-hybrid/hybrid_splade_bge15_rewrite_own/$domain/metrics.json"
    if [ -f "$metrics_file" ]; then
        ndcg10=$(python3 -c "import json; m=json.load(open('$metrics_file')); print(f'{m[\"nDCG\"][3]:.5f}')")
        echo "   • $domain (SPLADE+BGE15): NDCG@10 = $ndcg10"
    fi
done

echo ""
echo "🔍 Comparar con baselines:"
echo "   Baselines (ground truth rewrites):"
echo "   • CLAPNQ: NDCG@10 = 0.56266"
echo "   • GOVT:   NDCG@10 = 0.53445"
echo "   • CLOUD:  NDCG@10 = 0.44028"
echo "   • FIQA:   NDCG@10 = 0.40589"
echo ""
