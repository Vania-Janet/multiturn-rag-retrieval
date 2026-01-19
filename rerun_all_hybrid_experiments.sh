#!/bin/bash
# RE-EJECUTAR TODOS LOS EXPERIMENTOS HÍBRIDOS CON EL BUG FIX
# Bug fix: Remover truncamiento hardcoded a 10 documentos

set -e

# Activar entorno virtual
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "🔧 RE-EJECUCIÓN COMPLETA - BUG FIX APLICADO"
echo "=========================================="
echo "Bug: Truncamiento hardcoded contexts[:10]"
echo "Fix: Truncamiento configurable via output.top_k"
echo ""
date
echo ""

TOTAL_START=$(date +%s)

# ============================================
# 1. Sin Rewrites - Todos los dominios
# ============================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️⃣  EXPERIMENTOS SIN REWRITES (4 dominios)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

DOMAINS="clapnq govt cloud fiqa"

echo "🔹 SPLADE + Voyage-3 (No Rewrite)"
for domain in $DOMAINS; do
    echo "  → $domain"
    python -m src.pipeline.run \
        --config configs/experiments/02-hybrid/hybrid_splade_voyage_norewrite.yaml \
        --domain $domain \
        --force
    echo "    ✓"
done
echo ""

echo "🔹 SPLADE + BGE-1.5 (No Rewrite)"
for domain in $DOMAINS; do
    echo "  → $domain"
    python -m src.pipeline.run \
        --config configs/experiments/02-hybrid/hybrid_splade_bge15_norewrite.yaml \
        --domain $domain \
        --force
    echo "    ✓"
done
echo ""

# ============================================
# 2. Ground Truth Rewrites - Todos los dominios
# ============================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2️⃣  GROUND TRUTH REWRITES (4 dominios)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "🔹 SPLADE + Voyage-3 (GT Rewrite)"
for domain in $DOMAINS; do
    echo "  → $domain"
    python -m src.pipeline.run \
        --config configs/experiments/02-hybrid/hybrid_splade_voyage_rewrite.yaml \
        --domain $domain \
        --force
    echo "    ✓"
done
echo ""

echo "🔹 SPLADE + BGE-1.5 (GT Rewrite)"
for domain in $DOMAINS; do
    echo "  → $domain"
    python -m src.pipeline.run \
        --config configs/experiments/02-hybrid/hybrid_splade_bge15_rewrite.yaml \
        --domain $domain \
        --force
    echo "    ✓"
done
echo ""

# ============================================
# 3. Cohere Rewrites - Dominios específicos
# ============================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3️⃣  COHERE REWRITES (dominios específicos)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "🔹 SPLADE + Voyage-3 (Cohere) - cloud, fiqa"
for domain in cloud fiqa; do
    echo "  → $domain"
    python -m src.pipeline.run \
        --config configs/experiments/02-hybrid/hybrid_splade_voyage_rewrite_own.yaml \
        --domain $domain \
        --force
    echo "    ✓"
done
echo ""

echo "🔹 SPLADE + BGE-1.5 (Cohere) - clapnq, govt"
for domain in clapnq govt; do
    echo "  → $domain"
    python -m src.pipeline.run \
        --config configs/experiments/02-hybrid/hybrid_splade_bge15_rewrite_own.yaml \
        --domain $domain \
        --force
    echo "    ✓"
done
echo ""

# ============================================
# Resumen Final
# ============================================
TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))
TOTAL_MINUTES=$((TOTAL_DURATION / 60))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ TODOS LOS EXPERIMENTOS COMPLETADOS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
date
echo "Tiempo total: ${TOTAL_MINUTES} minutos"
echo ""

echo "📊 RESUMEN DE MÉTRICAS NDCG@10:"
echo ""

# Función para extraer NDCG@10
get_ndcg10() {
    local metrics_file=$1
    if [ -f "$metrics_file" ]; then
        python3 -c "import json; m=json.load(open('$metrics_file')); print(f'{m[\"nDCG\"][3]:.5f}')" 2>/dev/null || echo "ERROR"
    else
        echo "N/A"
    fi
}

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  VOYAGE-3 CONFIGURATIONS                                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
for domain in $DOMAINS; do
    no_rew=$(get_ndcg10 "experiments/02-hybrid/hybrid_splade_voyage_norewrite/$domain/metrics.json")
    gt_rew=$(get_ndcg10 "experiments/02-hybrid/hybrid_splade_voyage_rewrite/$domain/metrics.json")
    own_rew="N/A"
    if [ "$domain" == "cloud" ] || [ "$domain" == "fiqa" ]; then
        own_rew=$(get_ndcg10 "experiments/02-hybrid/hybrid_splade_voyage_rewrite_own/$domain/metrics.json")
    fi
    printf "  %-10s | No Rewrite: %s | GT Rewrite: %s | Cohere: %s\n" "$domain" "$no_rew" "$gt_rew" "$own_rew"
done

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  BGE-1.5 CONFIGURATIONS                                    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
for domain in $DOMAINS; do
    no_rew=$(get_ndcg10 "experiments/02-hybrid/hybrid_splade_bge15_norewrite/$domain/metrics.json")
    gt_rew=$(get_ndcg10 "experiments/02-hybrid/hybrid_splade_bge15_rewrite/$domain/metrics.json")
    own_rew="N/A"
    if [ "$domain" == "clapnq" ] || [ "$domain" == "govt" ]; then
        own_rew=$(get_ndcg10 "experiments/02-hybrid/hybrid_splade_bge15_rewrite_own/$domain/metrics.json")
    fi
    printf "  %-10s | No Rewrite: %s | GT Rewrite: %s | Cohere: %s\n" "$domain" "$no_rew" "$gt_rew" "$own_rew"
done

echo ""
echo "💾 Resultados guardados en: experiments/02-hybrid/"
echo ""
