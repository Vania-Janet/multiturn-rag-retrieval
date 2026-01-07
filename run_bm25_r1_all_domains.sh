#!/bin/bash
set -e

DOMAINS=("clapnq" "cloud" "fiqa" "govt")

echo "🚀 Running BM25 R1 Condensation for all domains with keyword-preserving prompt"
echo "=================================================================="

for domain in "${DOMAINS[@]}"; do
    echo ""
    echo "📍 Running domain: $domain"
    echo "------------------------------------------------------------------"
    
    # Remove existing results to force fresh run
    rm -rf experiments/bm25_r1_condensation/$domain
    
    # Run experiment
    python scripts/run_experiment.py \
        --experiment bm25_r1_condensation \
        --domain $domain \
        --force
    
    echo "✓ Completed $domain"
done

echo ""
echo "=================================================================="
echo "✅ All domains completed!"
echo ""
echo "📊 Results summary:"
for domain in "${DOMAINS[@]}"; do
    if [ -f "experiments/bm25_r1_condensation/$domain/metrics.json" ]; then
        ndcg=$(python -c "import json; m=json.load(open('experiments/bm25_r1_condensation/$domain/metrics.json')); print(f\"{m['nDCG'][1]:.4f}\")" 2>/dev/null || echo "N/A")
        echo "  $domain: NDCG@5 = $ndcg"
    fi
done
