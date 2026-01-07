#!/bin/bash
set -e

EXPERIMENTS=(
    "bm25_r1_condensation"
    "bm25_r2_multi"
    "splade_r1_condensation"
    "splade_r3_hyde"
    "bgem3_r1_condensation"
    "bgem3_r2_multi"
)

DOMAINS=("clapnq" "cloud" "fiqa" "govt")

echo "🚀 Running ALL Query Transformation Experiments"
echo "=================================================================="
echo "Experiments: ${EXPERIMENTS[@]}"
echo "Domains: ${DOMAINS[@]}"
echo "Total runs: $((${#EXPERIMENTS[@]} * ${#DOMAINS[@]}))"
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    for domain in "${DOMAINS[@]}"; do
        echo ""
        echo "📍 Running: $exp on $domain"
        echo "------------------------------------------------------------------"
        
        # Remove existing results to force fresh run
        rm -rf "experiments/$exp/$domain"
        
        # Clear cache for clean run
        rm -rf .cache/rewrites
        
        # Run experiment
        python scripts/run_experiment.py \
            --experiment $exp \
            --domain $domain \
            --force
        
        echo "✓ Completed $exp/$domain"
        
        # Small delay to avoid GPU conflicts
        sleep 2
    done
done

echo ""
echo "=================================================================="
echo "✅ All experiments completed!"
