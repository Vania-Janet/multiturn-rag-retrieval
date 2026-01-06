#!/bin/bash
set -e

# Domains
DOMAINS=("clapnq" "cloud" "fiqa" "govt")

# Experiments (excluding Voyage)
EXPERIMENTS=(
    "replication_bm25"
    "replication_splade"
    "replication_bge15"
    "replication_bgem3"
    "A0_baseline_bm25_fullhist"
    "A0_baseline_splade_fullhist"
    "A1_baseline_bgem3_fullhist"
)

OUTPUT_DIR="experiments/0-baselines"

mkdir -p "$OUTPUT_DIR"

echo "Running baselines into $OUTPUT_DIR..."

for exp in "${EXPERIMENTS[@]}"; do
    echo "=================================================="
    echo "Experiment: $exp"
    echo "=================================================="
    
    for domain in "${DOMAINS[@]}"; do
        echo "  Domain: $domain"
        # Using --force to overwrite if needed, or maybe I should check if it exists?
        # User said "corre", implies execute.
        python3 scripts/run_experiment.py \
            --experiment "$exp" \
            --domain "$domain" \
            --output-dir "$OUTPUT_DIR" \
            --force
    done
done

echo "All requested baselines finished."
