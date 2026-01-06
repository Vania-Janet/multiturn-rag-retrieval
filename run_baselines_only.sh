#!/bin/bash
# Ejecutar SOLO baselines (sin query experiments problemáticos)

DOMAINS="clapnq cloud fiqa govt"
BASELINES="A0_baseline_bm25_fullhist A0_baseline_splade_fullhist A1_baseline_bgem3_fullhist replication_bge15 replication_bgem3 replication_bm25 replication_splade"

LOG_FILE="logs/baselines_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "EJECUTANDO SOLO BASELINES"
echo "=========================================="
echo "Inicio: $(date)"
echo "Log: $LOG_FILE"
echo ""

for exp in $BASELINES; do
    echo "=========================================="
    echo "Experimento: $exp"
    echo "=========================================="
    for domain in $DOMAINS; do
        echo "  [$(date '+%H:%M:%S')] → Dominio: $domain"
        python3 scripts/run_experiment.py -e $exp -d $domain --force 2>&1 | tee -a "$LOG_FILE"
        echo ""
    done
done

echo "=========================================="
echo "COMPLETADO"
echo "Fin: $(date)"
echo "=========================================="
