#!/bin/bash
# Script para ejecutar todos los experimentos faltantes

DOMAINS="clapnq cloud fiqa govt"

echo "=========================================="
echo "EJECUTANDO EXPERIMENTOS FALTANTES"
echo "=========================================="
echo "Inicio: $(date)"
echo ""

# Baselines faltantes
BASELINES="A0_baseline_splade_fullhist replication_bm25 replication_splade replication_bgem3 replication_bge15"

for exp in $BASELINES; do
    echo "----------------------------------------"
    echo "Experimento: $exp"
    echo "----------------------------------------"
    for domain in $DOMAINS; do
        echo "  → Dominio: $domain"
        python3 scripts/run_experiment.py -e $exp -d $domain --force 2>&1 | tee -a logs/batch_run_$(date +%Y%m%d).log
    done
    echo ""
done

# Query experiments faltantes
QUERY_EXPS="bm25_r1_condensation bm25_r2_multi bgem3_r1_condensation bgem3_r2_multi splade_r1_condensation splade_r3_hyde"

for exp in $QUERY_EXPS; do
    echo "----------------------------------------"
    echo "Experimento: $exp"
    echo "----------------------------------------"
    for domain in $DOMAINS; do
        echo "  → Dominio: $domain"
        python3 scripts/run_experiment.py -e $exp -d $domain --force 2>&1 | tee -a logs/batch_run_$(date +%Y%m%d).log
    done
    echo ""
done

echo "=========================================="
echo "COMPLETADO"
echo "Fin: $(date)"
echo "=========================================="
