#!/bin/bash
# Reintentar experimentos fallidos SEQUENCIALMENTE para evitar OOM

DOMAINS="clapnq cloud fiqa govt"
FAILED_EXPS="A1_baseline_bgem3_fullhist replication_bge15"

echo "=== REINTENTANDO EXPERIMENTOS FALLIDOS ==="
date

for exp in $FAILED_EXPS; do
    echo "----------------------------------------"
    echo "Experimento: $exp"
    echo "----------------------------------------"
    for domain in $DOMAINS; do
        echo "  → Dominio: $domain"
        # Sin nohup, queremos que sea secuencial y bloqueante
        python3 scripts/run_experiment.py -e $exp -d $domain --force
        echo ""
    done
done

echo "=== COMPLETADO ==="
date
