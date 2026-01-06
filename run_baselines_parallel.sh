#!/bin/bash
# Ejecutar SOLO baselines en paralelo (4 dominios simultáneos)

DOMAINS="clapnq cloud fiqa govt"
BASELINES="A0_baseline_bm25_fullhist A0_baseline_splade_fullhist A1_baseline_bgem3_fullhist replication_bge15 replication_bgem3 replication_bm25 replication_splade"

echo "=========================================="
echo "EJECUTANDO BASELINES EN PARALELO"
echo "=========================================="
date

for exp in $BASELINES; do
    echo ""
    echo ">>> Experimento: $exp (4 dominios en paralelo)"
    
    # Lanzar los 4 dominios en paralelo
    for domain in $DOMAINS; do
        (
            echo "  [$(date '+%H:%M:%S')] Iniciando $exp/$domain"
            python3 scripts/run_experiment.py -e $exp -d $domain --force 2>&1 | \
                grep -E "(INFO - ✓|ERROR|Traceback)" | \
                sed "s/^/  [$domain] /"
            echo "  [$(date '+%H:%M:%S')] ✓ Completado $exp/$domain"
        ) &
    done
    
    # Esperar a que terminen los 4 antes de continuar
    wait
    echo "  ✓ $exp completado en todos los dominios"
done

echo ""
echo "=========================================="
echo "TODOS LOS BASELINES COMPLETADOS"
echo "=========================================="
date
