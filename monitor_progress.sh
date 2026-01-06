#!/bin/bash
# Script para monitorear el progreso de los experimentos

clear
echo "=========================================="
echo "MONITOR DE EXPERIMENTOS"
echo "=========================================="
echo ""

# Función para contar resultados
count_results() {
    local path=$1
    local count=$(find "$path" -name "metrics.json" 2>/dev/null | wc -l)
    echo $count
}

echo "BASELINES (0-baselines/):"
echo "-------------------------"
for exp in A0_baseline_splade_fullhist A0_baseline_bm25_fullhist replication_bm25 replication_splade replication_bgem3 replication_bge15 A1_baseline_voyage_fullhist replication_voyage; do
    count=$(count_results "experiments/0-baselines/$exp")
    printf "  %-35s %d/4 dominios\n" "$exp" "$count"
done

echo ""
echo "QUERY EXPERIMENTS (01-query/):"
echo "------------------------------"
for exp in bm25_r1_condensation bm25_r2_multi bgem3_r1_condensation bgem3_r2_multi splade_r1_condensation splade_r3_hyde voyage_r1_condensation voyage_r2_multi; do
    count=$(count_results "experiments/01-query/$exp")
    printf "  %-35s %d/4 dominios\n" "$exp" "$count"
done

echo ""
echo "=========================================="
echo "ÚLTIMO EXPERIMENTO EN LOG:"
echo "=========================================="
tail -5 logs/batch_run_$(date +%Y%m%d).log 2>/dev/null || echo "No hay log disponible aún"

echo ""
echo "Para ver el progreso en tiempo real:"
echo "  screen -r experiments"
echo "Para salir de screen sin detener: Ctrl+A, luego D"
