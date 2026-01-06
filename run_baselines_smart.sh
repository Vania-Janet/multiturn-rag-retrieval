#!/bin/bash
# Ejecutar baselines en paralelo, evitando conflictos

DOMAINS="clapnq cloud fiqa govt"
# Quitamos A0_baseline_bm25_fullhist de la lista prioritaria si da problemas, 
# pero probemos de nuevo ya que lo matamos.
BASELINES="replication_bm25 replication_splade replication_bgem3 replication_bge15 A0_baseline_splade_fullhist A1_baseline_bgem3_fullhist A0_baseline_bm25_fullhist"

echo "=== LANZANDO BASELINES EN PARALELO ==="
date

for exp in $BASELINES; do
    echo "Lanzando $exp..."
    for domain in $DOMAINS; do
        log_file="logs/experiments/${exp}_${domain}.log"
        # Ejecutar en background (nohup)
        nohup python3 scripts/run_experiment.py -e $exp -d $domain --force > "$log_file" 2>&1 &
    done
    # Esperar un poco entre lanzamientos para no saturar CPU (BM25 es single core pero intenso)
    sleep 5
done

echo "Todos los procesos lanzados en background."
echo "Monitorea con: tail -f logs/experiments/*.log"
