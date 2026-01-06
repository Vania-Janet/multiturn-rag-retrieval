#!/bin/bash
# Script para correr experimentos baseline con checkpoints

set -e
set -o pipefail

DOMAINS=("clapnq" "cloud" "fiqa" "govt")
EXPERIMENTS=("replication_bm25" "replication_bge15" "replication_bgem3" "replication_splade")

LOG_DIR="logs/experiments"
CHECKPOINT_DIR="experiments/.checkpoints"
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

echo "════════════════════════════════════════════════════════════════"
echo "  Iniciando experimentos baseline (con SPLADE)"
echo "════════════════════════════════════════════════════════════════"
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    echo "📊 Experimento: $exp"
    
    for domain in "${DOMAINS[@]}"; do
        CHECKPOINT_FILE="$CHECKPOINT_DIR/${exp}_${domain}.done"
        
        # Verificar si ya se completó
        if [ -f "$CHECKPOINT_FILE" ]; then
            echo "  ✅ $domain - Ya completado (checkpoint existe)"
            continue
        fi
        
        echo "  🔄 Procesando $domain..."
        
        # Correr experimento
        if python scripts/run_experiment.py \
            --experiment "$exp" \
            --domain "$domain" \
            --output-dir "experiments/baselines" \
            2>&1 | tee "$LOG_DIR/${exp}_${domain}.log"; then
            
            # Crear checkpoint si tuvo éxito
            touch "$CHECKPOINT_FILE"
            echo "  ✅ $domain - Completado"
        else
            echo "  ❌ $domain - Falló (revisa log: $LOG_DIR/${exp}_${domain}.log)"
            exit 1
        fi
    done
    
    echo ""
done

echo "════════════════════════════════════════════════════════════════"
echo "  ✅ Todos los experimentos baseline completados"
echo "════════════════════════════════════════════════════════════════"
