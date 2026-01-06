#!/bin/bash
# Script para correr experimentos Full History (Voyage y SPLADE)

set -e
set -o pipefail

# Intentar cargar .env si existe
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Asegurarse que tenemos la API KEY
if [ -z "$VOYAGE_API_KEY" ]; then
    echo "❌ Error: VOYAGE_API_KEY no está definida."
    echo "Por favor exporta tu clave antes de correr el script:"
    echo "export VOYAGE_API_KEY='tu-clave-voyage'"
    exit 1
fi

DOMAINS=("clapnq" "cloud" "fiqa" "govt")
EXPERIMENTS=("A1_baseline_voyage_fullhist" "A0_baseline_splade_fullhist")

LOG_DIR="logs/experiments"
CHECKPOINT_DIR="experiments/.checkpoints"
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

echo "════════════════════════════════════════════════════════════════"
echo "  Iniciando experimentos Full History"
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
