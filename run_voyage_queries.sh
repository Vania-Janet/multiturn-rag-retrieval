#!/bin/bash
# Script para correr experimentos de Voyage Query Rewriting

set -e
set -o pipefail

# Intentar cargar .env si existe
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Validar claves
if [ -z "$VOYAGE_API_KEY" ] || [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: Faltan claves (VOYAGE_API_KEY o OPENAI_API_KEY)"
    exit 1
fi

DOMAINS=("clapnq" "cloud" "fiqa" "govt")
EXPERIMENTS=(
    "voyage_r1_condensation"
    "voyage_r2_multi"
)

LOG_DIR="logs/experiments"
CHECKPOINT_DIR="experiments/.checkpoints"
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

echo "════════════════════════════════════════════════════════════════"
echo "  Iniciando experimentos Voyage + Query Rewriting"
echo "════════════════════════════════════════════════════════════════"
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    echo "📊 Experimento: $exp"
    
    for domain in "${DOMAINS[@]}"; do
        CHECKPOINT_FILE="$CHECKPOINT_DIR/${exp}_${domain}.done"
        
        # Verificar checkpoint
        if [ -f "$CHECKPOINT_FILE" ]; then
            echo "  ✅ $domain - Ya completado"
            continue
        fi
        
        echo "  🔄 Procesando $domain..."
        
        if python scripts/run_experiment.py \
            --experiment "$exp" \
            --domain "$domain" \
            --output-dir "experiments/query_rewrite" \
            2>&1 | tee "$LOG_DIR/${exp}_${domain}.log"; then
            
            touch "$CHECKPOINT_FILE"
            echo "  ✅ $domain - Completado"
        else
            echo "  ❌ $domain - Falló"
        fi
    done
    echo ""
done
