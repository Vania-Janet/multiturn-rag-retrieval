#!/bin/bash
# Script para correr experimentos de Query Rewriting

set -e
set -o pipefail

# Intentar cargar .env si existe
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Asegurarse que tenemos la API KEY si vamos a usar GPT
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  ADVERTENCIA: OPENAI_API_KEY no encontrada."
    echo "Los experimentos que usan gpt-4o-mini fallarán si no se configura."
fi

# Lista de experimentos en configs/experiments/01-query/
# bm25_r1_condensation.yaml -> BM25 + Rewrite (Single)
# bm25_r2_multi.yaml        -> BM25 + Rewrite (Multi K=3) + RRF
# bgem3_r1_condensation.yaml -> BGE-M3 + Rewrite (Single)
# bgem3_r2_multi.yaml       -> BGE-M3 + Rewrite (Multi)
# splade_r1_condensation.yaml -> SPLADE + Rewrite (Single)
# splade_r3_hyde.yaml       -> SPLADE + HyDE (Hypothetical Document Embeddings)

DOMAINS=("clapnq" "cloud" "fiqa" "govt")
EXPERIMENTS=(
    "bm25_r1_condensation"
    "bm25_r2_multi"
    "bgem3_r1_condensation"
    "bgem3_r2_multi"
    "splade_r1_condensation"
    "splade_r3_hyde"
    "voyage_r1_condensation"
    "voyage_r2_multi"
)

LOG_DIR="logs/experiments"
CHECKPOINT_DIR="experiments/.checkpoints"
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

echo "════════════════════════════════════════════════════════════════"
echo "  Iniciando experimentos de Query Rewriting"
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
            --output-dir "experiments/query_rewrite" \
            2>&1 | tee "$LOG_DIR/${exp}_${domain}.log"; then
            
            # Crear checkpoint si tuvo éxito
            touch "$CHECKPOINT_FILE"
            echo "  ✅ $domain - Completado"
        else
            echo "  ❌ $domain - Falló (revisa log: $LOG_DIR/${exp}_${domain}.log)"
            # No hacemos exit 1 aquí para permitir que otros dominios/experimentos continúen
            # si falla uno (por ejemplo, por cuota de API)
        fi
    done
    
    echo ""
done
