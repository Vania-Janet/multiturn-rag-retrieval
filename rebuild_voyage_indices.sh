#!/bin/bash

echo "=========================================="
echo "REGENERANDO ÍNDICES VOYAGE CON IndexFlatIP"
echo "=========================================="
date

DOMAINS="clapnq cloud fiqa govt"

for domain in $DOMAINS; do
    echo ""
    echo ">>> Regenerando índice para $domain"
    python3 src/pipeline/indexing/voyage_gen/create_faiss_databases.py \
        --domain $domain \
        --force
done

echo ""
echo "=========================================="
echo "COMPLETADO"
echo "=========================================="
date
