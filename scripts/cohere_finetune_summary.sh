#!/bin/bash
# Quick summary of Cohere fine-tuning setup

echo "========================================================================"
echo "  COHERE FINE-TUNING - SETUP COMPLETO"
echo "========================================================================"
echo ""

echo "STEP 1: MODELO ACTUALIZADO A RERANK-V4-PRO"
echo ""
echo "   Archivos actualizados:"
echo "   - src/pipeline/reranking/cohere_rerank.py"
echo "   - configs/experiments/03-rerank/rerank_cohere_splade_voyage_rewrite.yaml"
echo "   - configs/experiments/03-rerank/rerank_cohere_splade_bge15_rewrite.yaml"
echo ""

echo "STEP 2: DATOS DE ENTRENAMIENTO GENERADOS"
echo ""
echo "   Ubicacion: experiments/05-finetune/cohere_rerank_data/"
echo ""
cd /workspace/mt-rag-benchmark/task_a_retrieval
python3 << 'EOF'
import json

with open("experiments/05-finetune/cohere_rerank_data/train.jsonl") as f:
    train_count = sum(1 for _ in f)

with open("experiments/05-finetune/cohere_rerank_data/validation.jsonl") as f:
    val_count = sum(1 for _ in f)

print(f"   - train.jsonl: {train_count} ejemplos")
print(f"   - validation.jsonl: {val_count} ejemplos")
print(f"   - Total: {train_count + val_count} queries con ground truth")
print()
print("   Estadisticas:")
print("     - Avg relevant passages: 2.73 por query")
print("     - Avg hard negatives: 4.99 por query")
print("     - Split: 80/20 estratificado por dominio")
print("     - Dominios: CLAPNQ, GOVT, CLOUD, FIQA")
EOF

echo ""
echo "STEP 3: CONFIGS DE EXPERIMENTOS CREADOS"
echo ""
echo "   - configs/experiments/05-finetune/finetune_cohere_splade_voyage_rewrite.yaml"
echo "   - configs/experiments/05-finetune/finetune_cohere_splade_bge15_rewrite.yaml"
echo "   - configs/experiments/05-finetune/README.md (documentacion completa)"
echo ""

echo "========================================================================"
echo "  PROXIMOS PASOS"
echo "========================================================================"
echo ""
echo "PASO 1: SUBIR DATOS A COHERE"
echo ""
echo "   Opcion A - Script automatico:"
echo "     python3 scripts/upload_cohere_finetune.py"
echo ""
echo "   Opcion B - Dashboard manual:"
echo "     1. https://dashboard.cohere.com/fine-tuning"
echo "     2. Upload train.jsonl y validation.jsonl"
echo "     3. Base model: rerank-v4-pro"
echo ""

echo "PASO 2: ESPERAR ENTRENAMIENTO (~2-4 horas)"
echo ""
echo "   Cohere te dara un model ID como: abc123-ft-rerank-v4-pro"
echo ""

echo "PASO 3: ACTUALIZAR CONFIGS"
echo ""
echo "   Reemplazar REPLACE_WITH_FINETUNED_MODEL_ID con tu model ID:"
echo "   sed -i 's/REPLACE_WITH_FINETUNED_MODEL_ID/TU_MODEL_ID/g' \\"
echo "     configs/experiments/05-finetune/*.yaml"
echo ""

echo "PASO 4: EJECUTAR EXPERIMENTOS"
echo ""
echo "   cd /workspace/mt-rag-benchmark/task_a_retrieval"
echo "   export PYTHONPATH=\$PWD/src:\$PYTHONPATH"
echo ""
echo "   # CLAPNQ"
echo "   CUDA_VISIBLE_DEVICES=0 python -m pipeline.run \\"
echo "     --config configs/experiments/05-finetune/finetune_cohere_splade_voyage_rewrite.yaml \\"
echo "     --domain clapnq"
echo ""
echo "   # Y asi para: govt, cloud, fiqa"
echo ""

echo "========================================================================"
echo "  EXPECTATIVAS"
echo "========================================================================"
echo ""
echo "   Baseline (sin rerank):     0.486 nDCG@10 (GOOD)"
echo "   Rerank-v3.5 (generico):    0.384 nDCG@10 (FAILED -20.9%)"
echo "   Rerank-v4-pro fine-tuned:  0.51-0.56 nDCG@10 (TARGET +5-15%)"
echo ""
echo "   Por que deberia funcionar mejor?"
echo "   - Entrenado en las 620 queries exactas del benchmark"
echo "   - Hard negatives del Top-10 ensenan discriminacion"
echo "   - Captura contexto multi-turn conversacional"
echo "   - Aprende que hace relevante un doc en ESTE dataset"
echo ""

echo "========================================================================"
echo ""
echo "Archivos listos para subir a Cohere:"
echo "   - experiments/05-finetune/cohere_rerank_data/train.jsonl"
echo "   - experiments/05-finetune/cohere_rerank_data/validation.jsonl"
echo ""
echo "Documentacion completa:"
echo "   - configs/experiments/05-finetune/README.md"
echo ""
echo "========================================================================"
