#!/bin/bash
# Verificación rápida de los fixes aplicados

echo "========================================"
echo "Verificación de Fixes 01-Query"
echo "========================================"
echo ""

cd /workspace/mt-rag-benchmark/task_a_retrieval

echo "1. Verificando que los archivos tasks.jsonl existen..."
for domain in clapnq cloud fiqa govt; do
    file="data/retrieval_tasks/$domain/${domain}_tasks.jsonl"
    if [ -f "$file" ]; then
        count=$(wc -l < "$file")
        echo "  ✓ $domain: $count queries"
    else
        echo "  ✗ $domain: FALTA $file"
    fi
done
echo ""

echo "2. Verificando configs de experimentos 01-query..."
for config in configs/experiments/01-query/*.yaml; do
    name=$(basename "$config" .yaml)
    
    # Check if has retrieval_type
    if grep -q "retrieval_type:" "$config"; then
        retrieval_type=$(grep "retrieval_type:" "$config" | awk '{print $2}')
        echo "  ✓ $name: retrieval_type = $retrieval_type"
    else
        echo "  ✗ $name: FALTA retrieval_type"
    fi
    
    # Check if has query_file override
    if grep -q "query_file:" "$config"; then
        echo "    ✓ tiene query_file override"
    else
        echo "    ✗ FALTA query_file override"
    fi
done
echo ""

echo "3. Verificando que VLLMRewriter tiene los nuevos prompts..."
if grep -q "CONDENSATION_PROMPT_SPARSE" src/pipeline/query_transform/rewriters.py; then
    echo "  ✓ CONDENSATION_PROMPT_SPARSE encontrado"
else
    echo "  ✗ FALTA CONDENSATION_PROMPT_SPARSE"
fi

if grep -q "CONDENSATION_PROMPT_DENSE" src/pipeline/query_transform/rewriters.py; then
    echo "  ✓ CONDENSATION_PROMPT_DENSE encontrado"
else
    echo "  ✗ FALTA CONDENSATION_PROMPT_DENSE"
fi

if grep -q "MULTIQUERY_PROMPT_SPARSE" src/pipeline/query_transform/rewriters.py; then
    echo "  ✓ MULTIQUERY_PROMPT_SPARSE encontrado"
else
    echo "  ✗ FALTA MULTIQUERY_PROMPT_SPARSE"
fi

if grep -q "MULTIQUERY_PROMPT_DENSE" src/pipeline/query_transform/rewriters.py; then
    echo "  ✓ MULTIQUERY_PROMPT_DENSE encontrado"
else
    echo "  ✗ FALTA MULTIQUERY_PROMPT_DENSE"
fi

if grep -q "retrieval_type: str = \"dense\"" src/pipeline/query_transform/rewriters.py; then
    echo "  ✓ Parámetro retrieval_type añadido al constructor"
else
    echo "  ✗ FALTA parámetro retrieval_type"
fi
echo ""

echo "4. Verificando función substitute_domain en run_experiment.py..."
if grep -q "substitute_domain" scripts/run_experiment.py; then
    echo "  ✓ Función substitute_domain encontrada"
else
    echo "  ✗ FALTA función substitute_domain"
fi
echo ""

echo "========================================"
echo "Verificación completada"
echo "========================================"
echo ""
echo "Para re-ejecutar los experimentos:"
echo "  ./run_all_01query_experiments.sh"
echo ""
echo "Para verificar un solo experimento:"
echo "  python scripts/run_experiment.py --experiment voyage_r1_condensation --domain clapnq"
