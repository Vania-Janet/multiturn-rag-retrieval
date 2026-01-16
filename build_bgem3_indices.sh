#!/bin/bash
# Build BGE-M3 indices (Dense, Sparse, ColBERT) for all domains in one optimized pass
# Using GPU resources

cd /workspace/mt-rag-benchmark/task_a_retrieval

output_dir="indices"
corpus_dir="data/passage_level_processed"
python_cmd=".venv/bin/python"

if [ ! -f "$python_cmd" ]; then
    python_cmd="python"
fi

echo "Building ALL BGE-M3 indices using optimized single-pass method..."

# Use "bgem3_all" to trigger simultaneous generation
model="bgem3_all"

echo "========================================================"
echo "Starting optimized build for $model..."
echo "========================================================"

# Using batch size of 64, which scripts/build_indices.py will divide by 4 for safety -> effective batch 16
$python_cmd scripts/build_indices.py     --model "$model"     --domain "all"     --corpus-dir "$corpus_dir"     --output-dir "$output_dir"     --batch-size 64     --verbose

if [ $? -eq 0 ]; then
    echo "Successfully built BGE-M3 indices (Dense, Sparse, ColBERT) for all domains."
else
    echo "Failed to build indices."
    exit 1
fi
