#!/bin/bash

# Ensure we are in the correct directory
# Assumes run from task_a_retrieval/

export PYTHONPATH=$PYTHONPATH:$(pwd)
export HF_HOME=/workspace/cache/huggingface
export TMPDIR=/workspace/tmp
mkdir -p $TMPDIR

experiments=(
    "bm25_r1_condensation"
    "bm25_r2_multi"
    "splade_r1_condensation"
    "bgem3_r1_condensation"
    "bgem3_r2_multi"
    "voyage_r1_condensation"
    "voyage_r2_multi"
)

domains=("clapnq" "cloud" "fiqa" "govt")

log_file="experiments_run.log"

echo "Starting benchmarks at $(date)" > $log_file

for exp in "${experiments[@]}"; do
    for dom in "${domains[@]}"; do
        # Skip if already completed
        if [ -f "experiments/01-query/$exp/$dom/metrics.json" ]; then
            echo "SKIPPED: $exp on $dom (already exists)" >> $log_file
            continue
        fi
        
        echo "--------------------------------------------------" >> $log_file
        echo "Running $exp on $dom at $(date)" >> $log_file
        
        # Run in separate process to clear GPU memory
        /workspace/myenv/bin/python3 scripts/run_experiment.py --experiment "$exp" --domain "$dom" --force >> $log_file 2>&1
        
        status=$?
        if [ $status -eq 0 ]; then
            echo "SUCCESS: $exp on $dom" >> $log_file
            echo "Uploading results for $exp..." >> $log_file
            /workspace/myenv/bin/python3 scripts/hf_sync.py --upload experiments/"$exp" >> $log_file 2>&1
        else
            echo "FAILURE: $exp on $dom (Exit code $status)" >> $log_file
        fi
        
        # Small sleep to ensure handles release
        sleep 5
    done
done

echo "All benchmarks completed at $(date)" >> $log_file
