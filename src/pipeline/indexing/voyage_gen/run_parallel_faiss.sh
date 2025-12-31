#!/bin/bash

# Script to generate FAISS databases in parallel
# Runs each corpus in a separate process with logging

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="$SCRIPT_DIR/logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "STARTING PARALLEL FAISS GENERATION"
echo "=========================================="
echo "Timestamp: $TIMESTAMP"
echo "Script dir: $SCRIPT_DIR"
echo "Log dir: $LOG_DIR"
echo ""

cd "$SCRIPT_DIR"

# Function to execute and monitor a corpus
run_corpus() {
    local corpus=$1
    local log_file="$LOG_DIR/${corpus}_${TIMESTAMP}.log"
    
    echo "Starting $corpus (log: $log_file)"
    
    python create_faiss_databases.py --corpus "$corpus" > "$log_file" 2>&1
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "$corpus completed successfully"
    else
        echo "$corpus failed with code $exit_code - check $log_file"
    fi
    
    return $exit_code
}

# Start parallel processes
echo "Launching 4 parallel processes..."
echo ""

run_corpus "clapnq" &
PID_CLAPNQ=$!

run_corpus "cloud" &
PID_CLOUD=$!

run_corpus "fiqa" &
PID_FIQA=$!

run_corpus "govt" &
PID_GOVT=$!

# Wait for all processes to finish
echo "Waiting for processes to complete..."


# Capture exit codes
wait $PID_CLAPNQ
EXIT_CLAPNQ=$?

wait $PID_CLOUD
EXIT_CLOUD=$?

wait $PID_FIQA
EXIT_FIQA=$?

wait $PID_GOVT
EXIT_GOVT=$?

# Final Summary
echo ""
echo "=========================================="
echo "EXECUTION SUMMARY"
echo "=========================================="
echo "clapnq: $([ $EXIT_CLAPNQ -eq 0 ] && echo 'Success' || echo 'Failed')"
echo "cloud:  $([ $EXIT_CLOUD -eq 0 ] && echo 'Success' || echo 'Failed')"
echo "fiqa:   $([ $EXIT_FIQA -eq 0 ] && echo 'Success' || echo 'Failed')"
echo "govt:   $([ $EXIT_GOVT -eq 0 ] && echo 'Success' || echo 'Failed')"
echo ""

# Verify generated files
echo "Generated files:"
for corpus in clapnq cloud fiqa govt; do
    # Note: The python script now outputs to ../../../../indices/{corpus}/voyage
    # We need to check the correct path.
    # Based on the python script logic: output_dir = project_root / "indices"
    # And it creates {corpus}/voyage/faiss_index.bin
    
    # Go up 4 levels from script dir to get root
    PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")")"
    INDEX_PATH="$PROJECT_ROOT/indices/$corpus/voyage/faiss_index.bin"
    
    if [ -f "$INDEX_PATH" ]; then
        size=$(du -h "$INDEX_PATH" | cut -f1)
        echo "  $corpus/voyage/faiss_index.bin ($size)"
    else
        echo "  $corpus/voyage/faiss_index.bin (not found)"
    fi
done

echo ""
echo "Logs saved in: $LOG_DIR"
echo "=========================================="

# Exit with error if any failed
if [ $EXIT_CLAPNQ -ne 0 ] || [ $EXIT_CLOUD -ne 0 ] || [ $EXIT_FIQA -ne 0 ] || [ $EXIT_GOVT -ne 0 ]; then
    echo "At least one process failed. Check the logs."
    exit 1
else
    echo "All processes completed successfully!"
    exit 0
fi
