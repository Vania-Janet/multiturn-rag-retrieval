#!/bin/bash
# ==============================================================================
# RUN FINE-TUNED BGE RERANKER EXPERIMENTS
# ==============================================================================
# Uses pedrovo9/bge-reranker-v2-m3-multirag-finetuned model
# Fine-tuned on multi-domain conversational RAG data with proper train/test splits
# ==============================================================================

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Experiments to run
EXPERIMENTS=(
    "A10_finetuned_reranker"
    "finetune_bge_splade_bge15_rewrite"
    "finetune_bge_splade_voyage_rewrite"
)

DOMAINS=("clapnq" "cloud" "fiqa" "govt")

LOG_DIR="logs/experiments/05-finetune"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/run_all_${TIMESTAMP}.log"

# ==============================================================================
# FUNCTIONS
# ==============================================================================

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1" | tee -a "$MAIN_LOG"
}

log_success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] ✓${NC} $1" | tee -a "$MAIN_LOG"
}

log_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ✗${NC} $1" | tee -a "$MAIN_LOG"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] ⚠${NC} $1" | tee -a "$MAIN_LOG"
}

print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"
}

# ==============================================================================
# MAIN
# ==============================================================================

print_header "FINE-TUNED BGE RERANKER EXPERIMENTS"

log "Model: pedrovo9/bge-reranker-v2-m3-multirag-finetuned"
log "Experiments: ${#EXPERIMENTS[@]}"
log "Domains: ${#DOMAINS[@]}"
log "Total runs: $((${#EXPERIMENTS[@]} * ${#DOMAINS[@]}))"
log ""

# Check if model is accessible
log "Checking model availability..."
python3 -c "
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained('pedrovo9/bge-reranker-v2-m3-multirag-finetuned')
    print('✓ Model accessible')
except Exception as e:
    print(f'✗ Error: {e}')
    exit(1)
" 2>&1 | tee -a "$MAIN_LOG"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log_error "Cannot access fine-tuned model. Check Hugging Face connection."
    exit 1
fi

log_success "Model accessible"
log ""

# Run experiments
TOTAL_RUNS=$((${#EXPERIMENTS[@]} * ${#DOMAINS[@]}))
CURRENT_RUN=0
FAILED_RUNS=()

for experiment in "${EXPERIMENTS[@]}"; do
    for domain in "${DOMAINS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        
        print_header "[$CURRENT_RUN/$TOTAL_RUNS] $experiment - $domain"
        
        EXP_LOG="$LOG_DIR/${experiment}_${domain}_${TIMESTAMP}.log"
        
        # Check if already completed
        METRICS_FILE="experiments/05-finetune/$experiment/$domain/metrics.json"
        if [ -f "$METRICS_FILE" ]; then
            log_warning "Already completed, skipping..."
            continue
        fi
        
        log "Running experiment..."
        
        # Run experiment
        if python scripts/run_experiment.py \
            --experiment "$experiment" \
            --domain "$domain" \
            --config-dir "configs" \
            --output-dir "experiments" \
            2>&1 | tee "$EXP_LOG"; then
            log_success "Completed: $experiment - $domain"
        else
            log_error "Failed: $experiment - $domain"
            FAILED_RUNS+=("$experiment - $domain")
        fi
        
        log ""
    done
done

# Summary
print_header "SUMMARY"

log "Total runs: $TOTAL_RUNS"
log "Successful: $((TOTAL_RUNS - ${#FAILED_RUNS[@]}))"
log "Failed: ${#FAILED_RUNS[@]}"

if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    log_error "Failed runs:"
    for run in "${FAILED_RUNS[@]}"; do
        log_error "  - $run"
    done
    exit 1
else
    log_success "All experiments completed successfully!"
fi

# Aggregate results
log ""
log "Aggregating results..."
python scripts/aggregate_results.py --experiment-dir experiments/05-finetune 2>&1 | tee -a "$MAIN_LOG"

log_success "Done! Results saved to experiments/05-finetune/"
log "Main log: $MAIN_LOG"
