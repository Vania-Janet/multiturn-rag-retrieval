#!/bin/bash
# ==============================================================================
# RUN ALL 01-QUERY EXPERIMENTS - OPTIMIZED FOR 2x RTX 4090
# ==============================================================================
# This script runs all query transformation experiments across all domains
# with intelligent parallelization to maximize GPU utilization.
#
# Total: 8 experiments × 4 domains = 32 runs
# Estimated time with batching: ~1-2 hours (vs 12+ hours without)
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

EXPERIMENTS=(
    "bm25_r1_condensation"
    "bm25_r2_multi"
    "splade_r1_condensation"
    "splade_r3_hyde"
    "bgem3_r1_condensation"
    "bgem3_r2_multi"
    "voyage_r1_condensation"
    "voyage_r2_multi"
)

DOMAINS=("clapnq" "cloud" "fiqa" "govt")

LOG_DIR="logs/experiments/01-query"
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
# PRE-FLIGHT CHECKS
# ==============================================================================

print_header "PRE-FLIGHT CHECKS"

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found. GPU required for these experiments."
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
log "Detected $GPU_COUNT GPU(s)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | while read line; do
    log "  GPU: $line"
done

# Check Python environment
if [ ! -d ".venv" ]; then
    log_warning "Virtual environment not found. Creating..."
    python3 -m venv .venv
fi

log "Activating virtual environment..."
source .venv/bin/activate

# Check required files
REQUIRED_FILES=(
    "scripts/run_experiment.py"
    "configs/base.yaml"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        log_error "Required file not found: $file"
        exit 1
    fi
done

log_success "All pre-flight checks passed"

# ==============================================================================
# EXPERIMENT EXECUTION
# ==============================================================================

print_header "STARTING EXPERIMENTS"

log "Total experiments to run: ${#EXPERIMENTS[@]} × ${#DOMAINS[@]} = $((${#EXPERIMENTS[@]} * ${#DOMAINS[@]}))"
log "Main log: $MAIN_LOG"

TOTAL_RUNS=$((${#EXPERIMENTS[@]} * ${#DOMAINS[@]}))
CURRENT_RUN=0
FAILED_RUNS=()
SUCCESSFUL_RUNS=()

START_TIME=$(date +%s)

# Run experiments sequentially (vLLM batching makes parallelization less beneficial)
for experiment in "${EXPERIMENTS[@]}"; do
    for domain in "${DOMAINS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        
        print_header "RUN $CURRENT_RUN/$TOTAL_RUNS: $experiment on $domain"
        
        EXP_LOG="$LOG_DIR/${experiment}_${domain}_${TIMESTAMP}.log"
        
        log "Starting: $experiment on $domain"
        log "Log file: $EXP_LOG"
        
        RUN_START=$(date +%s)
        
        # Run experiment
        if python scripts/run_experiment.py \
            --experiment "$experiment" \
            --domain "$domain" \
            --verbose \
            2>&1 | tee "$EXP_LOG"; then
            
            RUN_END=$(date +%s)
            RUN_DURATION=$((RUN_END - RUN_START))
            
            log_success "Completed: $experiment/$domain in ${RUN_DURATION}s"
            SUCCESSFUL_RUNS+=("$experiment/$domain")
            
        else
            RUN_END=$(date +%s)
            RUN_DURATION=$((RUN_END - RUN_START))
            
            log_error "Failed: $experiment/$domain (ran ${RUN_DURATION}s)"
            FAILED_RUNS+=("$experiment/$domain")
        fi
        
        # Small delay between runs to allow GPU cleanup
        sleep 2
    done
done

# ==============================================================================
# SUMMARY
# ==============================================================================

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

print_header "EXECUTION SUMMARY"

log "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
log "Successful runs: ${#SUCCESSFUL_RUNS[@]}/$TOTAL_RUNS"
log "Failed runs: ${#FAILED_RUNS[@]}/$TOTAL_RUNS"

if [ ${#SUCCESSFUL_RUNS[@]} -gt 0 ]; then
    log_success "\nSuccessful experiments:"
    for run in "${SUCCESSFUL_RUNS[@]}"; do
        echo "  ✓ $run" | tee -a "$MAIN_LOG"
    done
fi

if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    log_error "\nFailed experiments:"
    for run in "${FAILED_RUNS[@]}"; do
        echo "  ✗ $run" | tee -a "$MAIN_LOG"
    done
    exit 1
else
    log_success "\n🎉 All experiments completed successfully!"
    exit 0
fi
