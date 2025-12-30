#!/bin/bash
# Monitor and Sync Script
# Runs in the background and syncs artifacts to Hugging Face every 30 minutes.
# Usage: ./scripts/monitor_sync.sh &

INTERVAL=1800 # 30 minutes in seconds

echo "Starting HF Sync Monitor (Interval: ${INTERVAL}s)..."

while true; do
    echo "[$(date)] Syncing artifacts to Hugging Face..."
    # We use --upload-all to sync everything defined in LARGE_DIRS
    python scripts/hf_sync.py --upload-all
    
    if [ $? -eq 0 ]; then
        echo "[$(date)] Sync successful."
    else
        echo "[$(date)] Sync failed. Retrying next cycle."
    fi
    
    echo "[$(date)] Sleeping for ${INTERVAL}s..."
    sleep $INTERVAL
done
