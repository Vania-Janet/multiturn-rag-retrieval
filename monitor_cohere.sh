#!/bin/bash
# Monitor Cohere reranking experiments progress

echo "═══════════════════════════════════════════════════════════════════════"
echo "  COHERE RERANKING EXPERIMENTS - Status Monitor"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# Check running processes
RUNNING=$(ps aux | grep "pipeline.run.*cohere" | grep -v grep | wc -l)
echo "🔄 Procesos activos: $RUNNING/4"
echo ""

# Check each experiment
for domain in clapnq govt cloud fiqa; do
    LOG="/tmp/cohere_${domain}.log"
    if [ -f "$LOG" ]; then
        # Get last few lines with info
        LAST_LINE=$(tail -50 "$LOG" | grep -E "INFO|ERROR" | tail -1)
        
        # Count queries processed (look for voyage similarity conversions as proxy)
        PROCESSED=$(grep -c "Converted L2 distances" "$LOG" 2>/dev/null || echo "0")
        
        # Check for errors
        ERRORS=$(grep -c "ERROR" "$LOG" 2>/dev/null || echo "0")
        
        # Check if completed
        if grep -q "Results saved" "$LOG" 2>/dev/null; then
            STATUS="✅ COMPLETADO"
        elif [ "$ERRORS" -gt 5 ]; then
            STATUS="❌ ERRORES ($ERRORS)"
        elif [ "$RUNNING" -eq 0 ]; then
            STATUS="⏹️  DETENIDO"
        else
            STATUS="⏳ Corriendo"
        fi
        
        echo "[$STATUS] ${domain^^}: ~$PROCESSED queries procesadas | Errores: $ERRORS"
        
    else
        echo "[❓] ${domain^^}: Log no encontrado"
    fi
done

echo ""
echo "───────────────────────────────────────────────────────────────────────"
echo "Para ver logs en tiempo real:"
echo "  watch -n 5 'bash monitor_cohere.sh'"
echo "  tail -f /tmp/cohere_clapnq.log"
echo ""
echo "Estimación: ~10-15 minutos por dominio con rate limiting (1.8s/query)"
echo "═══════════════════════════════════════════════════════════════════════"
