#!/bin/bash
# Monitorear progreso de re-ejecución de experimentos

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 MONITOREO DE RE-EJECUCIÓN"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Verificar si el proceso está corriendo
if ps aux | grep -v grep | grep "rerun_all_hybrid_experiments.sh" > /dev/null; then
    echo "✅ Proceso activo"
else
    echo "⚠️  Proceso no encontrado (completado o no iniciado)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 ÚLTIMAS 30 LÍNEAS DEL LOG:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
tail -n 30 rerun_hybrid_fix.log

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📁 ARCHIVOS GENERADOS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Contar métricas generadas
total_metrics=$(find experiments/02-hybrid -name "metrics.json" -newer CRITICAL_BUG_FIX.md 2>/dev/null | wc -l)
echo "Archivos metrics.json generados después del fix: $total_metrics"

# Mostrar métricas recientes
echo ""
echo "Archivos recién modificados (últimos 5 minutos):"
find experiments/02-hybrid -name "metrics.json" -mmin -5 2>/dev/null | sort | while read f; do
    echo "  ✓ $f"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Para ver log completo: tail -f rerun_hybrid_fix.log"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
