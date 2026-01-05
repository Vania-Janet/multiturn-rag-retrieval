#!/bin/bash
# Script para monitorear el progreso de los experimentos

echo "════════════════════════════════════════════════════════════════"
echo "  ESTADO DE EXPERIMENTOS BASELINE"
echo "  $(date)"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Verificar screen session
if screen -ls | grep -q "baseline_experiments"; then
    echo "✅ Screen session 'baseline_experiments' está activa"
else
    echo "❌ Screen session 'baseline_experiments' no está corriendo"
fi
echo ""

# Contar checkpoints
TOTAL_EXPECTED=12  # 3 experiments × 4 domains
COMPLETED=$(find experiments/.checkpoints/ -name "*.done" 2>/dev/null | wc -l)
echo "📊 Progreso: $COMPLETED/$TOTAL_EXPECTED experimentos completados"
echo ""

# Mostrar checkpoints completados
echo "✅ Completados:"
find experiments/.checkpoints/ -name "*.done" 2>/dev/null | sort | sed 's|.*/||; s|.done||' | sed 's/^/  - /'
echo ""

# Mostrar último log
echo "📝 Últimas 15 líneas del log principal:"
tail -n 15 logs/baseline_experiments_master.log 2>/dev/null || echo "  (log no disponible aún)"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Comandos útiles:"
echo "  - Ver log en tiempo real: tail -f logs/baseline_experiments_master.log"
echo "  - Reconectar a screen: screen -r baseline_experiments"
echo "  - Ver todos los logs: ls -lh logs/experiments/"
echo "  - Este script: ./monitor_experiments.sh"
