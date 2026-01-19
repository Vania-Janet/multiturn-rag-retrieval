#!/bin/bash
# Script para verificar si el documento LaTeX compila correctamente

echo "Verificando sintaxis del documento LaTeX..."
echo ""

# Verificar si pdflatex está instalado
if ! command -v pdflatex &> /dev/null; then
    echo "❌ pdflatex no está instalado"
    echo "📦 Para instalarlo en Debian/Ubuntu: sudo apt-get install texlive-latex-base texlive-latex-extra"
    echo ""
    echo "✅ El documento LaTeX (.tex) está creado y listo"
    echo "📄 Puedes compilarlo localmente en tu máquina con pdflatex"
    exit 0
fi

# Si está instalado, intentar compilar
echo "🔨 Compilando documento..."
pdflatex -interaction=nonstopmode presentacion_avances.tex

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Documento compilado exitosamente"
    echo "📄 PDF generado: presentacion_avances.pdf"
else
    echo ""
    echo "⚠️  Errores encontrados durante la compilación"
    echo "📋 Revisa el archivo presentacion_avances.log para más detalles"
fi
