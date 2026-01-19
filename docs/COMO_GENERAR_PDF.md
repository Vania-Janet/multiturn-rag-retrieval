# 🚀 INSTRUCCIONES RÁPIDAS - GENERAR PDF

## Opción 1: Overleaf (RECOMENDADO - Más Fácil) ⭐

### Pasos:
1. Ve a https://www.overleaf.com
2. Crea cuenta gratuita (si no tienes)
3. Click en "New Project" → "Upload Project"
4. Sube el archivo `presentacion_avances.tex`
5. Click en "Recompile"
6. ¡Listo! Descarga el PDF

**Ventajas:**
- ✅ No requiere instalación
- ✅ Compila automáticamente
- ✅ Muestra errores claros
- ✅ Funciona en cualquier dispositivo

---

## Opción 2: Compilación Local (Si tienes LaTeX instalado)

### En Linux/Mac:
```bash
cd /workspace/mt-rag-benchmark/task_a_retrieval/docs
pdflatex presentacion_avances.tex
pdflatex presentacion_avances.tex  # Segunda vez para tabla de contenidos
```

### En Windows:
1. Instalar MiKTeX: https://miktex.org/download
2. Abrir cmd en la carpeta docs
3. Ejecutar:
```cmd
pdflatex presentacion_avances.tex
pdflatex presentacion_avances.tex
```

**Resultado:** Se genera `presentacion_avances.pdf`

---

## Opción 3: Visual Studio Code con LaTeX Workshop

### Instalación:
1. Instalar extensión "LaTeX Workshop" en VS Code
2. Abrir `presentacion_avances.tex`
3. Click derecho → "Build LaTeX project"

---

## Si Encuentras Errores

### Error: "Package X not found"
**Solución en Overleaf:** Ya tiene todos los paquetes ✅

**Solución local:**
```bash
# Linux/Mac
sudo apt-get install texlive-full  # Ubuntu/Debian
brew install --cask mactex         # macOS

# Esperar 20-30 minutos (instalación grande ~4GB)
```

### Error: "Babel spanish"
**Ya corregido** en el documento ✅

---

## Archivos que Necesitas

**Para compilar PDF:**
- ✅ `presentacion_avances.tex` (único archivo necesario)

**Para slides Canva:**
- ✅ `COMPARACION_BGE_M3_SLIDES.md` (estructura)
- ✅ `resumen_ejecutivo_presentacion.md` (datos y tablas)

---

## 📋 Checklist Pre-Compilación

- [ ] Archivo `presentacion_avances.tex` disponible
- [ ] Si usas Overleaf: Cuenta creada
- [ ] Si usas local: LaTeX instalado (texlive o MiKTeX)

---

## 🎯 Resultado Final

Después de compilar tendrás:
```
presentacion_avances.pdf (documento académico completo)
├─ 9 secciones
├─ 6 tablas
├─ 511 líneas
├─ Análisis riguroso
├─ Comparación BGE-m3
└─ Conclusiones y trabajo futuro
```

---

## ⏱️ Tiempo Estimado

| Método | Tiempo |
|--------|--------|
| Overleaf | 2-3 minutos |
| Local (LaTeX ya instalado) | 1 minuto |
| Local (instalar LaTeX primero) | 30-40 minutos |

---

## 💡 Tips

1. **Primera vez compilando LaTeX?** → Usa Overleaf
2. **Ya tienes LaTeX?** → Compila localmente (más rápido)
3. **Quieres editar después?** → Overleaf permite editar online

---

## 🆘 Ayuda

Si algo no funciona:
1. Lee `GUIA_COMPILACION.md` (guía detallada)
2. Revisa `FIX_LATEX_ERROR.md` (errores comunes ya corregidos)
3. Usa Overleaf (opción más segura)

---

## ✅ Verificación Final

Para confirmar que el PDF se generó correctamente:
- ✅ Tabla de contenidos presente (página 2)
- ✅ 6 tablas visibles
- ✅ Sección "Comparación con BGE-m3 Multi-Vector"
- ✅ Código Python visible en sección de retos
- ✅ Referencias cruzadas funcionan

**Total esperado:** ~15-18 páginas

---

## 🎉 ¡Éxito!

Una vez que tengas el PDF:
1. Revisalo completo
2. Guarda backup
3. Prepara tu presentación oral
4. ¡A brillar con tu profesora! ✨
