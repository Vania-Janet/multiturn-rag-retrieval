# Guía de Compilación del Documento LaTeX

## ✅ Estado del Documento

El archivo `presentacion_avances.tex` ha sido verificado y está **sintácticamente correcto**:

- ✅ Todos los entornos balanceados (begin/end)
- ✅ 5 tablas de resultados
- ✅ 453 líneas de contenido
- ✅ Estructura completa para presentación académica

## 📦 Requisitos para Compilar

### En Linux (Ubuntu/Debian)

```bash
# Instalación completa (recomendado)
sudo apt-get update
sudo apt-get install texlive-latex-base \
                     texlive-latex-extra \
                     texlive-fonts-recommended \
                     texlive-lang-spanish

# O instalación mínima
sudo apt-get install texlive-full
```

### En macOS

```bash
# Opción 1: MacTeX (recomendado)
# Descargar desde: https://www.tug.org/mactex/

# Opción 2: Homebrew
brew install --cask mactex
```

### En Windows

1. Descargar MiKTeX: https://miktex.org/download
2. O descargar TeX Live: https://www.tug.org/texlive/

## 🔨 Compilar el Documento

### Método 1: Línea de Comandos

```bash
cd /workspace/mt-rag-benchmark/task_a_retrieval/docs

# Primera compilación
pdflatex presentacion_avances.tex

# Segunda compilación (para tabla de contenidos y referencias)
pdflatex presentacion_avances.tex

# Resultado: presentacion_avances.pdf
```

### Método 2: Editor LaTeX

Editores recomendados:
- **Overleaf** (online): https://www.overleaf.com
- **TeXstudio** (desktop)
- **VS Code** con extensión LaTeX Workshop

#### Overleaf (más fácil):
1. Ir a https://www.overleaf.com
2. Crear nuevo proyecto → "Upload Project"
3. Subir `presentacion_avances.tex`
4. Click en "Recompile"

## 🐛 Solución de Problemas Comunes

### Error: "Package not found"

**Síntoma:**
```
! LaTeX Error: File `booktabs.sty' not found.
```

**Solución:**
```bash
# Ubuntu/Debian
sudo apt-get install texlive-latex-extra

# macOS (si usaste brew)
brew reinstall --cask mactex
```

### Error: "Babel language spanish not found"

**Síntoma:**
```
! Package babel Error: Unknown option `spanish'.
```

**Solución:**
```bash
# Ubuntu/Debian
sudo apt-get install texlive-lang-spanish

# O cambiar en el .tex:
\usepackage[spanish]{babel} → \usepackage[english]{babel}
```

### Error: "tcolorbox.sty not found"

**Solución:**
```bash
sudo apt-get install texlive-latex-extra
```

### Warning: "Overfull hbox"

**No es crítico** - significa que alguna línea es un poco larga. El PDF se generará correctamente.

### Error con comillas en modo matemático (CORREGIDO)

**Síntoma:**
```
! Argument of \language@active@arg" has an extra }.
! Paragraph ended before \language@active@arg" was complete.
```

**Causa:**
El paquete `babel-spanish` activa el carácter `"` para tipografía especial, causando conflictos dentro de entornos matemáticos (`align*`, `equation`, etc.).

**Solución:**
✅ **Ya corregido en el documento.** Se reemplazaron las comillas dobles (`"`) por comillas simples (`'`) dentro del entorno `align*` en la sección de Query Rewriting.

**Nota técnica:** Si necesitas usar comillas dobles en modo matemático en el futuro:
- Usar `\text{\textquotedbl}` en lugar de `"`
- O usar comillas LaTeX estándar: ``` y `''`
- O agregar `\shorthandoff{"}` antes del entorno matemático

### Error de encoding UTF-8

**Síntoma:**
```
! Package inputenc Error: Invalid UTF-8 byte sequence.
```

**Solución:**
Asegurar que el archivo está guardado en UTF-8:
```bash
file presentacion_avances.tex
# Debe mostrar: UTF-8 Unicode text
```

## 📄 Archivos Generados

Después de compilar exitosamente:

```
docs/
├── presentacion_avances.tex       # Fuente LaTeX (original)
├── presentacion_avances.pdf       # PDF final ✅
├── presentacion_avances.aux       # Auxiliar (puede ignorarse)
├── presentacion_avances.log       # Log de compilación
├── presentacion_avances.toc       # Tabla de contenidos
└── presentacion_avances.out       # Hyperlinks (opcional)
```

## 🎯 Compilación Rápida sin Instalación

### Usar Overleaf (Recomendado para presentación)

1. **Ir a**: https://www.overleaf.com
2. **Crear cuenta gratis**
3. **New Project** → **Upload Project**
4. **Arrastra** `presentacion_avances.tex`
5. **Click** en "Recompile"
6. **Descargar PDF**

**Ventajas:**
- ✅ No requiere instalación local
- ✅ Colaboración en tiempo real
- ✅ Todos los paquetes pre-instalados
- ✅ Preview instantáneo

## 📊 Verificación del Contenido

El documento incluye:

### Secciones principales:
1. **Introducción y Contexto** (datasets, métricas)
2. **Metodología y Tecnologías** (stack completo)
3. **Diseño Experimental** (tabla de ablación)
4. **Resultados** (2 tablas: nDCG@10 y Recall@100)
5. **Análisis por Dominio** (4 subsecciones)
6. **Retos y Soluciones** (bug crítico)
7. **Conclusiones** (configuraciones óptimas)
8. **Trabajo Futuro**
9. **Apéndice** (reproducibilidad)

### Tablas incluidas:
1. Tabla 1: Características de datasets
2. Tabla 2: Diseño de ablación
3. **Tabla 3: nDCG@10 (híbridos)** ← Principal
4. **Tabla 4: Recall@100 (híbridos)** ← Principal
5. Tabla 5: Mejores configuraciones

### Elementos visuales:
- 2 cajas destacadas (tcolorbox)
- 1 ecuación (RRF)
- 3 bloques de código (lstlisting)
- Formato profesional con colores

## 💡 Tips para la Presentación

### Extraer tablas para Canva:

Las tablas ya están en formato LaTeX profesional. Para Canva:

1. **Compilar PDF** con pdflatex
2. **Screenshot de tablas** desde el PDF
3. **Importar en Canva** como imágenes
4. O **copiar números** directamente del `.tex`

### Datos clave para slides:

```
ClapNQ:  Cohere 0.632 (+12.4% vs no-rewrite) 🏆
Govt:    Cohere 0.571 (+7.0% vs no-rewrite) 🏆
Cloud:   GT 0.451 (Cohere degrada -0.04%)
FiQA:    GT 0.442 (Cohere degrada -5.6%)
```

## 📝 Notas Finales

- El documento tiene **18 KB** de contenido riguroso
- Formato académico profesional (IEEE/ACM style)
- Listo para imprimir en papel A4
- Incluye tabla de contenidos automática
- Referencias cruzadas funcionan correctamente

## 🆘 Si Nada Funciona

**Opción más simple:**
1. Abre el archivo en **Google Docs** o **Word**
2. Copia el contenido de las tablas
3. Formatea manualmente
4. O usa directamente **resumen_ejecutivo_presentacion.md** que tiene las tablas en Markdown

---

**Para soporte adicional**, revisar:
- Overleaf Documentation: https://www.overleaf.com/learn
- LaTeX StackExchange: https://tex.stackexchange.com
