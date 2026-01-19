# 📋 Análisis de Scripts Python - Limpieza y Consolidación

## 🔍 Scripts Encontrados

### En Raíz (7 archivos)
1. **extract_test_queries.py** (3.5K) - Extrae queries del test set
2. **generate_submission.py** (3.1K) - Genera archivo de submission
3. **migrate_to_split.py** (0 bytes) - ⚠️ VACÍO - ELIMINAR
4. **run_simple_test_retrieval.py** (7.9K) - Test simple de retrieval
5. **run_test_submission.py** (10K) - Ejecuta test oficial
6. **run_vllm.py** (1.8K) - Wrapper para vLLM
7. **test_finetuned_integration.py** (5.8K) - Test de modelo fine-tuned

### En scripts/ (9 archivos)
1. **build_indices.py** (9.6K) - Construye índices
2. **build_retrieval_tasks.py** (3.0K) - Construye tasks
3. **convert_hf_data.py** (3.9K) - Convierte datos HF
4. **hf_sync.py** (13K) - Sincroniza con HF
5. **make_submission.py** (5.5K) - Crea submissions
6. **prepare_hf_tasks.py** (1.0K) - Prepara tasks HF
7. **run_experiment.py** (0 bytes) - ⚠️ VACÍO - ELIMINAR
8. **summarize_metrics.py** (0 bytes) - ⚠️ VACÍO - ELIMINAR
9. **training/** (3 archivos para fine-tuning)

## 🧹 Acciones Recomendadas

### ❌ ELIMINAR (archivos vacíos)
- `migrate_to_split.py` (0 bytes)
- `scripts/run_experiment.py` (0 bytes)
- `scripts/summarize_metrics.py` (0 bytes)

### 🔄 CONSOLIDAR (funcionalidad duplicada)
**Problema**: Múltiples scripts para submissions
- `generate_submission.py` (raíz)
- `scripts/make_submission.py` (scripts/)

**Solución**: Mantener solo `scripts/make_submission.py` y eliminar `generate_submission.py`

### 📦 MOVER A scripts/ (mejor organización)
- `extract_test_queries.py` → `scripts/extract_test_queries.py`
- `run_simple_test_retrieval.py` → `scripts/run_simple_test_retrieval.py`
- `run_test_submission.py` → `scripts/run_test_submission.py`
- `run_vllm.py` → `scripts/run_vllm.py`
- `test_finetuned_integration.py` → `scripts/test_finetuned_integration.py`

### ✅ MANTENER (archivos importantes en src/)
- Todos los archivos en `src/pipeline/` (código core)
- Todos los archivos en `src/utils/` (utilidades)
- `preprocessing/build_processed_corpus.py` (necesario)

## 📊 Resultado Final

**Antes**: 45 archivos .py dispersos
**Después**: ~40 archivos .py organizados

### Estructura Limpia:
```
task_a_retrieval/
├── src/
│   ├── pipeline/      # Código core (retrieval, reranking, etc)
│   └── utils/         # Utilidades compartidas
├── scripts/           # Scripts de utilidad (15 archivos)
│   ├── build_indices.py
│   ├── extract_test_queries.py (movido)
│   ├── make_submission.py
│   ├── run_test_submission.py (movido)
│   └── training/      # Scripts de fine-tuning
└── preprocessing/     # Procesamiento de datos
    └── build_processed_corpus.py
```

## 🐳 Docker para Reproducibilidad

Crear Dockerfile con:
- Python 3.10
- Todas las dependencias
- CUDA support
- Cache de modelos
- Scripts organizados
