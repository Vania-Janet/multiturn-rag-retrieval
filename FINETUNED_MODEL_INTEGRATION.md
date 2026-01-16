# Integración del Modelo Fine-tuned BGE Reranker

## Resumen

Se ha integrado exitosamente el modelo fine-tuned `pedrovo9/bge-reranker-v2-m3-multirag-finetuned` en el pipeline de experimentos de MT-RAG Benchmark.

## Modelo

**Nombre**: `pedrovo9/bge-reranker-v2-m3-multirag-finetuned`  
**Repositorio**: https://huggingface.co/pedrovo9/bge-reranker-v2-m3-multirag-finetuned

### Detalles del Modelo

- **Base**: BAAI/bge-reranker-v2-m3
- **Estrategia de entrenamiento**: Pairwise learning (ratio 1:2 positivos:negativos)
- **Hard negatives**: Generados con BM25
- **Épocas**: 3
- **Dominios**: ClapNQ, Cloud, FiQA, Govt

### Data Splits (Sin Data Leakage)

El repositorio del modelo incluye splits adecuados en la carpeta `data/`:
- `train.jsonl` - Datos de entrenamiento
- `test.jsonl` - Datos de prueba
- `val.jsonl` - Datos de validación

Estos splits fueron creados a nivel de conversación para prevenir data leakage.

## Archivos Creados

### 1. Clase de Reranker Fine-tuned

**Archivo**: `src/pipeline/reranking/finetuned_bge_reranker.py`

Implementa la clase `FineTunedBGEReranker` que:
- Carga el modelo desde Hugging Face usando transformers
- Soporta FP16 para inferencia rápida en GPU
- Procesa documentos en batches para eficiencia
- Calcula scores usando sigmoid de los logits del modelo

**Uso en código**:
```python
from pipeline.reranking import FineTunedBGEReranker

reranker = FineTunedBGEReranker(
    model_name="pedrovo9/bge-reranker-v2-m3-multirag-finetuned",
    config={"batch_size": 32, "use_fp16": True}
)

reranked_docs = reranker.rerank(query, documents, top_k=100)
```

### 2. Configuraciones de Experimentos

Actualizadas 3 configuraciones en `configs/experiments/05-finetune/`:

#### A10_finetuned_reranker.yaml
- Hybrid retrieval: SPLADE + BGE-M3
- Query rewriting: vLLM (Qwen2.5-7B)
- Reranking: Fine-tuned BGE

#### finetune_bge_splade_bge15_rewrite.yaml
- Hybrid retrieval: SPLADE + BGE-base-en-v1.5
- Query rewriting: vLLM (Qwen2.5-7B)
- Reranking: Fine-tuned BGE

#### finetune_bge_splade_voyage_rewrite.yaml
- Hybrid retrieval: SPLADE + Voyage-3-large
- Query rewriting: vLLM (Qwen2.5-7B)
- Reranking: Fine-tuned BGE

**Configuración estándar de reranking**:
```yaml
reranking:
  enabled: true
  reranker_type: "finetuned_bge"
  model_name: "pedrovo9/bge-reranker-v2-m3-multirag-finetuned"
  top_k: 100
  batch_size: 32
  use_fp16: true
```

### 3. Script de Ejecución

**Archivo**: `run_finetuned_experiments.sh`

Script bash para ejecutar todos los experimentos de fine-tuning:
- Verifica acceso al modelo en Hugging Face
- Ejecuta 3 experimentos × 4 dominios = 12 runs
- Logs detallados para cada ejecución
- Agrega resultados al final

**Uso**:
```bash
./run_finetuned_experiments.sh
```

### 4. README Actualizado

**Archivo**: `configs/experiments/05-finetune/README.md`

Documentación completa de:
- Detalles del modelo fine-tuned
- Instrucciones de uso
- Comandos para ejecutar experimentos
- Resultados esperados

## Cambios en el Código Base

### 1. src/pipeline/reranking/__init__.py

Agregado import y export de `FineTunedBGEReranker`:
```python
from .finetuned_bge_reranker import FineTunedBGEReranker

__all__ = [
    # ... otros
    "FineTunedBGEReranker",
]
```

### 2. src/pipeline/run.py

Agregado soporte para el reranker fine-tuned:
```python
from .reranking import CohereReranker, BGEReranker, FineTunedBGEReranker

# En la inicialización del reranker:
elif reranker_type == "finetuned_bge":
    reranker = FineTunedBGEReranker(
        model_name=config["reranking"].get("model_name", "pedrovo9/bge-reranker-v2-m3-multirag-finetuned"),
        config=config["reranking"]
    )
```

También se actualizó para soportar `reranker_type` además de `type` en la configuración.

## Cómo Ejecutar

### Opción 1: Todos los Experimentos

```bash
cd /workspace/mt-rag-benchmark/task_a_retrieval
./run_finetuned_experiments.sh
```

Esto ejecutará:
- 3 experimentos (A10, finetune_bge_splade_bge15, finetune_bge_splade_voyage)
- 4 dominios cada uno (clapnq, cloud, fiqa, govt)
- Total: 12 ejecuciones

### Opción 2: Experimento Individual

```bash
# A10 con hybrid SPLADE + BGE-M3
python scripts/run_experiment.py \
    --experiment A10_finetuned_reranker \
    --domain clapnq \
    --experiment-dir configs/experiments/05-finetune

# Con SPLADE + BGE-1.5
python scripts/run_experiment.py \
    --experiment finetune_bge_splade_bge15_rewrite \
    --domain cloud \
    --experiment-dir configs/experiments/05-finetune

# Con SPLADE + Voyage-3
python scripts/run_experiment.py \
    --experiment finetune_bge_splade_voyage_rewrite \
    --domain fiqa \
    --experiment-dir configs/experiments/05-finetune
```

### Opción 3: Dominio Específico

```bash
# Solo ClapNQ
for exp in A10_finetuned_reranker finetune_bge_splade_bge15_rewrite finetune_bge_splade_voyage_rewrite; do
    python scripts/run_experiment.py -e $exp -d clapnq --experiment-dir configs/experiments/05-finetune
done
```

## Resultados

Los resultados se guardarán en:
```
experiments/05-finetune/
├── A10_finetuned_reranker/
│   ├── clapnq/
│   │   ├── metrics.json
│   │   ├── retrieval_results.jsonl
│   │   └── analysis_report.json
│   ├── cloud/
│   ├── fiqa/
│   └── govt/
├── finetune_bge_splade_bge15_rewrite/
└── finetune_bge_splade_voyage_rewrite/
```

### Métricas Esperadas

Basándose en los baselines actuales:

| Experimento | Baseline | Esperado con Fine-tuning |
|-------------|----------|--------------------------|
| SPLADE solo | nDCG@10: 0.44 | - |
| Hybrid (SPLADE + Voyage) | nDCG@10: 0.48-0.52 | nDCG@10: 0.50-0.55 |
| + Query Rewriting | nDCG@10: +2-5% | nDCG@10: +2-5% |
| **+ Fine-tuned Reranking** | - | **nDCG@10: 0.52-0.58** |

**Ventajas esperadas**:
1. Mejor discriminación en documentos difíciles (hard negatives)
2. Mejor manejo de queries multi-turno conversacionales
3. Mejor adaptación a terminología específica de dominio
4. Mejor ranking en los top-10 documentos

## Verificación Rápida

Para verificar que todo está configurado correctamente:

```bash
cd /workspace/mt-rag-benchmark/task_a_retrieval

# Test 1: Verificar que el modelo es accesible
python3 -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('pedrovo9/bge-reranker-v2-m3-multirag-finetuned')
print('✓ Model accessible')
"

# Test 2: Verificar que la clase de reranker funciona
python3 -c "
from src.pipeline.reranking import FineTunedBGEReranker
reranker = FineTunedBGEReranker()
print('✓ FineTunedBGEReranker initialized')
"

# Test 3: Verificar configuraciones
for exp in A10_finetuned_reranker finetune_bge_splade_bge15_rewrite finetune_bge_splade_voyage_rewrite; do
    echo "Checking $exp..."
    cat configs/experiments/05-finetune/$exp.yaml | grep -A5 reranking
done
```

## Notas Importantes

1. **Dependencias**: Asegúrate de tener `transformers` instalado:
   ```bash
   pip install transformers torch
   ```

2. **GPU**: El modelo usará GPU automáticamente si está disponible. Con FP16 en RTX 4090:
   - Memoria GPU: ~2-3 GB
   - Velocidad: ~50-100 queries/segundo para reranking de 100 docs

3. **Acceso a Hugging Face**: La primera vez que se ejecute, el modelo se descargará (~600MB). Asegúrate de tener:
   - Conexión a internet
   - Suficiente espacio en disco
   - Token de Hugging Face si el modelo requiere autenticación (actualmente es público)

4. **Comparación con Baselines**: Para una evaluación justa, compara con:
   - `hybrid_splade_voyage_rewrite` (mejor baseline actual)
   - `rerank_cohere_splade_voyage_rewrite` (reranking con Cohere)

## Próximos Pasos

1. **Ejecutar experimentos**: Correr `./run_finetuned_experiments.sh`
2. **Analizar resultados**: Comparar métricas con baselines
3. **Documentar hallazgos**: Actualizar `docs/experiment_table.md` con resultados
4. **Paper**: Incluir análisis del modelo fine-tuned en la sección de resultados

## Contacto

- Modelo fine-tuned por: pedrovo9
- Integración: Este proyecto
- Issues: GitHub repository
