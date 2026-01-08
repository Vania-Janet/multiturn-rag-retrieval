# Fixes para Experimentos 01-Query

## Resumen
Se identificaron y corrigieron **2 problemas críticos** que causaban que los experimentos 01-query tuvieran métricas mucho más bajas de lo esperado en comparación con los baselines.

---

## Problema 1: Prompts Genéricos No Optimizados por Tipo de Retrieval

### Descripción
El `VLLMRewriter` usaba el mismo prompt para todos los modelos de retrieval (BM25, SPLADE, Voyage, BGE-M3), sin diferenciar entre:
- **Sparse retrieval** (BM25, SPLADE): Requiere keywords específicos, términos técnicos, y variación léxica
- **Dense retrieval** (Voyage, BGE-M3): Requiere preservación semántica y contexto natural

### Impacto
- BM25 y SPLADE no recibían queries con suficiente diversidad de keywords
- Las queries de multi-query no estaban optimizadas para cada tipo de búsqueda
- El prompt de condensación era genérico y no maximizaba el rendimiento

### Solución
**Archivo modificado:** `src/pipeline/query_transform/rewriters.py`

1. **Creados 4 prompts especializados:**
   - `CONDENSATION_PROMPT_SPARSE`: Para BM25/SPLADE
     - Enfatiza preservación de keywords técnicos
     - Instrucciones para expandir acrónimos cuando ayude al matching
     - Incluir sinónimos y términos relacionados del contexto
   
   - `CONDENSATION_PROMPT_DENSE`: Para Voyage/BGE-M3
     - Enfoque en preservación semántica
     - Menos énfasis en keywords adicionales
     - Natural language flow
   
   - `MULTIQUERY_PROMPT_SPARSE`: Para BM25/SPLADE multi-query
     - Enfatiza diversidad léxica entre queries
     - Variación de acrónimos vs nombres completos
     - Diferentes términos técnicos para cada query
   
   - `MULTIQUERY_PROMPT_DENSE`: Para Voyage/BGE-M3 multi-query
     - Enfatiza diversidad semántica
     - Diferentes ángulos/perspectivas de la pregunta
     - Natural language variations

2. **Añadido parámetro `retrieval_type`:**
   - El VLLMRewriter ahora acepta `retrieval_type: "sparse"` o `"dense"`
   - Selecciona automáticamente el prompt apropiado según el tipo y max_rewrites

3. **Actualización de cache keys:**
   - El cache ahora incluye `retrieval_type` para evitar colisiones entre sparse/dense

### Cambios en Configuraciones
**Archivos modificados:** Todos los configs en `configs/experiments/01-query/*.yaml`

```yaml
# Antes
rewriter_config:
  model_name: Qwen/Qwen2.5-7B-Instruct
  temperature: 0
  max_rewrites: 1

# Después  
rewriter_config:
  model_name: Qwen/Qwen2.5-7B-Instruct
  temperature: 0
  max_rewrites: 1
  retrieval_type: sparse  # o "dense" según el modelo
```

---

## Problema 2: Archivo de Queries Incorrecto (BUG CRÍTICO)

### Descripción
Los experimentos 01-query estaban usando `{domain}_questions.jsonl` en lugar de `{domain}_tasks.jsonl`:

- **`{domain}_questions.jsonl`**: Formato con campo `"text"` que contiene el historial ya concatenado
  ```json
  {"_id": "...", "text": "|user|: turn1\n|user|: turn2\n|user|: turn3"}
  ```
  
- **`{domain}_tasks.jsonl`**: Formato estructurado con campo `"input"` que contiene array de turnos
  ```json
  {
    "task_id": "...",
    "input": [
      {"speaker": "user", "text": "turn1"},
      {"speaker": "agent", "text": "response1"},
      {"speaker": "user", "text": "turn2"}
    ]
  }
  ```

### Impacto
**CRÍTICO**: El query rewriter **NUNCA se ejecutaba** porque:
1. El código en `src/pipeline/run.py` detecta el campo `"text"` primero
2. Si existe `"text"`, usa ese valor directamente sin procesar el rewriter
3. El rewriter solo se ejecuta cuando hay campo `"input"` estructurado
4. Como resultado, las queries iban con el historial completo concatenado, igual que los baselines

**Esto explica por qué:**
- Voyage R1 condensation tenía métricas de ~0.0 (casi nulos)
- Todos los experimentos 01-query tenían resultados similares o peores que baselines
- El vLLM nunca se inicializaba correctamente

### Solución
**Archivos modificados:** Todos los configs en `configs/experiments/01-query/*.yaml`

```yaml
# Antes (usaba el default del domain config)
data:
  query_mode: last_turn

# Después (override explícito)
data:
  query_mode: last_turn
  query_file: "data/retrieval_tasks/{domain}/{domain}_tasks.jsonl"
```

**Archivo modificado:** `scripts/run_experiment.py`

Añadida función de sustitución de `{domain}` placeholder:

```python
def substitute_domain(obj, domain_name):
    """Recursively substitute {domain} in strings."""
    if isinstance(obj, dict):
        return {k: substitute_domain(v, domain_name) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [substitute_domain(item, domain_name) for item in obj]
    elif isinstance(obj, str):
        return obj.replace("{domain}", domain_name)
    else:
        return obj

config = substitute_domain(config, domain)
```

---

## Verificación de los Fixes

### Test Manual - Voyage R1 Condensation
```bash
cd /workspace/mt-rag-benchmark/task_a_retrieval

# Verificar que los documentos se recuperan correctamente
python3 recalc_metrics.py
```

**Resultados esperados DESPUÉS del fix:**
- Recall@10: ~0.40 (vs 0.00 antes)
- Recall@100: ~0.76 (vs 0.02 antes)
- El rewriter ahora se ejecuta y condensa las queries correctamente

---

## Próximos Pasos

1. **Re-ejecutar todos los experimentos 01-query:**
   ```bash
   ./run_all_01query_experiments.sh
   ```

2. **Verificar logs** para confirmar que:
   - vLLM se inicializa correctamente
   - Los prompts específicos se seleccionan
   - Las queries se reescriben (verificar en retrieval_results.jsonl)

3. **Comparar métricas** con baselines:
   - Los experimentos de query rewriting ahora deberían superar a los baselines
   - BM25 R2 multi debería tener mejor recall que BM25 baseline
   - Voyage R1 condensation debería tener métricas comparables o mejores que baseline

4. **Revisar otras carpetas de experimentos:**
   - Verificar que 02-hybrid, 03-rerank también usen el formato correcto

---

## Archivos Modificados

### Código
- `src/pipeline/query_transform/rewriters.py`: Prompts especializados y retrieval_type
- `scripts/run_experiment.py`: Sustitución de {domain} placeholder

### Configuraciones (8 archivos)
- `configs/experiments/01-query/bm25_r1_condensation.yaml`
- `configs/experiments/01-query/bm25_r2_multi.yaml`
- `configs/experiments/01-query/splade_r1_condensation.yaml`
- `configs/experiments/01-query/splade_r3_hyde.yaml`
- `configs/experiments/01-query/bgem3_r1_condensation.yaml`
- `configs/experiments/01-query/bgem3_r2_multi.yaml`
- `configs/experiments/01-query/voyage_r1_condensation.yaml`
- `configs/experiments/01-query/voyage_r2_multi.yaml`

---

## Notas Técnicas

### Por qué las métricas estaban en ~0.0
Las métricas guardadas en los archivos `metrics.json` eran incorrectas porque las queries nunca fueron procesadas por el rewriter. El sistema recuperaba documentos usando el historial completo concatenado (similar a fullhist baseline), pero al evaluar contra qrels que esperan queries condensadas, el matching era muy pobre.

### Diferencia entre query modes
- **last_turn**: Solo el último turno del usuario
- **full_history**: Todos los turnos del usuario concatenados
- **full_context**: Todos los turnos (usuario + agente) concatenados

Los experimentos 01-query deberían usar `last_turn` con el rewriter para condensar el contexto.
