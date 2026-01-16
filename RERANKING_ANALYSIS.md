# Análisis de Resultados de Reranking

## 🔍 Problema Reportado
Los experimentos de reranking con Cohere y BGE mostraron incrementos prácticamente nulos en las métricas.

## ✅ Diagnóstico

### 1. **Cohere Reranker está funcionando, pero EMPEORA los resultados**

**Dominio: ClapNQ**
- Baseline (SPLADE + Voyage hybrid): nDCG@10 = **0.51721**
- Cohere rerank-v3.5: nDCG@10 = **0.48169**
- **Cambio: -6.87%** ❌

**Dominio: Govt**
```bash
# Verificar con:
python -c "
import json
b = json.load(open('experiments/hybrid_splade_voyage_rewrite/govt/metrics.json'))
c = json.load(open('experiments/rerank_cohere_splade_voyage_rewrite/govt/metrics.json'))
print(f'Baseline: {b[\"nDCG\"][0]:.5f}')
print(f'Cohere:   {c[\"nDCG\"][1]:.5f}')
print(f'Cambio:   {((c[\"nDCG\"][1]/b[\"nDCG\"][0])-1)*100:.2f}%')
"
```

### 2. **BGE Reranker SÍ mejora los resultados**

**Dominio: Cloud**
- Baseline (SPLADE + BGE-1.5 hybrid): nDCG@10 = **0.38820**
- BGE rerank-v2-m3: nDCG@10 = **0.44225**
- **Cambio: +13.92%** ✅

**Dominio: FiQA**
```bash
# Verificar con:
python -c "
import json
b = json.load(open('experiments/hybrid_splade_bge15_rewrite/fiqa/metrics.json'))
r = json.load(open('experiments/rerank_splade_bge15_rewrite/fiqa/metrics.json'))
print(f'Baseline: {b[\"nDCG\"][0]:.5f}')
print(f'BGE:      {r[\"nDCG\"][1]:.5f}')
print(f'Cambio:   {((r[\"nDCG\"][1]/b[\"nDCG\"][0])-1)*100:.2f}%')
"
```

## 🐛 Bug Encontrado: Diferencia en valores de k

Los experimentos con reranking usan **diferentes valores de k** que los baselines:

- **Baseline**: k = [10, 20, 50, 100]
- **Reranking**: k = [5, 10, 20, 50, 100, 1000]

Esto causó confusión al comparar resultados si no se alinean correctamente los índices:
- Baseline nDCG[0] = k@10
- Reranking nDCG[0] = k@5, nDCG[1] = k@10

## 🔬 Por qué Cohere EMPEORA los resultados

### Hipótesis principales:

#### 1. **Cohere rerank-v3.5 no está optimizado para conversaciones multi-turn**
   - El modelo puede no entender bien el contexto conversacional
   - Está diseñado para single-turn queries más tradicionales
   - Los queries condensados (R1) pueden perder información crítica que Cohere necesita

#### 2. **El baseline hybrid (SPLADE + Voyage) ya es muy fuerte**
   - nDCG@10 = 0.517 es un baseline excelente
   - SPLADE (sparse) captura matches lexicales precisos
   - Voyage-3 (dense) captura semántica
   - RRF fusion combina lo mejor de ambos
   - **Cohere puede estar "sobre-corrigiendo"** un ranking que ya era óptimo

#### 3. **Problema de calibración de scores**
Revisando el código en [cohere_rerank.py](src/pipeline/reranking/cohere_rerank.py#L118):
```python
doc_copy["score"] = result.relevance_score  # Sobreescribe el score original
doc_copy["original_score"] = original_doc.get("score", 0.0)
doc_copy["rerank_score"] = result.relevance_score
```
- Cohere devuelve scores en escala diferente (0-1)
- Los scores originales de SPLADE/Voyage tienen rangos distintos
- La re-ordenación puede estar desbalanceada

#### 4. **Query rewriting puede confundir al reranker**
   - R1 condensa el historial conversacional en una sola query
   - Ejemplo: "¿Qué es OAuth?" → "OAuth authentication methods IBM Cloud"
   - El documento original podría ser más relevante para la pregunta sin reescribir
   - Cohere reranker evalúa contra la query reescrita, no la original

## 🎯 Recomendaciones

### Inmediatas (para entender el problema):

1. **Analizar casos específicos donde Cohere falla**
   ```bash
   python analyze_cohere_failure.py  # Ya existe
   ```
   
2. **Comparar con/sin query rewriting**
   - Correr Cohere reranking sin R1 (query original)
   - Ver si el problema es la condensación de queries

3. **Verificar top-k retrieval**
   - Cohere recibe top-100 documentos del baseline
   - ¿Son suficientes? ¿Deberíamos aumentar a top-200?

### Medio plazo (para mejorar resultados):

1. **Fine-tune BGE reranker** (ya en progreso con `pedrovo9/bge-reranker-v2-m3-multirag-finetuned`)
   - BGE ya muestra +13.92% de mejora sin fine-tuning
   - Con fine-tuning en datos conversacionales multi-turn, debería mejorar aún más

2. **Probar Cohere con diferentes configuraciones**
   ```yaml
   reranking:
     enabled: true
     reranker_type: cohere
     model_name: rerank-v4.0-pro  # Modelo más reciente
     top_k_candidates: 200  # Aumentar candidatos
     max_chunks_per_doc: 1
   ```

3. **Implementar reranking en dos etapas**
   - Stage 1: Retrieval top-200
   - Stage 2: BGE reranker → top-100
   - Stage 3: Cohere reranker → top-20 (para generación)

### Largo plazo (investigación):

1. **Evaluar otros rerankers**
   - Jina Reranker v2
   - Mixedbread AI reranker
   - Voyage reranker (cuando esté disponible)

2. **Entrenar reranker específico para multi-turn**
   - Usar los 4 dominios (ClapNQ, Cloud, FiQA, Govt)
   - Aprovechar negativos duros del baseline hybrid
   - Ratio positivo:negativo 1:3

## 📊 Tabla Resumen de Resultados

| Experimento | Dominio | Baseline nDCG@10 | Reranked nDCG@10 | Cambio |
|-------------|---------|------------------|------------------|--------|
| **Cohere rerank-v3.5** | ClapNQ | 0.51721 | 0.48169 | **-6.87%** ❌ |
| **Cohere rerank-v3.5** | Govt | ? | ? | ? |
| **BGE rerank-v2-m3** | Cloud | 0.38820 | 0.44225 | **+13.92%** ✅ |
| **BGE rerank-v2-m3** | FiQA | ? | ? | ? |

## 🔧 Scripts de Verificación

### Comparar todos los experimentos de reranking:
```bash
#!/bin/bash
# compare_all_reranking.sh

for domain in clapnq cloud fiqa govt; do
    echo "=== $domain ==="
    
    # Cohere
    if [ -f experiments/hybrid_splade_voyage_rewrite/$domain/metrics.json ]; then
        python -c "
import json
b = json.load(open('experiments/hybrid_splade_voyage_rewrite/$domain/metrics.json'))
c = json.load(open('experiments/rerank_cohere_splade_voyage_rewrite/$domain/metrics.json'))
print(f'Cohere: {b[\"nDCG\"][0]:.5f} → {c[\"nDCG\"][1]:.5f} ({((c[\"nDCG\"][1]/b[\"nDCG\"][0])-1)*100:+.2f}%)')
        " 2>/dev/null || echo "Cohere: N/A"
    fi
    
    # BGE
    if [ -f experiments/hybrid_splade_bge15_rewrite/$domain/metrics.json ]; then
        python -c "
import json
b = json.load(open('experiments/hybrid_splade_bge15_rewrite/$domain/metrics.json'))
r = json.load(open('experiments/rerank_splade_bge15_rewrite/$domain/metrics.json'))
print(f'BGE:    {b[\"nDCG\"][0]:.5f} → {r[\"nDCG\"][1]:.5f} ({((r[\"nDCG\"][1]/b[\"nDCG\"][0])-1)*100:+.2f}%)')
        " 2>/dev/null || echo "BGE: N/A"
    fi
    
    echo
done
```

### Analizar qué documentos cambian de ranking:
```python
#!/usr/bin/env python3
# analyze_reranking_changes.py

import json
from collections import defaultdict

def analyze_ranking_changes(baseline_file, reranked_file, domain):
    """Compare document rankings before and after reranking."""
    
    with open(baseline_file) as f:
        baseline = [json.loads(line) for line in f]
    with open(reranked_file) as f:
        reranked = [json.loads(line) for line in f]
    
    improvements = 0
    degradations = 0
    
    for b_query, r_query in zip(baseline, reranked):
        b_docs = {ctx["document_id"]: i for i, ctx in enumerate(b_query["contexts"][:10])}
        r_docs = {ctx["document_id"]: i for i, ctx in enumerate(r_query["contexts"][:10])}
        
        # Check top-10 overlap
        overlap = set(b_docs.keys()) & set(r_docs.keys())
        
        # Count position changes
        for doc_id in overlap:
            if r_docs[doc_id] < b_docs[doc_id]:
                improvements += 1
            elif r_docs[doc_id] > b_docs[doc_id]:
                degradations += 1
    
    print(f"{domain}:")
    print(f"  Documentos que mejoraron posición: {improvements}")
    print(f"  Documentos que empeoraron posición: {degradations}")
    print(f"  Ratio: {improvements/max(degradations,1):.2f}")
    print()

# Usage:
analyze_ranking_changes(
    'experiments/hybrid_splade_voyage_rewrite/clapnq/retrieval_results.jsonl',
    'experiments/rerank_cohere_splade_voyage_rewrite/clapnq/retrieval_results.jsonl',
    'Cohere ClapNQ'
)
```

## 🎓 Conclusión

**El reranking SÍ está funcionando, pero con resultados mixtos:**

1. ✅ **BGE reranker funciona bien**: +13.92% en Cloud (sin fine-tuning)
   - Esperable mejora mayor con el modelo fine-tuned de tu amigo

2. ❌ **Cohere reranker empeora**: -6.87% en ClapNQ
   - Puede ser inadecuado para queries conversacionales multi-turn
   - Baseline hybrid ya es muy fuerte (0.517 nDCG@10)

3. 🎯 **Siguiente paso**: Evaluar modelo fine-tuned `pedrovo9/bge-reranker-v2-m3-multirag-finetuned`
   - Ya integrado en el código
   - Entrenar con datos multi-turn debería superar estos resultados

4. 🔬 **Investigación adicional necesaria**:
   - ¿Por qué Cohere falla con queries condensadas?
   - ¿Funcionaría mejor con query original?
   - ¿Hay over-fitting del baseline hybrid a los queries reescritos?
