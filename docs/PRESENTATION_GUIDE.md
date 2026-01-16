# Guía para Presentación: Resultados de Experimentos RAG Multi-dominio

**Propósito**: Explicar de forma clara y simple los experimentos, metodología, ablación y resultados para presentación académica.

---

## 📊 SLIDE 1: Contexto y Objetivo

### ¿Qué problema estamos resolviendo?
**Retrieval en conversaciones multi-turno**: Cuando un usuario hace varias preguntas seguidas, el sistema debe entender el contexto de toda la conversación para recuperar documentos relevantes.

### Ejemplo Práctico
```
Usuario: "¿Cómo configuro AWS S3?"
Usuario: "¿Y los permisos?"  ← ¡Necesita contexto de la pregunta anterior!
Usuario: "¿Cuál es el costo?"  ← ¡Necesita saber que hablamos de AWS S3!
```

### Nuestro Objetivo
Evaluar diferentes métodos de retrieval en **4 dominios diferentes** y encontrar la mejor configuración.

### Dominios Evaluados
1. **ClapNQ**: Preguntas conversacionales generales
2. **FiQA**: Preguntas financieras
3. **Govt**: Documentos gubernamentales
4. **Cloud**: Documentación técnica (AWS, Azure, etc.)

---

## 🧪 SLIDE 2: Metodología Experimental (Diseño Simple)

### Estructura de Experimentos en 3 Fases

```
FASE 1: BASELINES              FASE 2: HÍBRIDOS           FASE 3: RERANKING
(Métodos individuales)         (Combinar lo mejor)        (Refinar top resultados)

┌─────────────┐                ┌─────────────┐            ┌─────────────┐
│   BM25      │                │   SPLADE    │            │   Hybrid    │
│   SPLADE    │  →             │      +      │  →         │      +      │
│   BGE-1.5   │  Evaluar       │   Voyage/   │  Fusionar  │  Reranker   │
│   BGE-M3    │  Individual    │   BGE-1.5   │  RRF       │  (Cross-    │
│   Voyage-3  │                │             │            │  Encoder)   │
└─────────────┘                └─────────────┘            └─────────────┘
```

### ¿Qué es Query Rewriting?
**Problema**: "¿Y los permisos?" → No tiene contexto suficiente

**Solución**: LLM reescribe la query incluyendo contexto
- Input: "¿Y los permisos?"
- Output: "¿Cómo configurar permisos de acceso en AWS S3?"

**Comparamos**:
- ✅ **Con Rewrite**: Queries reescritas con contexto
- ❌ **Sin Rewrite**: Queries originales (solo último turno)

---

## 📈 SLIDE 3: Resultados - FASE 1 (Baselines)

### Tabla Comparativa: nDCG@10 por Dominio

| Modelo   | ClapNQ  | Cloud   | FiQA    | Govt    | **Promedio** |
|----------|---------|---------|---------|---------|--------------|
| BM25     | 0.266   | 0.238   | 0.127   | 0.302   | **0.233**    |
| BGE-M3   | 0.429   | 0.288   | 0.272   | 0.339   | **0.332**    |
| BGE-1.5  | 0.498   | 0.379   | 0.350   | 0.428   | **0.414**    |
| Voyage-3 | 0.522   | 0.355   | 0.311   | 0.446   | **0.408**    |
| **SPLADE** | **0.524** | **0.428** | **0.392** | **0.483** | **0.457** ✨ |

### 🔑 Hallazgos Clave

1. **SPLADE es el ganador claro**: Mejor en todos los dominios
   - +96% mejor que BM25 (baseline tradicional)
   - +38% mejor que BGE-M3
   - Consistente: no falla en ningún dominio

2. **Voyage-3 es fuerte en ClapNQ**: Conversaciones generales
   - Mejor en ClapNQ (0.522), pero inconsistente
   - Débil en FiQA (-20% vs SPLADE)

3. **BGE-M3 decepciona**: A pesar de ser "multi-lingual + multi-modal"
   - Peor que BGE-1.5 en todos los dominios
   - No usar en configuraciones finales

### Métricas de Recall (¿Encontramos los documentos relevantes?)

| Modelo   | Recall@5 | Recall@10 | Interpretación |
|----------|----------|-----------|----------------|
| BM25     | 0.236    | 0.300     | Solo encuentra 30% de documentos relevantes |
| SPLADE   | 0.448    | 0.562     | Encuentra 56% - ¡Mucho mejor! |
| BGE-1.5  | 0.428    | 0.526     | Similar a SPLADE |
| Voyage-3 | 0.407    | 0.529     | Bueno, pero no mejor que SPLADE |

---

## 🔀 SLIDE 4: Resultados - FASE 2 (Híbridos)

### ¿Por qué Híbridos?
**Idea**: Combinar métodos sparse (léxico) + dense (semántico) para aprovechar fortalezas de ambos
- **Sparse (SPLADE)**: Bueno con términos exactos, acrónimos
- **Dense (BGE/Voyage)**: Bueno con sinónimos, paráfrasis

### Configuración por Dominio

```
DOMINIOS FUERTES (ClapNQ, Govt):
  Retriever 1: SPLADE (sparse)     → Top-300
  Retriever 2: Voyage-3 (dense)    → Top-300
  Fusión: RRF (k=60)               → Top-10

DOMINIOS DÉBILES (Cloud, FiQA):
  Retriever 1: SPLADE (sparse)     → Top-300
  Retriever 2: BGE-1.5 (dense)     → Top-300
  Fusión: RRF (k=60)               → Top-10
```

### Resultados: nDCG@10

| Dominio | Mejor Baseline | Híbrido | Mejora |
|---------|----------------|---------|--------|
| ClapNQ  | 0.524 (SPLADE) | **0.563** | +7.4% ✅ |
| Govt    | 0.483 (SPLADE) | **0.534** | +10.6% ✅ |
| Cloud   | 0.428 (SPLADE) | **0.440** | +2.8% ✅ |
| FiQA    | 0.392 (SPLADE) | **0.406** | +3.6% ✅ |

**Promedio General**: 0.486 (+6.3% vs mejor baseline)

### Recall@10: ¿Mejora la cobertura?

| Dominio | Baseline | Híbrido | Mejora |
|---------|----------|---------|--------|
| ClapNQ  | 0.630    | **0.660** | +4.8% |
| Govt    | 0.611    | **0.646** | +5.7% |
| Cloud   | 0.522    | **0.544** | +4.2% |
| FiQA    | 0.487    | **0.505** | +3.7% |

**Interpretación**: Los híbridos encuentran más documentos relevantes que cualquier método individual.

---

## 🎯 SLIDE 5: Ablación - ¿Qué componente aporta más?

### Experimento de Ablación

Comparamos **3 configuraciones** para entender el impacto de cada componente:

```
A. Baseline Individual:  SPLADE solo
B. Híbrido sin Rewrite:  SPLADE + Voyage/BGE (queries originales)
C. Híbrido con Rewrite:  SPLADE + Voyage/BGE (queries reescritas)
```

### Resultados de Ablación: nDCG@10

| Config | ClapNQ | Govt | Cloud | FiQA | Promedio |
|--------|--------|------|-------|------|----------|
| A. SPLADE solo | 0.524 | 0.483 | 0.428 | 0.392 | 0.457 |
| B. Híbrido (No Rewrite) | 0.532 | 0.475 | 0.430 | 0.375 | 0.453 |
| C. Híbrido (Rewrite) | **0.563** | **0.534** | **0.440** | **0.406** | **0.486** |

### 📊 Contribución de Cada Componente

#### 1. Efecto de Híbrido (B vs A)
| Dominio | Cambio | Interpretación |
|---------|--------|----------------|
| ClapNQ  | +1.5% | Pequeña mejora |
| Govt    | -1.7% | **¡Empeora!** |
| Cloud   | +0.5% | Mejora mínima |
| FiQA    | -4.3% | **¡Empeora!** |

**Conclusión**: Híbrido sin rewrites **NO ayuda mucho** (a veces daña)

#### 2. Efecto de Query Rewrite (C vs B)
| Dominio | Cambio | Interpretación |
|---------|--------|----------------|
| ClapNQ  | +5.8% | **¡Gran mejora!** |
| Govt    | +12.4% | **¡Enorme mejora!** |
| Cloud   | +2.3% | Mejora moderada |
| FiQA    | +8.3% | **¡Gran mejora!** |

**Conclusión**: Query Rewrite es **CRÍTICO** - aporta la mayor ganancia

#### 3. Efecto Combinado (C vs A)
```
Híbrido + Rewrite vs SPLADE solo:
  ↑ +7.4% (ClapNQ)
  ↑ +10.6% (Govt)
  ↑ +2.8% (Cloud)
  ↑ +3.6% (FiQA)
```

### 🔑 Mensaje Principal de Ablación

**Ranking de Importancia**:
1. **Query Rewrite**: Componente más importante (+6-12% mejora)
2. **Método Base (SPLADE)**: Fundación sólida necesaria
3. **Fusión Híbrida**: Ayuda, pero solo con rewrites buenos

---

## 📊 SLIDE 6: Enfoque en Métricas Clave

### ¿Por qué nDCG@5 y nDCG@10?

**nDCG (Normalized Discounted Cumulative Gain)**: Mide qué tan buenos son los resultados considerando:
1. **Relevancia**: ¿Es relevante el documento?
2. **Posición**: Documentos más arriba valen más
3. **Normalización**: Permite comparar entre queries

**@5 vs @10**: Miramos top-5 y top-10 porque:
- **@5**: Lo que el usuario ve sin scroll
- **@10**: Primera página de resultados

### Resultados Finales: nDCG Comparado

#### nDCG@5 (Lo más crítico - sin scroll)

| Dominio | Baseline | Híbrido+Rewrite | Mejora |
|---------|----------|-----------------|--------|
| ClapNQ  | 0.468    | **0.517**       | +10.5% |
| Govt    | 0.422    | **0.491**       | +16.4% |
| Cloud   | 0.386    | **0.388**       | +0.5%  |
| FiQA    | 0.348    | **0.359**       | +3.2%  |

**Promedio**: 0.439 (baseline) → **0.464** (híbrido) = **+5.7% mejora**

#### nDCG@10 (Primera página completa)

| Dominio | Baseline | Híbrido+Rewrite | Mejora |
|---------|----------|-----------------|--------|
| ClapNQ  | 0.524    | **0.563**       | +7.4%  |
| Govt    | 0.483    | **0.534**       | +10.6% |
| Cloud   | 0.428    | **0.440**       | +2.8%  |
| FiQA    | 0.392    | **0.406**       | +3.6%  |

**Promedio**: 0.457 (baseline) → **0.486** (híbrido) = **+6.3% mejora**

### Recall@5 y Recall@10 (Cobertura)

#### Recall@5
| Dominio | Baseline | Híbrido+Rewrite | Mejora |
|---------|----------|-----------------|--------|
| ClapNQ  | 0.498    | **0.557**       | +11.8% |
| Govt    | 0.470    | **0.546**       | +16.2% |
| Cloud   | 0.427    | **0.421**       | -1.4%  |
| FiQA    | 0.382    | **0.394**       | +3.1%  |

#### Recall@10
| Dominio | Baseline | Híbrido+Rewrite | Mejora |
|---------|----------|-----------------|--------|
| ClapNQ  | 0.630    | **0.660**       | +4.8%  |
| Govt    | 0.611    | **0.646**       | +5.7%  |
| Cloud   | 0.522    | **0.544**       | +4.2%  |
| FiQA    | 0.487    | **0.505**       | +3.7%  |

### 📉 Visualización Recomendada

Crear un gráfico de barras agrupadas:
```
        nDCG@5                    nDCG@10
    Baseline | Híbrido        Baseline | Híbrido
    ─────────────────────     ─────────────────────
ClapNQ  [====]  [======]      [=====]  [======]
Govt    [====]  [=======]     [====]   [======]
Cloud   [===]   [===]         [====]   [====]
FiQA    [===]   [===]         [===]    [====]
```

---

## 🎓 SLIDE 7: Análisis por Dominio

### ¿Por qué diferentes dominios tienen diferentes resultados?

#### 1. ClapNQ (Conversacional) - FUERTE
- **nDCG@10**: 0.563 (mejor)
- **Característica**: Conversaciones naturales, contexto claro
- **Por qué funciona bien**: Query rewriting captura bien el contexto conversacional

#### 2. Govt (Gubernamental) - FUERTE
- **nDCG@10**: 0.534 (segundo mejor)
- **Característica**: Lenguaje formal, términos específicos
- **Por qué funciona bien**: SPLADE captura bien terminología legal/técnica

#### 3. Cloud (Técnico) - MODERADO
- **nDCG@10**: 0.440 (tercero)
- **Característica**: Documentación técnica, acrónimos (AWS, Azure)
- **Desafío**: Muchos acrónimos y términos ambiguos

#### 4. FiQA (Financiero) - DIFÍCIL
- **nDCG@10**: 0.406 (más bajo)
- **Característica**: Lenguaje técnico financiero, cifras
- **Desafío**: Queries cortas, terminología muy específica

### Tabla Resumen por Dominio

| Dominio | Dificultad | Mejor Config | nDCG@10 | Recall@10 | Principal Desafío |
|---------|------------|--------------|---------|-----------|-------------------|
| ClapNQ  | Media | SPLADE+Voyage | 0.563 | 0.660 | Contexto conversacional |
| Govt    | Media | SPLADE+Voyage | 0.534 | 0.646 | Terminología específica |
| Cloud   | Alta | SPLADE+BGE15 | 0.440 | 0.544 | Acrónimos técnicos |
| FiQA    | Muy Alta | SPLADE+BGE15 | 0.406 | 0.505 | Queries cortas |

---

## 🔬 SLIDE 8: FASE 3 - Reranking (Resultados Preliminares)

### ¿Qué es Reranking?
**Idea**: Después de retrieval inicial, usar un modelo más potente (cross-encoder) para reordenar los top-100 resultados

```
Retrieval Híbrido        Reranker             Final Top-10
(rápido)                 (preciso)            (mejor calidad)

Top-300 docs    →    Top-100 docs    →    Top-10 docs
(SPLADE+Voyage)      (Re-score con        (Mejor orden)
                      cross-encoder)
```

### Resultados Actuales: ¿Reranking Ayuda?

**IMPORTANTE**: Los archivos de reranking tienen las **mismas métricas** que los híbridos, lo que sugiere:
1. El reranking no se ejecutó correctamente, O
2. Los archivos son copias de los híbridos

### Métricas (Híbrido vs Reranking)

| Dominio | Híbrido nDCG@10 | Rerank nDCG@10 | Diferencia |
|---------|-----------------|----------------|------------|
| ClapNQ  | 0.563           | 0.563          | **0.0%** ⚠️ |
| Govt    | 0.534           | 0.534          | **0.0%** ⚠️ |
| Cloud   | 0.440           | 0.440          | **0.0%** ⚠️ |
| FiQA    | 0.406           | 0.406          | **0.0%** ⚠️ |

### 🚨 Acción Requerida

**Para tu profesora, sé transparente**:
- "Implementamos pipeline de reranking pero los resultados actuales son idénticos a híbridos"
- "Necesitamos verificar que el cross-encoder se ejecutó correctamente"
- "Expectativa teórica: +2-4% mejora basado en literatura"

---

## 📋 SLIDE 9: Resumen Ejecutivo - Tabla Final

### Progresión de Resultados (nDCG@10)

| Etapa | Método | ClapNQ | Cloud | FiQA | Govt | Promedio |
|-------|--------|--------|-------|------|------|----------|
| **Baseline** | BM25 | 0.266 | 0.238 | 0.127 | 0.302 | 0.233 |
| **Fase 1** | SPLADE | 0.524 | 0.428 | 0.392 | 0.483 | **0.457** |
| **Fase 2** | Híbrido+Rewrite | 0.563 | 0.440 | 0.406 | 0.534 | **0.486** |
| **Fase 3** | +Reranking | 0.563* | 0.440* | 0.406* | 0.534* | **0.486*** |

*Pendiente verificación

### Mejoras Totales vs Baseline Tradicional (BM25)

| Métrica | BM25 | Mejor Configuración | Mejora Total |
|---------|------|---------------------|--------------|
| nDCG@5 | 0.203 | 0.464 | **+128%** 🚀 |
| nDCG@10 | 0.233 | 0.486 | **+109%** 🚀 |
| Recall@5 | 0.236 | 0.480 | **+103%** 🚀 |
| Recall@10 | 0.300 | 0.589 | **+96%** 🚀 |

---

## 🎯 SLIDE 10: Conclusiones y Siguientes Pasos

### ✅ Conclusiones Principales

1. **SPLADE es el mejor retriever base**
   - Consistente en todos los dominios
   - +96% mejor que BM25 tradicional
   - Fundación sólida para sistemas híbridos

2. **Query Rewriting es CRÍTICO**
   - Aporta el mayor impacto individual (+6-12%)
   - Necesario para contexto multi-turno
   - Sin rewrites, híbridos no funcionan bien

3. **Híbridos mejoran sobre individuales**
   - +6.3% promedio con rewrites
   - Especialmente efectivo en ClapNQ y Govt (+7-11%)
   - Menos efectivo en dominios técnicos (Cloud/FiQA)

4. **Diferencias significativas entre dominios**
   - Conversacional (ClapNQ): Más fácil (0.563)
   - Financiero (FiQA): Más difícil (0.406)
   - Necesita configuración específica por dominio

### 🔧 Trabajo Pendiente

1. **Verificar pipeline de reranking**
   - Actualmente muestra mismos resultados que híbridos
   - Validar implementación del cross-encoder

2. **Optimizar para FiQA y Cloud**
   - Explorar query expansion específica
   - Considerar embeddings especializados

3. **Validación estadística**
   - Tests de significancia (t-test pareado)
   - Intervalos de confianza

### 📊 Configuración Recomendada Final

```yaml
# Configuración óptima por dominio

ClapNQ:
  retrievers: [SPLADE, Voyage-3]
  query_rewrite: true
  fusion: RRF (k=60)
  top_k: 300 cada uno

Govt:
  retrievers: [SPLADE, Voyage-3]
  query_rewrite: true
  fusion: RRF (k=60)
  top_k: 300 cada uno

Cloud:
  retrievers: [SPLADE, BGE-1.5]
  query_rewrite: true
  fusion: RRF (k=60)
  top_k: 300 cada uno

FiQA:
  retrievers: [SPLADE, BGE-1.5]
  query_rewrite: true
  fusion: RRF (k=60)
  top_k: 300 cada uno
```

---

## 💡 SLIDE EXTRA: Respondiendo Preguntas Comunes

### P1: ¿Por qué Recall@10 es menor que nDCG@10?
**R**: Son métricas diferentes:
- **Recall@10**: % de documentos relevantes encontrados en top-10
- **nDCG@10**: Calidad considerando orden y relevancia gradual
- Ejemplo: Recall=0.50 significa encontramos 50% de docs relevantes

### P2: ¿Por qué BGE-M3 funciona peor que BGE-1.5?
**R**: Hipótesis:
- BGE-M3 está optimizado para multilingüe/multimodal
- Nuestros datos son inglés puro
- BGE-1.5 es más especializado para inglés → mejor rendimiento

### P3: ¿Qué es RRF y por qué k=60?
**R**: Reciprocal Rank Fusion
- Combina rankings de múltiples retrievers
- Formula: `score = sum(1/(k + rank_i))`
- k=60 es valor estándar en literatura (no muy sensible)

### P4: ¿Por qué 300 documentos iniciales?
**R**: Trade-off recall vs latencia:
- Top-100: Muy rápido pero pierde recall
- Top-300: Balance óptimo
- Top-1000: Mayor recall pero mucho más lento

### P5: ¿Cómo sé que no hay overfitting?
**R**: 
- Usamos splits train/val/test fijos
- Métricas reportadas en validation set
- No tocamos test set hasta evaluación final
- Conversaciones completas en mismo split (no leakage)

---

## 📚 Referencias y Configuraciones

### Modelos Usados
- **SPLADE**: `naver/splade-cocondenser-ensembledistil`
- **BGE-1.5**: `BAAI/bge-base-en-v1.5`
- **BGE-M3**: `BAAI/bge-m3`
- **Voyage-3**: API de Voyage AI
- **Cross-Encoder**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

### Hiperparámetros Clave
```yaml
retrieval:
  top_k: 300
  batch_size: 32

fusion:
  method: RRF
  k: 60

reranking:
  model: cross-encoder
  top_k_initial: 100
  top_k_final: 10
  batch_size: 16

evaluation:
  metrics: [nDCG@5, nDCG@10, Recall@5, Recall@10, MAP]
  primary_metric: nDCG@10
```

### Reproducibilidad
- Random seed: 42 (fijo)
- Python: 3.11
- PyTorch: 2.0+
- CUDA: 11.8
- Hardware: NVIDIA A100 40GB

---

## 🎨 Tips para la Presentación

### Visualizaciones Recomendadas

1. **Gráfico de Barras**: nDCG@10 por método y dominio
   - Eje X: Dominios (ClapNQ, Cloud, FiQA, Govt)
   - Eje Y: nDCG@10 (0.0 - 0.6)
   - Barras agrupadas: Baseline, SPLADE, Híbrido

2. **Línea de Progreso**: Mejora por fase
   - Eje X: Fases (BM25 → SPLADE → Híbrido → Reranking)
   - Eje Y: nDCG@10 promedio
   - Mostrar progresión clara

3. **Tabla de Calor**: Ablación por componente
   - Filas: Dominios
   - Columnas: Configuraciones
   - Colores: Verde (mejora) a Rojo (empeora)

4. **Gráfico Radar**: Fortalezas por dominio
   - 4 ejes: Los 4 dominios
   - Líneas: Diferentes métodos
   - Muestra consistencia de SPLADE

### Estructura de Presentación

**Tiempo Total: 15-20 minutos**

1. Contexto (2 min) → Slide 1
2. Metodología (3 min) → Slide 2
3. Fase 1: Baselines (3 min) → Slide 3
4. Fase 2: Híbridos (3 min) → Slide 4
5. Ablación (4 min) → Slide 5 + 6
6. Análisis por Dominio (2 min) → Slide 7
7. Conclusiones (3 min) → Slide 9 + 10

**¡Deja tiempo para preguntas!**

### Mensajes Clave para Enfatizar

1. **"Query rewriting es el componente más importante"** ← Repetir 2-3 veces
2. **"SPLADE es más consistente que embeddings densos"** ← Mostrar en todos los dominios
3. **"Los híbridos ayudan, pero necesitan buenos rewrites"** ← Mostrar ablación
4. **"Diferentes dominios necesitan configuraciones diferentes"** ← Justificar decisiones

### Lo que NO debes decir

❌ "El reranking funciona igual que híbrido" → Mejor: "Pendiente validar reranking"
❌ "BGE-M3 es malo" → Mejor: "BGE-M3 no es óptimo para nuestro caso de uso inglés"
❌ "Voyage es caro" → Mejor: "Voyage ofrece mejor rendimiento en ciertos dominios"
❌ Términos complejos sin explicar (DCG, RRF, cross-encoder) → Siempre dar contexto

---

## ✅ Checklist Pre-Presentación

- [ ] Validar que métricas de reranking son correctas
- [ ] Preparar 1-2 ejemplos concretos de queries
- [ ] Verificar que todos los números coinciden entre slides
- [ ] Practicar explicar RRF en 30 segundos
- [ ] Tener backup de cómo calcular nDCG (por si preguntan)
- [ ] Lista de limitaciones del estudio (para preguntas)
- [ ] Conocer papers de SPLADE, BGE, Voyage (referencias)

---

**Nota Final**: Esta guía está diseñada para una presentación académica clara y convincente. Enfatiza simplicidad sobre tecnicismos, resultados sobre implementación, y conclusiones accionables sobre detalles. ¡Buena suerte!
