# Cohere v3 Embeddings Baseline Implementation

## 🎯 ¿Qué es esto?

Implementación de **Cohere embed-english-v3.0** como baseline para comparar contra SPLADE, BGE, y Voyage.

## ⚡ Quick Start (Prueba Rápida)

Para probar solo en **ClapNQ** y ver si vale la pena:

```bash
cd /workspace/mt-rag-benchmark/task_a_retrieval
./test_cohere_quick.sh
```

Esto hará:
1. Crear índice FAISS con embeddings de Cohere v3 para ClapNQ (~5-10 min)
2. Ejecutar retrieval con queries reescritas
3. Mostrar métricas vs SPLADE baseline

**Tiempo estimado**: 10-15 minutos
**Costo**: ~$0.50-1.00 USD (usando Cohere API)

## 📊 Comparación Esperada

| Métrica | SPLADE (actual) | Cohere v3 (esperado) | Diferencia |
|---------|-----------------|----------------------|------------|
| nDCG@10 | 0.524 | 0.48-0.52 | -4% a -1% |
| Recall@10 | 0.630 | 0.58-0.62 | -5% a -2% |

**Predicción**: Cohere v3 probablemente NO supera a SPLADE individualmente, pero podría funcionar bien en híbrido.

## 🚀 Ejecutar Todos los Dominios

Si la prueba rápida muestra resultados prometedores:

```bash
./run_cohere_baseline.sh
```

Esto procesará todos los dominios (clapnq, cloud, fiqa, govt).

**Tiempo estimado**: 40-60 minutos
**Costo**: ~$3-5 USD

## 📁 Estructura de Archivos

```
task_a_retrieval/
├── configs/experiments/0-baselines/
│   └── A2_baseline_cohere_rewrite.yaml       # Config del experimento
├── src/pipeline/
│   ├── retrieval/
│   │   └── cohere_embeddings.py              # Retriever de Cohere
│   └── indexing/
│       └── create_cohere_indices.py          # Script de indexación
├── test_cohere_quick.sh                       # Prueba rápida (solo ClapNQ)
└── run_cohere_baseline.sh                     # Todos los dominios
```

## 🔧 Configuración

### API Key

Ya está configurada en `.env`:
```bash
COHERE_API_KEY=PixhcshKCqAgLZ15gT7DQUrdqSiC2x8ogvcAP5AW
```

### Modelo

Usando `embed-english-v3.0`:
- **Dimensiones**: 1024
- **Batch size**: 96 documentos
- **Input type**: `search_document` para indexar, `search_query` para buscar
- **Costo**: ~$0.10 por 1M tokens

## 📈 Interpretando Resultados

### Si Cohere v3 es MEJOR que SPLADE:
✅ Vale la pena usarlo en producción  
✅ Consideralo para híbridos  
✅ Publicable como hallazgo

### Si Cohere v3 es SIMILAR a SPLADE (±2%):
🤔 Considera híbrido SPLADE + Cohere  
🤔 Evalúa costo vs beneficio  
💡 Podría complementar bien a SPLADE

### Si Cohere v3 es PEOR que SPLADE (-5% o más):
❌ No vale la pena para baselines  
💡 Enfócate en mejorar reranking  
💡 O fine-tuning de SPLADE

## 🔍 Próximos Pasos

Dependiendo de los resultados:

1. **Si funciona bien**: Probar híbrido `SPLADE + Cohere` (mejor que actual)
2. **Si funciona mal**: Arreglar reranking pipeline (más barato y efectivo)
3. **Si es intermedio**: Comparar costo/beneficio vs alternativas

## 💰 Costos Estimados

### Indexación (una sola vez)
- ClapNQ: ~20K docs × 100 tokens = 2M tokens = **$0.20**
- Cloud: ~15K docs = **$0.15**
- FiQA: ~18K docs = **$0.18**
- Govt: ~25K docs = **$0.25**
- **Total**: ~$0.80

### Retrieval (por experimento)
- ~500 queries × 50 tokens = 25K tokens = **$0.003**
- Insignificante comparado con indexación

### Total para baseline completo
**~$1.00 USD** (solo indexación, retrieval es gratis básicamente)

## ⚠️ Notas Importantes

1. **Cache**: Los embeddings de queries se cachean en `.cache/embeddings/cohere/`
2. **Checkpoints**: Si falla la indexación, se reanuda automáticamente
3. **Rate limits**: Script maneja rate limits automáticamente (espera 60s y reintenta)
4. **GPU**: FAISS usa GPU automáticamente si está disponible (solo para búsqueda)

## 🐛 Troubleshooting

### Error: "COHERE_API_KEY not found"
```bash
# Verifica que el .env existe
cat .env | grep COHERE_API_KEY
```

### Error: "Module 'cohere' not found"
```bash
pip install cohere
```

### Indexación muy lenta
- Normal: ~2-3 docs/segundo
- Si es más lento, revisa conexión a internet
- Checkpoints se guardan cada 4800 docs

### Resultados iguales a otro modelo
- Verifica que el índice correcto se creó en `indices/{domain}/cohere/`
- Borra índice y recrea si hay dudas

## 📚 Referencias

- [Cohere Embed v3 Docs](https://docs.cohere.com/docs/embed-api)
- [Cohere Pricing](https://cohere.com/pricing)
- Paper baseline: SPLADE (nDCG@10: 0.457 promedio)

## ✅ Checklist

Antes de ejecutar:
- [ ] API key configurada en `.env`
- [ ] `pip install cohere` ejecutado
- [ ] Suficiente espacio en disco (~500MB por dominio)
- [ ] Conexión a internet estable

Para validar implementación:
- [ ] Índice FAISS creado correctamente
- [ ] Doc IDs coinciden con número de vectores en índice
- [ ] Métricas se calculan correctamente
- [ ] Resultados se guardan en `experiments/A2_baseline_cohere_rewrite/`

---

**Creado**: 2026-01-14  
**Status**: ✅ Listo para probar
