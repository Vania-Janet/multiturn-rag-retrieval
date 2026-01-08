#!/usr/bin/env python3
"""
Análisis detallado: Por qué Cohere reranking falló (-20.9%)
"""

import json
from pathlib import Path

print("=" * 80)
print("  ANÁLISIS DETALLADO: Por qué Cohere reranking falló (-20.9%)")
print("=" * 80)
print()

# Analizar CLAPNQ como caso de estudio
domain = "clapnq"
baseline_file = "experiments/hybrid_splade_voyage_rewrite/clapnq/retrieval_results.jsonl"
cohere_file = "experiments/rerank_cohere_splade_voyage_rewrite/clapnq/retrieval_results.jsonl"

print(f"📊 Analizando dominio: {domain.upper()}")
print()

# Leer primeras 5 queries de cada uno
baseline_queries = []
with open(baseline_file) as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        baseline_queries.append(json.loads(line))

cohere_queries = []
with open(cohere_file) as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        cohere_queries.append(json.loads(line))

# Análisis 1: Verificar si rerank_score existe y se usa
print("1️⃣  VERIFICACIÓN DE SCORES:")
print("-" * 80)

sample_contexts = cohere_queries[0]['contexts'][:3]
print(f"Top-3 documentos del primer query (Cohere):")
for i, ctx in enumerate(sample_contexts, 1):
    keys = list(ctx.keys())
    print(f"  Doc {i}: Keys disponibles: {keys}")
    if 'rerank_score' in ctx:
        print(f"         original_score={ctx.get('original_score', 'N/A'):.5f}, rerank_score={ctx['rerank_score']:.5f}, score={ctx['score']:.5f}")
    else:
        print(f"         ⚠️  NO HAY rerank_score! Solo score={ctx['score']:.5f}")

print()

# Análisis 2: Comparar rankings entre baseline y cohere
print("2️⃣  COMPARACIÓN DE RANKINGS (Top-10):")
print("-" * 80)

for q_idx in range(min(2, len(baseline_queries))):
    task_id = baseline_queries[q_idx]['task_id']
    
    baseline_top10 = [ctx['document_id'] for ctx in baseline_queries[q_idx]['contexts'][:10]]
    cohere_top10 = [ctx['document_id'] for ctx in cohere_queries[q_idx]['contexts'][:10]]
    
    # Calcular overlap
    overlap = len(set(baseline_top10) & set(cohere_top10))
    
    print(f"\nQuery {q_idx+1} (task_id: {task_id}):")
    print(f"  Overlap en Top-10: {overlap}/10 documentos")
    
    # Ver posiciones de los docs del baseline en cohere
    baseline_in_cohere = []
    for rank, doc_id in enumerate(baseline_top10[:5], 1):
        if doc_id in cohere_top10:
            new_rank = cohere_top10.index(doc_id) + 1
            direction = "⬆️" if new_rank < rank else "⬇️" if new_rank > rank else "="
            baseline_in_cohere.append(f"{rank}→{new_rank}{direction}")
        else:
            baseline_in_cohere.append(f"{rank}→OUT❌")
    
    print(f"  Movimientos Top-5: {', '.join(baseline_in_cohere)}")

print()

# Análisis 3: Distribución de scores
print("3️⃣  DISTRIBUCIÓN DE SCORES:")
print("-" * 80)

baseline_scores = [ctx['score'] for ctx in baseline_queries[0]['contexts'][:100]]
cohere_scores = [ctx['score'] for ctx in cohere_queries[0]['contexts'][:100]]

if 'rerank_score' in cohere_queries[0]['contexts'][0]:
    rerank_scores = [ctx['rerank_score'] for ctx in cohere_queries[0]['contexts'][:100]]
    original_scores = [ctx.get('original_score', 0) for ctx in cohere_queries[0]['contexts'][:100]]
    
    print(f"Baseline (RRF scores):")
    print(f"  Min: {min(baseline_scores):.5f}, Max: {max(baseline_scores):.5f}, Media: {sum(baseline_scores)/len(baseline_scores):.5f}")
    
    print(f"\nCohere (rerank_scores):")
    print(f"  Min: {min(rerank_scores):.5f}, Max: {max(rerank_scores):.5f}, Media: {sum(rerank_scores)/len(rerank_scores):.5f}")
    
    print(f"\nCohere (scores finales usados):")
    print(f"  Min: {min(cohere_scores):.5f}, Max: {max(cohere_scores):.5f}, Media: {sum(cohere_scores)/len(cohere_scores):.5f}")
    
    # Verificar si score == rerank_score
    if abs(cohere_scores[0] - rerank_scores[0]) < 0.0001:
        print(f"  ✅ Confirmado: score = rerank_score (correcto)")
    else:
        print(f"  ⚠️  PROBLEMA: score ≠ rerank_score (score={cohere_scores[0]:.5f}, rerank={rerank_scores[0]:.5f})")
else:
    print("❌ ERROR: No hay rerank_score en los resultados de Cohere!")

print()

# Análisis 4: Ver scores específicos de documentos relevantes
print("4️⃣  ANÁLISIS DE DOCUMENTOS RELEVANTES:")
print("-" * 80)

# Cargar qrels para saber qué docs son relevantes
qrels_file = "data/retrieval_tasks/clapnq/qrels/dev.tsv"
qrels = {}
with open(qrels_file) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            task_id, _, doc_id, rel = parts[0], parts[1], parts[2], int(parts[3])
            if task_id not in qrels:
                qrels[task_id] = {}
            qrels[task_id][doc_id] = rel

# Para el primer query, ver dónde están los docs relevantes
task_id = baseline_queries[0]['task_id']
if task_id in qrels:
    relevant_docs = {doc_id: rel for doc_id, rel in qrels[task_id].items() if rel > 0}
    
    print(f"Query: {task_id}")
    print(f"  Documentos relevantes totales: {len(relevant_docs)}")
    
    # Encontrar posiciones en baseline
    baseline_docs = [ctx['document_id'] for ctx in baseline_queries[0]['contexts']]
    baseline_rel_positions = []
    for doc_id in relevant_docs:
        if doc_id in baseline_docs:
            pos = baseline_docs.index(doc_id) + 1
            baseline_rel_positions.append(pos)
    
    # Encontrar posiciones en cohere
    cohere_docs = [ctx['document_id'] for ctx in cohere_queries[0]['contexts']]
    cohere_rel_positions = []
    for doc_id in relevant_docs:
        if doc_id in cohere_docs:
            pos = cohere_docs.index(doc_id) + 1
            cohere_rel_positions.append(pos)
    
    print(f"\n  Baseline: {len(baseline_rel_positions)} relevantes en Top-100")
    print(f"  Posiciones: {sorted(baseline_rel_positions[:10])}")
    
    print(f"\n  Cohere: {len(cohere_rel_positions)} relevantes en Top-100")
    print(f"  Posiciones: {sorted(cohere_rel_positions[:10])}")
    
    # Ver si Cohere empujó documentos relevantes hacia abajo
    if baseline_rel_positions and cohere_rel_positions:
        avg_baseline = sum(baseline_rel_positions) / len(baseline_rel_positions)
        avg_cohere = sum(cohere_rel_positions) / len(cohere_rel_positions)
        
        print(f"\n  Posición promedio de docs relevantes:")
        print(f"    Baseline: {avg_baseline:.1f}")
        print(f"    Cohere: {avg_cohere:.1f}")
        if avg_cohere > avg_baseline:
            print(f"    ⚠️  Cohere empujó docs relevantes {avg_cohere - avg_baseline:.1f} posiciones ABAJO")
        else:
            print(f"    ✅ Cohere mejoró {avg_baseline - avg_cohere:.1f} posiciones")

print()
print("=" * 80)
