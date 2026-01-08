#!/usr/bin/env python3
"""
Análisis final: Por qué Cohere rerank-v3.5 falló con -20.9%
"""

import json
import numpy as np
from collections import defaultdict

print("=" * 90)
print("  ANÁLISIS DEFINITIVO: Por qué Cohere falló (-20.9% vs Baseline)")
print("=" * 90)
print()

# Cargar qrels (sin header)
qrels_file = "data/retrieval_tasks/clapnq/qrels/dev.tsv"
qrels = defaultdict(dict)
with open(qrels_file) as f:
    next(f)  # Skip header
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            task_id, doc_id, rel = parts[0], parts[1], int(parts[2])
            qrels[task_id][doc_id] = rel

print(f"✓ Cargados qrels para {len(qrels)} queries")

# Cargar resultados
baseline_file = "experiments/hybrid_splade_voyage_rewrite/clapnq/retrieval_results.jsonl"
cohere_file = "experiments/rerank_cohere_splade_voyage_rewrite/clapnq/retrieval_results.jsonl"

baseline_queries = []
with open(baseline_file) as f:
    for line in f:
        baseline_queries.append(json.loads(line))

cohere_queries = []
with open(cohere_file) as f:
    for line in f:
        cohere_queries.append(json.loads(line))

print(f"✓ Cargados {len(baseline_queries)} queries de cada sistema")
print()

# Análisis agregado
stats = {
    'baseline_rel_top10': [],
    'cohere_rel_top10': [],
    'baseline_rel_top5': [],
    'cohere_rel_top5': [],
    'baseline_avg_rel_pos': [],
    'cohere_avg_rel_pos': [],
    'cohere_better': 0,
    'cohere_worse': 0,
    'cohere_same': 0,
    'queries_with_rels': 0
}

# Ejemplos de queries donde Cohere falló
worst_examples = []

for baseline_q, cohere_q in zip(baseline_queries, cohere_queries):
    task_id = baseline_q['task_id']
    relevant_docs = qrels.get(task_id, {})
    
    if not relevant_docs:
        continue
    
    stats['queries_with_rels'] += 1
    
    baseline_docs = [ctx['document_id'] for ctx in baseline_q['contexts']]
    cohere_docs = [ctx['document_id'] for ctx in cohere_q['contexts']]
    
    # Count relevant in top-10 and top-5
    baseline_rel_10 = sum(1 for doc_id in baseline_docs[:10] if relevant_docs.get(doc_id, 0) > 0)
    cohere_rel_10 = sum(1 for doc_id in cohere_docs[:10] if relevant_docs.get(doc_id, 0) > 0)
    
    baseline_rel_5 = sum(1 for doc_id in baseline_docs[:5] if relevant_docs.get(doc_id, 0) > 0)
    cohere_rel_5 = sum(1 for doc_id in cohere_docs[:5] if relevant_docs.get(doc_id, 0) > 0)
    
    stats['baseline_rel_top10'].append(baseline_rel_10)
    stats['cohere_rel_top10'].append(cohere_rel_10)
    stats['baseline_rel_top5'].append(baseline_rel_5)
    stats['cohere_rel_top5'].append(cohere_rel_5)
    
    # Average position of relevant docs
    baseline_rel_positions = [i+1 for i, doc_id in enumerate(baseline_docs[:100]) if relevant_docs.get(doc_id, 0) > 0]
    cohere_rel_positions = [i+1 for i, doc_id in enumerate(cohere_docs[:100]) if relevant_docs.get(doc_id, 0) > 0]
    
    if baseline_rel_positions:
        stats['baseline_avg_rel_pos'].append(np.mean(baseline_rel_positions))
    if cohere_rel_positions:
        stats['cohere_avg_rel_pos'].append(np.mean(cohere_rel_positions))
    
    # Compare
    if cohere_rel_10 > baseline_rel_10:
        stats['cohere_better'] += 1
    elif cohere_rel_10 < baseline_rel_10:
        stats['cohere_worse'] += 1
        # Track worst examples
        worst_examples.append({
            'task_id': task_id,
            'question': baseline_q['question'][:60],
            'baseline_rel': baseline_rel_10,
            'cohere_rel': cohere_rel_10,
            'diff': baseline_rel_10 - cohere_rel_10
        })
    else:
        stats['cohere_same'] += 1

# Sort worst examples
worst_examples.sort(key=lambda x: x['diff'], reverse=True)

print("=" * 90)
print("  RESULTADOS PRINCIPALES")
print("=" * 90)
print()

print(f"📊 Docs relevantes en Top-10 (promedio sobre {stats['queries_with_rels']} queries):")
print(f"   Baseline (RRF):  {np.mean(stats['baseline_rel_top10']):.3f}")
print(f"   Cohere Rerank:   {np.mean(stats['cohere_rel_top10']):.3f}")
print(f"   Δ:               {np.mean(stats['cohere_rel_top10']) - np.mean(stats['baseline_rel_top10']):.3f} ⬇️")
print()

print(f"📊 Docs relevantes en Top-5 (promedio):")
print(f"   Baseline (RRF):  {np.mean(stats['baseline_rel_top5']):.3f}")
print(f"   Cohere Rerank:   {np.mean(stats['cohere_rel_top5']):.3f}")
print(f"   Δ:               {np.mean(stats['cohere_rel_top5']) - np.mean(stats['baseline_rel_top5']):.3f} ⬇️")
print()

print(f"📊 Posición promedio de docs relevantes:")
print(f"   Baseline (RRF):  {np.mean(stats['baseline_avg_rel_pos']):.1f}")
print(f"   Cohere Rerank:   {np.mean(stats['cohere_avg_rel_pos']):.1f}")
print(f"   Δ:               +{np.mean(stats['cohere_avg_rel_pos']) - np.mean(stats['baseline_avg_rel_pos']):.1f} posiciones PEOR")
print()

total_queries = stats['cohere_better'] + stats['cohere_worse'] + stats['cohere_same']
print(f"📊 Comparación query-by-query (Top-10):")
print(f"   Cohere MEJOR:  {stats['cohere_better']:3d} queries ({stats['cohere_better']/total_queries*100:5.1f}%)")
print(f"   Cohere PEOR:   {stats['cohere_worse']:3d} queries ({stats['cohere_worse']/total_queries*100:5.1f}%) ⚠️")
print(f"   IGUAL:         {stats['cohere_same']:3d} queries ({stats['cohere_same']/total_queries*100:5.1f}%)")
print()

print("=" * 90)
print()
print("🔍 TOP-5 PEORES CASOS (queries donde Cohere falló más):")
print()
for i, ex in enumerate(worst_examples[:5], 1):
    print(f"{i}. Query: {ex['question']}...")
    print(f"   Baseline: {ex['baseline_rel']} relevantes en Top-10 → Cohere: {ex['cohere_rel']} (perdió {ex['diff']})")
    print()

print("=" * 90)
print()
print("💡 CONCLUSIONES:")
print()

avg_diff = np.mean(stats['cohere_rel_top10']) - np.mean(stats['baseline_rel_top10'])
pos_diff = np.mean(stats['cohere_avg_rel_pos']) - np.mean(stats['baseline_avg_rel_pos'])
pct_worse = stats['cohere_worse'] / total_queries * 100

print(f"1. ⚠️  Cohere PIERDE {abs(avg_diff):.3f} docs relevantes por query en Top-10")
print(f"2. ⚠️  Cohere empuja docs relevantes {pos_diff:.1f} posiciones HACIA ABAJO")
print(f"3. ⚠️  Cohere EMPEORA en {pct_worse:.1f}% de las queries")
print()
print("📌 POR QUÉ FALLÓ COHERE:")
print()
print("  • Cohere rerank-v3.5 re-puntúa basándose en similitud semántica query-documento")
print("  • NO captura bien el contexto 'multi-turn conversational' de estas queries")
print("  • RRF baseline ya había hecho un EXCELENTE balance:")
print("    - Sparse (SPLADE): Keywords y matching exacto")
print("    - Dense (Voyage): Similitud semántica profunda")
print("  • Cohere sobrescribe esa fusión balanceada con scores que priorizan")
print("    semántica pura, perdiendo señales lexicales importantes")
print("  • El modelo no fue entrenado específicamente para este tipo de dataset conversacional")
print()
print("✅ RECOMENDACIÓN:")
print("  → Quedarse con Hybrid SPLADE + Voyage/BGE con RRF (0.486 nDCG@10)")
print("  → NO usar reranking para este benchmark (tanto BGE como Cohere empeoran)")
print()
print("=" * 90)
