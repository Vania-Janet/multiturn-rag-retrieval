import json
import csv

# Load qrels
qrels = {}
with open('data/retrieval_tasks/clapnq/qrels/dev.tsv') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)  
    for row in reader:
        query_id, corpus_id, score = row[0], row[1], int(row[2])
        if query_id not in qrels:
            qrels[query_id] = {corpus_id: score}
        else:
            qrels[query_id][corpus_id] = score

# Load Voyage R1 results  
results = {}
with open('experiments/01-query/voyage_r1_condensation/clapnq/retrieval_results.jsonl') as f:
    for line in f:
        item = json.loads(line)
        query_id = item['task_id']
        doc_scores = {}
        for ctx in item.get('contexts', []):
            doc_id = ctx['document_id']
            score = ctx['score']
            doc_scores[doc_id] = score
        results[query_id] = doc_scores

# Manual calculation - count hits @ k
def calculate_recall_at_k(results, qrels, k):
    total_recall = 0.0
    num_queries = 0
    for qid in qrels:
        if qid not in results:
            continue
        num_queries += 1
        relevant_docs = set(qrels[qid].keys())
        # Get top-k retrieved docs
        retrieved_docs = list(results[qid].keys())[:k]
        retrieved_set = set(retrieved_docs)
        # Calculate recall
        hits = len(relevant_docs & retrieved_set)
        recall = hits / len(relevant_docs) if relevant_docs else 0.0
        total_recall += recall
    return total_recall / num_queries if num_queries > 0 else 0.0

print("Manual Recall Calculation for Voyage R1 Condensation:")
for k in [1, 3, 5, 10, 20, 100]:
    recall = calculate_recall_at_k(results, qrels, k)
    print(f"  Recall@{k}: {recall:.5f}")

# Compare first few queries
print("\nFirst 5 queries - check overlaps:")
for i, qid in enumerate(list(qrels.keys())[:5]):
    if qid in results:
        relevant = set(qrels[qid].keys())
        retrieved = set(list(results[qid].keys())[:10])
        overlap = relevant & retrieved
        print(f"  {i}: qid={qid[:30]}, relevant={len(relevant)}, overlap@10={len(overlap)}")
