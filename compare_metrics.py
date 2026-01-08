import json
from pathlib import Path

# Voyage comparison
print("=" * 80)
print("VOYAGE - ClapNQ")
print("=" * 80)
baseline = json.load(open("experiments/0-baselines/A1_baseline_voyage_fullhist/clapnq/metrics.json"))
condensation = json.load(open("experiments/01-query/voyage_r1_condensation/clapnq/metrics.json"))
multi = json.load(open("experiments/01-query/voyage_r2_multi/clapnq/metrics.json"))

print(f"Baseline (fullhist)    - nDCG@10: {baseline['nDCG'][0]:.4f}, Recall@100: {baseline['Recall'][2]:.4f}")
print(f"R1 Condensation        - nDCG@10: {condensation['nDCG'][0]:.4f}, Recall@100: {condensation['Recall'][2]:.4f}")
print(f"R2 Multi               - nDCG@10: {multi['nDCG'][0]:.4f}, Recall@100: {multi['Recall'][2]:.4f}")

print("\n" + "=" * 80)
print("BM25 - ClapNQ")
print("=" * 80)
baseline_bm25 = json.load(open("experiments/0-baselines/replication_bm25/clapnq/metrics.json"))
condensation_bm25 = json.load(open("experiments/01-query/bm25_r1_condensation/clapnq/metrics.json"))
multi_bm25 = json.load(open("experiments/01-query/bm25_r2_multi/clapnq/metrics.json"))

print(f"Baseline (replication) - nDCG@10: {baseline_bm25['nDCG'][0]:.4f}, Recall@100: {baseline_bm25['Recall'][2]:.4f}")
print(f"R1 Condensation        - nDCG@10: {condensation_bm25['nDCG'][0]:.4f}, Recall@100: {condensation_bm25['Recall'][2]:.4f}")
print(f"R2 Multi               - nDCG@10: {multi_bm25['nDCG'][0]:.4f}, Recall@100: {multi_bm25['Recall'][2]:.4f}")
