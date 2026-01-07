#!/usr/bin/env python3
"""
Quick progress checker for 01-query experiments.
Shows which experiments have completed and their metrics.
"""

import json
from pathlib import Path

EXPERIMENTS = [
    "bm25_r1_condensation",
    "bm25_r2_multi", 
    "splade_r1_condensation",
    "splade_r3_hyde",
    "bgem3_r1_condensation",
    "bgem3_r2_multi",
    "voyage_r1_condensation",
    "voyage_r2_multi",
]

DOMAINS = ["clapnq", "cloud", "fiqa", "govt"]

def check_progress():
    """Check which experiments have completed."""
    
    results = []
    total = len(EXPERIMENTS) * len(DOMAINS)
    completed = 0
    
    for exp in EXPERIMENTS:
        exp_results = {"experiment": exp}
        
        for domain in DOMAINS:
            metrics_file = Path(f"experiments/01-query/{exp}/{domain}/metrics.json")
            
            if metrics_file.exists():
                try:
                    with open(metrics_file) as f:
                        metrics = json.load(f)
                    
                    # Get NDCG@10 (usually index 2 in nDCG array)
                    ndcg10 = metrics.get("nDCG", [0,0,0])[2] if len(metrics.get("nDCG", [])) > 2 else 0
                    recall10 = metrics.get("Recall", [0,0,0])[1] if len(metrics.get("Recall", [])) > 1 else 0
                    
                    exp_results[domain] = f"✓ {ndcg10:.3f}"
                    completed += 1
                except:
                    exp_results[domain] = "⚠ error"
            else:
                exp_results[domain] = "⏳"
        
        results.append(exp_results)
    
    # Print table
    print("\n" + "="*80)
    print("01-QUERY EXPERIMENTS PROGRESS")
    print("="*80)
    print(f"Completed: {completed}/{total} ({100*completed//total}%)\n")
    
    # Simple table format
    print(f"{'Experiment':<30} {'clapnq':<12} {'cloud':<12} {'fiqa':<12} {'govt':<12}")
    print("-" * 80)
    
    for r in results:
        exp_name = r["experiment"]
        row = f"{exp_name:<30}"
        for d in DOMAINS:
            row += f" {r.get(d, '-'):<12}"
        print(row)
    
    print("\nLegend: ✓ = completed (shows NDCG@10), ⏳ = pending, ⚠ = error")
    print("="*80 + "\n")

if __name__ == "__main__":
    check_progress()
