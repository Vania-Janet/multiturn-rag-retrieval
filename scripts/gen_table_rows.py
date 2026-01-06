
import json
import os
import glob

# Config
BASE_PATH = "experiments/baselines"
DOMAINS = ["clapnq", "fiqa", "govt", "cloud"] # Order matches table columns
EXPERIMENTS_MAP = {
    "A0: Sparse Baseline": "replication_bm25",
    "A1: Dense Baseline": "replication_bgem3",
    "SPLADE Baseline": "replication_splade" # Adding this as it's computed
}

def get_metrics(exp_id, domain):
    path = os.path.join(BASE_PATH, exp_id, domain, "metrics.json")
    report_path = os.path.join(BASE_PATH, exp_id, domain, "analysis_report.json")
    
    metrics = {"ndcg": "-", "recall": "-", "map": "-", "latency": "-"}
    
    if os.path.exists(path):
        try:
            with open(path) as f:
                d = json.load(f)
                # Index 2 is @10
                metrics["ndcg"] = f"{d.get('nDCG', [0,0,0])[2]:.4f}"
                metrics["recall"] = f"{d.get('Recall', [0,0,0])[2]:.4f}"
                metrics["map"] = f"{d.get('MAP', [0,0,0])[2]:.4f}"
        except:
            pass

    if os.path.exists(report_path):
        try:
            with open(report_path) as f:
                d = json.load(f)
                lat = d.get("latency", {}).get("avg_latency_sec", 0)
                metrics["latency"] = f"{lat*1000:.0f}"
        except:
            pass
            
    return metrics

print("Generating table rows...")
for label, exp_folder in EXPERIMENTS_MAP.items():
    print(f"\n### {label}")
    print("| Domain | Recall@10 | MRR | NDCG@10 | MAP | Latency (ms) |")
    print("|---|---|---|---|---|---|")
    
    vals = []
    for domain in DOMAINS: # clapnq, fiqa, govt, cloud
        m = get_metrics(exp_folder, domain)
        print(f"| {domain.capitalize()} | {m['recall']} | - | {m['ndcg']} | {m['map']} | {m['latency']} |")
        vals.append(float(m['ndcg']) if m['ndcg'] != '-' else 0)
    
    avg = sum(vals)/len(vals) if vals else 0
    print(f"AVG NDCG: {avg:.4f}")
