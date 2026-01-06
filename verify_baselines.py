import os
import json
import glob

BASE_PATH = "/workspace/mt-rag-benchmark/task_a_retrieval/experiments/0-baselines"
DOMAINS = ["clapnq", "cloud", "fiqa", "govt"]
EXPERIMENTS = [
    "A0_baseline_bm25_fullhist",
    "A0_baseline_splade_fullhist",
    "A1_baseline_bgem3_fullhist",
    "A1_baseline_voyage_fullhist",
    "replication_bge15",
    "replication_bgem3",
    "replication_bm25",
    "replication_splade",
    "replication_voyage",
]

print(f"{'Experiment':<30} | {'Domain':<8} | {'Status':<8} | {'nDCG@10':<8} | {'Mode':<12} | {'Check'}")
print("-" * 100)

for exp in EXPERIMENTS:
    for domain in DOMAINS:
        path = os.path.join(BASE_PATH, exp, domain)
        metrics_file = os.path.join(path, "metrics.json")
        config_file = os.path.join(path, "..", "config_resolved.yaml") # Config is usually at experiment level? Or domain? 
        # Actually checking file structure: experiments/0-baselines/replication_bm25/config_resolved.yaml vs experiments/0-baselines/replication_bm25/clapnq/metrics.json
        # Wait, context said: "The user's current file is /workspace/mt-rag-benchmark/task_a_retrieval/experiments/0-baselines/replication_bm25/config_resolved.yaml"
        # So it seems config is per experiment, not per domain.
        
        config_file_exp = os.path.join(BASE_PATH, exp, "config_resolved.yaml") 
        
        status = "MISSING"
        ndcg = "-"
        check = []
        mode = "-"

        # Check Config
        if os.path.exists(config_file_exp):
            try:
                # Simple parsing to avoid pyyaml dependency if not installed, or use json if it was json
                # But it is yaml. I'll just check if file exists for now or grep simple string
                with open(config_file_exp, 'r') as f:
                     content = f.read()
                     if "full_history" in content:
                         mode = "FullHist"
                     elif "last_turn" in content:
                         mode = "LastTurn"
                     else:
                         mode = "Unknown"
            except:
                mode = "Error"
        
        # Check Metrics
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    ndcg_val = data.get('nDCG')
                    if isinstance(ndcg_val, list) and len(ndcg_val) > 2:
                        val = ndcg_val[2] # Index 2 for @10 usually
                        ndcg = f"{val:.4f}"
                        if val == 0:
                            check.append("ZERO_SCORE")
                        elif val > 1:
                            check.append("INVALID_SCORE")
                    else:
                         check.append("BAD_METRICS")
                    status = "OK"
            except Exception as e:
                status = "ERR"
                check.append("JSON_ERR")
        else:
            status = "MISSING"

        # Logic Check
        if "fullhist" in exp and mode != "FullHist" and mode != "-":
             check.append(f"CFIG_MISMATCH({mode})")
        if "replication" in exp and mode != "LastTurn" and mode != "-":
             check.append(f"CFIG_MISMATCH({mode})")

        check_str = ", ".join(check) if check else "OK"
        print(f"{exp:<30} | {domain:<8} | {status:<8} | {ndcg:<8} | {mode:<12} | {check_str}")
