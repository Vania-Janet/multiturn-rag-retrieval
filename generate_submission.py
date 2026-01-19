#!/usr/bin/env python3
"""
Generate submission file by combining best results per domain.
"""
import json
import sys
from pathlib import Path

# Best configuration per domain (based on nDCG@10)
BEST_CONFIGS = {
    "clapnq": "experiments/02-hybrid/hybrid_splade_voyage_rewrite/clapnq/retrieval_results.jsonl",
    "fiqa": "experiments/02-hybrid/hybrid_splade_bge15_rewrite/fiqa/retrieval_results.jsonl",
    "govt": "experiments/02-hybrid/hybrid_splade_voyage_rewrite/govt/retrieval_results.jsonl",
    "cloud": "experiments/03-rerank/rerank_splade_bge15_rewrite/cloud/retrieval_results.jsonl",
}

OUTPUT_FILE = "data/submissions/submission_best_hybrid.jsonl"

def load_results(file_path):
    """Load JSONL file and return list of records."""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results

def format_for_submission(record):
    """
    Transform record to submission format.
    Required fields: task_id, Collection, contexts
    Each context needs: document_id, text, score
    """
    submission_record = {
        "task_id": record["task_id"],
        "Collection": record["Collection"],
        "contexts": []
    }
    
    # Take top 10 contexts
    contexts = record.get("contexts", [])[:10]
    
    for ctx in contexts:
        submission_record["contexts"].append({
            "document_id": ctx["document_id"],
            "text": ctx["text"],
            "score": ctx["score"]
        })
    
    return submission_record

def main():
    print("=" * 60)
    print("Generating Task A Submission File")
    print("=" * 60)
    
    all_submissions = []
    
    for domain, file_path in BEST_CONFIGS.items():
        print(f"\n📂 Processing {domain.upper()}...")
        print(f"   Source: {file_path}")
        
        if not Path(file_path).exists():
            print(f"   ❌ ERROR: File not found!")
            sys.exit(1)
        
        results = load_results(file_path)
        print(f"   ✅ Loaded {len(results)} records")
        
        # Convert to submission format
        for record in results:
            submission_record = format_for_submission(record)
            all_submissions.append(submission_record)
    
    # Write output
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for record in all_submissions:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"\n{'=' * 60}")
    print(f"✅ SUCCESS!")
    print(f"{'=' * 60}")
    print(f"Total records: {len(all_submissions)}")
    print(f"Output file: {OUTPUT_FILE}")
    
    # Print domain breakdown
    domain_counts = {}
    for record in all_submissions:
        domain = record["Collection"]
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    print(f"\nDomain breakdown:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count} tasks")

if __name__ == "__main__":
    main()
