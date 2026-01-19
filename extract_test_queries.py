#!/usr/bin/env python3
"""
Extract queries from test set (rag_taskAC.jsonl) and create query files per domain.
"""
import json
from pathlib import Path
from collections import defaultdict

TEST_FILE = "src/pipeline/evaluation/rag_taskAC.jsonl"
OUTPUT_DIR = "data/retrieval_tasks"

def extract_test_queries():
    """Extract queries from test set and organize by domain."""
    print(f"📖 Reading test set from: {TEST_FILE}")
    
    # Group queries by domain
    queries_by_domain = defaultdict(list)
    
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                task = json.loads(line)
                domain = task.get("Collection", "").lower()
                
                # Handle domain name variations
                if domain == "ibmcloud":
                    domain = "cloud"
                
                if domain not in ["clapnq", "cloud", "fiqa", "govt"]:
                    print(f"⚠️  Warning: Unknown domain '{domain}' at line {line_num}")
                    continue
                
                # Extract the query (last turn)
                messages = task.get("input", [])
                if not messages:
                    print(f"⚠️  Warning: No input messages at line {line_num}")
                    continue
                
                # Get last message from user
                last_user_msg = None
                for msg in reversed(messages):
                    if msg.get("speaker") == "user":
                        last_user_msg = msg.get("text", "")
                        break
                
                if not last_user_msg:
                    print(f"⚠️  Warning: No user message found at line {line_num}")
                    continue
                
                query_record = {
                    "task_id": task["task_id"],
                    "question": last_user_msg,
                    "Collection": domain,
                    "turn_id": task.get("turn_id", 0)
                }
                
                queries_by_domain[domain].append(query_record)
                
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing line {line_num}: {e}")
                continue
    
    # Write query files for each domain
    total_queries = 0
    for domain, queries in sorted(queries_by_domain.items()):
        output_file = Path(OUTPUT_DIR) / domain / f"{domain}_test_questions.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for query in queries:
                f.write(json.dumps(query, ensure_ascii=False) + '\n')
        
        print(f"✅ {domain}: {len(queries)} queries -> {output_file}")
        total_queries += len(queries)
    
    print(f"\n📊 Total: {total_queries} queries across {len(queries_by_domain)} domains")
    return queries_by_domain

if __name__ == "__main__":
    print("="*60)
    print("🔧 TEST SET QUERY EXTRACTION")
    print("="*60)
    print()
    
    queries_by_domain = extract_test_queries()
    
    print("\n" + "="*60)
    print("✅ COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Optionally run query rewriting on test queries")
    print("  2. Run retrieval using run_test_submission.py")
    print()
