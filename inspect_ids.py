
import pickle
import json
import sys
from pathlib import Path

# Index Path
index_path = Path("indices/clapnq/voyage")
pkl_path = index_path / "documents.pkl"

if pkl_path.exists():
    with open(pkl_path, 'rb') as f:
        docs = pickle.load(f)
        print(f"Loaded {len(docs)} docs from pickle")
        
        # Build set of IDs
        if isinstance(docs, list) and isinstance(docs[0], dict):
            doc_ids = set(d.get("_id") or d.get("id") for d in docs)
        elif isinstance(docs, list):
            doc_ids = set(docs)
            
        print(f"Total Unique Doc IDs: {len(doc_ids)}")
        
        target_id = "822086267_7384-8758-0-1374"
        if target_id in doc_ids:
            print(f"SUCCESS: Target ID '{target_id}' FOUND in Index.")
        else:
            print(f"FAILURE: Target ID '{target_id}' NOT FOUND in Index.")
            print(f"Sample IDs: {list(doc_ids)[:3]}")

# Query/Qrels
query_file = Path("data/retrieval_tasks/clapnq/clapnq_lastturn.jsonl")
print(f"\nChecking Qrels in {query_file}")
with open(query_file, 'r') as f:
    for i, line in enumerate(f):
        if i >= 5: break
        data = json.loads(line)
        # Expected 'target_ids' or similar in retrieval task
        print(f"Query {i} Target IDs: {data.get('target_ids') or data.get('relevant_docs') or data.get('gold_ids')}")

