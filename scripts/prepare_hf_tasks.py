import json
import argparse
import os

def main():
    input_path = "data/finetune_splits/test.jsonl"
    output_path = "data/finetune_splits/test_tasks.jsonl"
    
    print(f"Converting {input_path} -> {output_path}")
    
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for i, line in enumerate(fin):
            item = json.loads(line)
            query = item['query']
            positive = item['positive']
            
            # Create a task object compatible with the pipeline
            # We explicitly store the positive text to use it for evaluation later
            task = {
                "task_id": f"hf_test_{i}",
                "input": [{"speaker": "user", "text": query}],
                "ground_truth_text": positive,
                "domain": "mixed" # The HF repo mixes domains? (ClapNQ, Cloud, FiQA, Govt)
            }
            fout.write(json.dumps(task) + "\n")
            
    print(f"Created {i+1} tasks.")

if __name__ == "__main__":
    main()
