#!/usr/bin/env python3
"""
Upload new experimental results to HuggingFace

Includes:
- 01-query experiments (baseline query rewrites)
- 03-rerank experiments (BGE and Cohere rerankers)
- analysis_report.json files
"""

import os
import json
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

load_dotenv()

# Configuration
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
HF_USERNAME = os.getenv("HUGGINGFACE_USERNAME")
HF_REPO = os.getenv("HUGGINGFACE_REPO")
REPO_ID = f"{HF_USERNAME}/{HF_REPO}"

# Experiments to upload
EXPERIMENTS_TO_UPLOAD = [
    # 01-query experiments
    "experiments/01-query/baseline/bm25_rewrite",
    "experiments/01-query/baseline/splade_rewrite",
    "experiments/01-query/baseline/bge15_rewrite",
    "experiments/01-query/baseline/voyage_rewrite",
    
    # 03-rerank experiments (BGE)
    "experiments/rerank_splade_voyage_rewrite",
    "experiments/rerank_splade_bge15_rewrite",
    
    # 03-rerank experiments (Cohere)
    "experiments/rerank_cohere_splade_voyage_rewrite",
    "experiments/rerank_cohere_splade_bge15_rewrite",
]

def get_experiment_size(exp_path: Path) -> int:
    """Calculate total size of experiment directory"""
    total = 0
    for file in exp_path.rglob("*"):
        if file.is_file():
            total += file.stat().st_size
    return total

def count_files(exp_path: Path, pattern: str = "*") -> int:
    """Count files matching pattern"""
    return len(list(exp_path.rglob(pattern)))

def main():
    print("=" * 90)
    print("  UPLOADING NEW RESULTS TO HUGGINGFACE")
    print("=" * 90)
    print()
    
    if not HF_TOKEN:
        print("ERROR: HUGGINGFACE_TOKEN not found in .env")
        return
    
    print(f"Repository: {REPO_ID}")
    print()
    
    # Initialize HuggingFace API
    api = HfApi(token=HF_TOKEN)
    
    # Check if repo exists
    try:
        api.repo_info(repo_id=REPO_ID, repo_type="dataset")
        print(f"SUCCESS: Repository {REPO_ID} found")
    except Exception as e:
        print(f"ERROR: Could not access repository: {e}")
        return
    
    print()
    
    # Collect experiments to upload
    experiments_found = []
    total_size = 0
    
    print("Scanning experiments...")
    print()
    
    for exp_path_str in EXPERIMENTS_TO_UPLOAD:
        exp_path = Path(exp_path_str)
        
        if not exp_path.exists():
            print(f"  SKIP: {exp_path_str} (not found)")
            continue
        
        # Check if it has results
        has_metrics = count_files(exp_path, "metrics.json") > 0
        has_results = count_files(exp_path, "retrieval_results.jsonl") > 0
        
        if not (has_metrics or has_results):
            print(f"  SKIP: {exp_path_str} (no results)")
            continue
        
        size = get_experiment_size(exp_path)
        total_size += size
        
        experiments_found.append({
            'path': exp_path,
            'size': size,
            'metrics': count_files(exp_path, "metrics.json"),
            'analysis': count_files(exp_path, "analysis_report.json"),
            'results': count_files(exp_path, "retrieval_results.jsonl"),
        })
        
        print(f"  FOUND: {exp_path_str}")
        print(f"    Size: {size / 1024 / 1024:.2f} MB")
        print(f"    Files: {experiments_found[-1]['metrics']} metrics, " 
              f"{experiments_found[-1]['analysis']} analysis, "
              f"{experiments_found[-1]['results']} results")
        print()
    
    if not experiments_found:
        print("ERROR: No experiments found to upload")
        return
    
    print(f"Total experiments: {len(experiments_found)}")
    print(f"Total size: {total_size / 1024 / 1024:.2f} MB")
    print()
    
    # Upload each experiment
    print("=" * 90)
    print("  UPLOADING")
    print("=" * 90)
    print()
    
    for i, exp in enumerate(experiments_found, 1):
        exp_path = exp['path']
        rel_path = str(exp_path)
        
        print(f"[{i}/{len(experiments_found)}] Uploading {rel_path}...")
        
        try:
            # Upload the entire directory
            api.upload_folder(
                folder_path=str(exp_path),
                path_in_repo=rel_path,
                repo_id=REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN,
            )
            
            print(f"  SUCCESS: Uploaded {exp['size'] / 1024 / 1024:.2f} MB")
            print()
            
        except Exception as e:
            print(f"  ERROR: Failed to upload {rel_path}")
            print(f"    {e}")
            print()
            continue
    
    print("=" * 90)
    print("  UPLOAD COMPLETE")
    print("=" * 90)
    print()
    print(f"View at: https://huggingface.co/datasets/{REPO_ID}")
    print()
    
    # Summary
    print("Uploaded experiments:")
    for exp in experiments_found:
        print(f"  - {exp['path']}")
    print()
    
    print(f"Total uploaded: {total_size / 1024 / 1024:.2f} MB")
    print()
    
    # Verification
    print("=" * 90)
    print("  VERIFICATION")
    print("=" * 90)
    print()
    print("Verifying uploaded files...")
    
    try:
        repo_files = list(api.list_repo_files(repo_id=REPO_ID, repo_type="dataset"))
        
        # Check key experiments
        key_patterns = [
            "experiments/01-query/baseline/bm25_rewrite",
            "experiments/01-query/baseline/splade_rewrite",
            "experiments/rerank_splade_voyage_rewrite",
            "experiments/rerank_cohere_splade_voyage_rewrite",
        ]
        
        for pattern in key_patterns:
            matching = [f for f in repo_files if pattern in f]
            if matching:
                # Count analysis_report.json files
                analysis_files = [f for f in matching if "analysis_report.json" in f]
                metrics_files = [f for f in matching if "metrics.json" in f]
                
                print(f"  SUCCESS: {pattern}")
                print(f"    {len(metrics_files)} metrics.json")
                print(f"    {len(analysis_files)} analysis_report.json")
            else:
                print(f"  WARNING: {pattern} not found in repo")
        
        print()
        print(f"Total files in repo: {len(repo_files)}")
        
    except Exception as e:
        print(f"ERROR: Could not verify: {e}")
    
    print()
    print("=" * 90)

if __name__ == "__main__":
    main()
