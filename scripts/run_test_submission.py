#!/usr/bin/env python3
"""
Execute retrieval on official test set (rag_taskAC.jsonl) using best configurations
identified from dev set analysis.
"""
import sys
import os
import json
import subprocess
import yaml
import shutil
from pathlib import Path

# Best configurations per domain (based on nDCG@10 analysis)
BEST_CONFIGS = {
    "clapnq": {
        "config": "configs/experiments/02-hybrid/hybrid_splade_voyage_rewrite.yaml",
        "ndcg@10_dev": 0.63378,
        "description": "Hybrid SPLADE + Voyage + Query Rewrite"
    },
    "fiqa": {
        "config": "configs/experiments/02-hybrid/hybrid_splade_bge15_rewrite.yaml",
        "ndcg@10_dev": 0.50291,
        "description": "Hybrid SPLADE + BGE-1.5 + Query Rewrite"
    },
    "govt": {
        "config": "configs/experiments/02-hybrid/hybrid_splade_voyage_rewrite.yaml",
        "ndcg@10_dev": 0.60689,
        "description": "Hybrid SPLADE + Voyage + Query Rewrite"
    },
    "cloud": {
        "config": "configs/experiments/03-rerank/rerank_splade_bge15_rewrite.yaml",
        "ndcg@10_dev": 0.51988,
        "description": "Hybrid SPLADE + BGE + BGE-Reranker-v2-m3"
    }
}

TEST_FILE = "src/pipeline/evaluation/rag_taskAC.jsonl"
OUTPUT_BASE = "experiments/test_submission"
SUBMISSION_FILE = "submission.jsonl"
TEMP_CONFIG_DIR = "configs/temp_test_configs"

def create_test_config(original_config_path: str, domain: str) -> Path:
    """Create temporary config file that points to test queries."""
    # Load original config
    with open(original_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update to use test query file
    if 'data' not in config:
        config['data'] = {}
    
    # Point to test queries
    config['data']['query_file'] = f"data/retrieval_tasks/{domain}/{domain}_test_questions.jsonl"
    config['data']['domains'] = [domain]
    
    # Ensure output goes to test submission dir
    if 'output' not in config:
        config['output'] = {}
    config['output']['results_dir'] = f"experiments/test_submission/{domain}"
    
    # Save temporary config
    temp_config_dir = Path(TEMP_CONFIG_DIR)
    temp_config_dir.mkdir(parents=True, exist_ok=True)
    
    temp_config_path = temp_config_dir / f"{domain}_test.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"   Created temp config: {temp_config_path}")
    return temp_config_path

def check_prerequisites():
    """Verify that all required files and indices exist."""
    print("🔍 Checking prerequisites...")
    
    # Check test file
    if not Path(TEST_FILE).exists():
        print(f"❌ Test file not found: {TEST_FILE}")
        return False
    
    # Count test tasks
    with open(TEST_FILE) as f:
        test_count = sum(1 for _ in f)
    print(f"✅ Test file found: {test_count} tasks")
    
    # Check config files
    for domain, info in BEST_CONFIGS.items():
        config_file = info["config"]
        if not Path(config_file).exists():
            print(f"❌ Config not found: {config_file}")
            return False
    print(f"✅ All {len(BEST_CONFIGS)} config files found")
    
    # Check test query files
    for domain in BEST_CONFIGS.keys():
        query_file = Path(f"data/retrieval_tasks/{domain}/{domain}_test_questions.jsonl")
        if not query_file.exists():
            print(f"❌ Test query file not found: {query_file}")
            print(f"   Run: python3 extract_test_queries.py")
            return False
    print(f"✅ All test query files found")
    
    # Check indices (basic check)
    required_indices = ["splade", "voyage", "bge"]
    for domain in BEST_CONFIGS.keys():
        for idx_type in required_indices:
            idx_path = Path(f"indices/{domain}/{idx_type}")
            if not idx_path.exists():
                print(f"⚠️  Warning: Index may be missing: {idx_path}")
    
    return True

def run_domain_retrieval(domain: str, config_path: str) -> bool:
    """Execute retrieval for a single domain using pipeline runner."""
    print(f"\n{'='*60}")
    print(f"🚀 Processing {domain.upper()}")
    print(f"   Config: {config_path}")
    print(f"   Description: {BEST_CONFIGS[domain]['description']}")
    print(f"   Expected nDCG@10 (dev): {BEST_CONFIGS[domain]['ndcg@10_dev']}")
    print(f"{'='*60}\n")
    
    output_dir = Path(OUTPUT_BASE) / domain
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary config that points to test queries
    temp_config = create_test_config(config_path, domain)
    
    # Add current directory to PYTHONPATH for imports
    env = os.environ.copy()
    current_dir = os.getcwd()
    env['PYTHONPATH'] = f"{current_dir}:{env.get('PYTHONPATH', '')}"
    
    cmd = [
        "python3", "src/pipeline/run.py",
        "--config", str(temp_config),
        "--domain", domain,
        "--output_dir", str(output_dir),
        "--force"  # Force re-run even if files exist
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print(f"✅ {domain} completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {domain} failed!")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def combine_results():
    """Combine all domain results into single submission file."""
    print("\n" + "="*60)
    print("📦 Combining results into submission file...")
    print("="*60 + "\n")
    
    all_results = []
    domain_counts = {}
    
    for domain in BEST_CONFIGS.keys():
        results_file = Path(OUTPUT_BASE) / domain / "retrieval_results.jsonl"
        
        if not results_file.exists():
            print(f"❌ Results not found for {domain}: {results_file}")
            continue
        
        count = 0
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                record = json.loads(line)
                
                # Format for submission
                submission_record = {
                    "task_id": record["task_id"],
                    "Collection": record["Collection"],
                    "contexts": []
                }
                
                # Take only top 10 contexts
                for ctx in record.get("contexts", [])[:10]:
                    submission_record["contexts"].append({
                        "document_id": ctx["document_id"],
                        "text": ctx["text"],
                        "score": ctx["score"]
                    })
                
                all_results.append(submission_record)
                count += 1
        
        domain_counts[domain] = count
        print(f"  ✅ {domain}: {count} tasks")
    
    # Write submission file
    with open(SUBMISSION_FILE, 'w', encoding='utf-8') as f:
        for record in all_results:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Submission file created: {SUBMISSION_FILE}")
    print(f"   Total tasks: {len(all_results)}")
    print(f"   Domain breakdown: {domain_counts}")
    
    return len(all_results) > 0

def validate_submission():
    """Run format checker on submission file."""
    print("\n" + "="*60)
    print("✅ Validating submission format...")
    print("="*60 + "\n")
    
    cmd = [
        "python3", "src/pipeline/evaluation/format_checker.py",
        "--mode", "retrieval_taska",
        "--prediction_file", SUBMISSION_FILE,
        "--input_file", TEST_FILE
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print("\n✅ Validation passed!")
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Validation failed!")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def main():
    print("="*60)
    print("🎯 TEST SET SUBMISSION GENERATOR")
    print("="*60)
    print(f"\nTest file: {TEST_FILE}")
    print(f"Output: {SUBMISSION_FILE}")
    print(f"\nBest configurations (from dev set analysis):")
    for domain, info in BEST_CONFIGS.items():
        print(f"  • {domain}: {info['description']} (nDCG@10={info['ndcg@10_dev']})")
    print()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed. Please fix issues above.")
        return 1
    
    # Ask for confirmation
    response = input("\n⚠️  This will execute retrieval on the TEST SET. Continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Cancelled.")
        return 0
    
    # Run retrieval for each domain
    success_count = 0
    for domain, info in BEST_CONFIGS.items():
        if run_domain_retrieval(domain, info["config"]):
            success_count += 1
    
    if success_count < len(BEST_CONFIGS):
        print(f"\n⚠️  Only {success_count}/{len(BEST_CONFIGS)} domains completed successfully")
        response = input("Continue with combining results? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            return 1
    
    # Combine results
    if not combine_results():
        print("\n❌ Failed to combine results")
        return 1
    
    # Validate submission
    if not validate_submission():
        print("\n❌ Validation failed")
        return 1
    
    print("\n" + "="*60)
    print("🎉 SUCCESS!")
    print("="*60)
    print(f"\nSubmission file ready: {SUBMISSION_FILE}")
    print("\nNext steps:")
    print("  1. Review the submission file")
    print("  2. Submit to evaluation platform")
    print("  3. Document the results")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
