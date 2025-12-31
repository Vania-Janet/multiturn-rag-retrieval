#!/usr/bin/env python3
"""
Comprehensive code validation script.

Tests all imports, dependencies, configs, and data files before A100 deployment.
"""

import sys
import os
from pathlib import Path
import json

# Color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header(text):
    print(f"\n{BOLD}{BLUE}{'='*80}{RESET}")
    print(f"{BOLD}{BLUE}{text:^80}{RESET}")
    print(f"{BOLD}{BLUE}{'='*80}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✓{RESET} {text}")

def print_error(text):
    print(f"{RED}✗{RESET} {text}")

def print_warning(text):
    print(f"{YELLOW}⚠{RESET} {text}")

def test_imports():
    """Test all critical imports."""
    print_header("Testing Critical Imports")
    
    sys.path.insert(0, 'src')
    errors = []
    
    # Test utils
    try:
        from utils.logger import setup_logger
        print_success("utils.logger")
    except Exception as e:
        print_error(f"utils.logger: {e}")
        errors.append(("utils.logger", e))
    
    try:
        from utils.config_loader import load_config, merge_configs
        print_success("utils.config_loader")
    except Exception as e:
        print_error(f"utils.config_loader: {e}")
        errors.append(("utils.config_loader", e))
    
    try:
        from utils.reproducibility import set_seed
        print_success("utils.reproducibility")
    except Exception as e:
        print_error(f"utils.reproducibility: {e}")
        errors.append(("utils.reproducibility", e))
    
    # Test retrieval modules (without evaluation which needs pytrec_eval)
    try:
        from pipeline.retrieval.dense import get_dense_retriever, BGERetriever
        print_success("pipeline.retrieval.dense")
    except Exception as e:
        print_error(f"pipeline.retrieval.dense: {e}")
        errors.append(("pipeline.retrieval.dense", e))
    
    try:
        from pipeline.retrieval.sparse import BM25Retriever, ELSERRetriever, get_sparse_retriever
        print_success("pipeline.retrieval.sparse")
    except Exception as e:
        print_error(f"pipeline.retrieval.sparse: {e}")
        errors.append(("pipeline.retrieval.sparse", e))
    
    try:
        from pipeline.retrieval.hybrid import HybridRetriever
        print_success("pipeline.retrieval.hybrid")
    except Exception as e:
        print_error(f"pipeline.retrieval.hybrid: {e}")
        errors.append(("pipeline.retrieval.hybrid", e))
    
    try:
        from pipeline.retrieval.fusion import reciprocal_rank_fusion
        print_success("pipeline.retrieval.fusion")
    except Exception as e:
        print_error(f"pipeline.retrieval.fusion: {e}")
        errors.append(("pipeline.retrieval.fusion", e))
    
    try:
        from pipeline.retrieval.reproducibility import set_seed
        print_success("pipeline.retrieval.reproducibility")
    except Exception as e:
        print_error(f"pipeline.retrieval.reproducibility: {e}")
        errors.append(("pipeline.retrieval.reproducibility", e))
    
    try:
        from pipeline.retrieval.analysis import LatencyMonitor
        print_success("pipeline.retrieval.analysis")
    except Exception as e:
        print_error(f"pipeline.retrieval.analysis: {e}")
        errors.append(("pipeline.retrieval.analysis", e))
    
    # Test indexing
    try:
        from pipeline.indexing.build_indices import BGEIndexer, BM25Indexer, load_corpus
        print_success("pipeline.indexing.build_indices")
    except Exception as e:
        print_error(f"pipeline.indexing.build_indices: {e}")
        errors.append(("pipeline.indexing.build_indices", e))
    
    # Test evaluation (may fail if pytrec_eval not installed)
    try:
        from pipeline.evaluation.run_retrieval_eval import compute_results, load_qrels
        print_success("pipeline.evaluation.run_retrieval_eval")
    except Exception as e:
        print_warning(f"pipeline.evaluation.run_retrieval_eval: {e}")
        print_warning("  This is expected if pytrec-eval is not yet installed")
        errors.append(("pipeline.evaluation.run_retrieval_eval", e))
    
    # Test main pipeline (will fail if evaluation fails)
    try:
        from pipeline.run import run_pipeline
        print_success("pipeline.run")
    except Exception as e:
        print_warning(f"pipeline.run: {e}")
        print_warning("  This is expected if pytrec-eval is not yet installed")
        errors.append(("pipeline.run", e))
    
    return errors

def test_data_files():
    """Test existence of all required data files."""
    print_header("Testing Data Files")
    
    domains = ["clapnq", "fiqa", "govt", "cloud"]
    errors = []
    
    for domain in domains:
        # Check corpus
        corpus_path = Path(f"data/passage_level_processed/{domain}/corpus.jsonl")
        if corpus_path.exists():
            size_mb = corpus_path.stat().st_size / (1024 * 1024)
            print_success(f"{domain} corpus: {size_mb:.1f}MB")
        else:
            print_error(f"{domain} corpus not found: {corpus_path}")
            errors.append(("data", f"Missing corpus for {domain}"))
        
        # Check qrels
        qrels_path = Path(f"data/retrieval_tasks/{domain}/qrels/dev.tsv")
        if qrels_path.exists():
            with open(qrels_path) as f:
                lines = sum(1 for _ in f) - 1  # Subtract header
            print_success(f"{domain} qrels: {lines} relevance judgments")
        else:
            print_error(f"{domain} qrels not found: {qrels_path}")
            errors.append(("data", f"Missing qrels for {domain}"))
        
        # Check query files
        query_variants = ["lastturn", "questions", "rewrite", "tasks"]
        for variant in query_variants:
            query_path = Path(f"data/retrieval_tasks/{domain}/{domain}_{variant}.jsonl")
            if query_path.exists():
                with open(query_path) as f:
                    count = sum(1 for _ in f)
                print_success(f"  {domain}_{variant}: {count} queries")
            else:
                print_warning(f"  {domain}_{variant} not found (may be optional)")
    
    return errors

def test_configs():
    """Test configuration files."""
    print_header("Testing Configuration Files")
    
    errors = []
    
    # Test base config
    base_config = Path("configs/base.yaml")
    if base_config.exists():
        print_success(f"Base config: {base_config}")
    else:
        print_error(f"Base config not found: {base_config}")
        errors.append(("config", "Missing base.yaml"))
    
    # Test domain configs
    domains = ["clapnq", "fiqa", "govt", "cloud"]
    for domain in domains:
        domain_config = Path(f"configs/domains/{domain}.yaml")
        if domain_config.exists():
            print_success(f"Domain config: {domain}.yaml")
        else:
            print_error(f"Domain config not found: {domain_config}")
            errors.append(("config", f"Missing {domain}.yaml"))
    
    # Test experiment configs
    experiment_dirs = {
        "0-baselines": [
            "replication_bm25.yaml",
            "replication_bge15.yaml", 
            "replication_elser.yaml",
            "A0_baseline_bm25_fullhist.yaml",
            "A0_baseline_elser_fullhist.yaml",
            "A1_baseline_bgem3_fullhist.yaml"
        ]
    }
    
    for exp_dir, exp_files in experiment_dirs.items():
        for exp_file in exp_files:
            exp_path = Path(f"configs/experiments/{exp_dir}/{exp_file}")
            if exp_path.exists():
                print_success(f"Experiment: {exp_dir}/{exp_file}")
            else:
                print_error(f"Experiment not found: {exp_path}")
                errors.append(("config", f"Missing {exp_dir}/{exp_file}"))
    
    return errors

def test_scripts():
    """Test script files."""
    print_header("Testing Scripts")
    
    errors = []
    
    scripts = [
        "scripts/run_experiment.py",
        "scripts/build_indices.py",
    ]
    
    for script in scripts:
        script_path = Path(script)
        if script_path.exists():
            print_success(f"{script}")
        else:
            print_error(f"Script not found: {script}")
            errors.append(("scripts", f"Missing {script}"))
    
    return errors

def test_config_merging():
    """Test configuration merging works correctly."""
    print_header("Testing Configuration Merging")
    
    sys.path.insert(0, 'src')
    errors = []
    
    try:
        from utils.config_loader import merge_configs
        
        # Test merging baseline config
        config = merge_configs(
            "configs/base.yaml",
            "configs/domains/clapnq.yaml",
            "configs/experiments/0-baselines/replication_bm25.yaml"
        )
        
        # Verify key fields
        assert "data" in config, "Missing 'data' section"
        assert "retrieval" in config, "Missing 'retrieval' section"
        assert "query_file" in config["data"], "Missing query_file"
        assert "qrels_file" in config["data"], "Missing qrels_file"
        
        print_success("Config merging works correctly")
        print(f"  - data.query_mode: {config['data'].get('query_mode', 'NOT SET')}")
        print(f"  - retrieval.type: {config['retrieval'].get('type', 'NOT SET')}")
        print(f"  - retrieval.method: {config['retrieval'].get('method', 'NOT SET')}")
        
    except Exception as e:
        print_error(f"Config merging failed: {e}")
        errors.append(("config_merging", str(e)))
    
    return errors

def main():
    print_header("MT-RAG Code Validation")
    print("Validating code before A100 deployment...\n")
    
    all_errors = []
    
    # Run all tests
    all_errors.extend(test_imports())
    all_errors.extend(test_data_files())
    all_errors.extend(test_configs())
    all_errors.extend(test_scripts())
    all_errors.extend(test_config_merging())
    
    # Summary
    print_header("Validation Summary")
    
    if not all_errors:
        print(f"{GREEN}{BOLD}✓ ALL CHECKS PASSED{RESET}")
        print(f"\n{GREEN}Code is ready for A100 deployment!{RESET}\n")
        return 0
    else:
        print(f"{RED}{BOLD}✗ {len(all_errors)} ERRORS FOUND{RESET}\n")
        
        # Group errors by category
        error_groups = {}
        for category, error in all_errors:
            if category not in error_groups:
                error_groups[category] = []
            error_groups[category].append(error)
        
        for category, errors in error_groups.items():
            print(f"{BOLD}{category.upper()}:{RESET}")
            for error in errors:
                print(f"  - {error}")
            print()
        
        # Check if only pytrec_eval is missing
        pytrec_only = all(
            "pytrec" in str(e).lower() for _, e in all_errors
        )
        
        if pytrec_only:
            print(f"{YELLOW}Note: Only pytrec-eval is missing.{RESET}")
            print(f"{YELLOW}Install it with: pip install pytrec-eval{RESET}\n")
            return 1
        
        print(f"{RED}Please fix these errors before deployment.{RESET}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
