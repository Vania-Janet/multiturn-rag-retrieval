#!/usr/bin/env python3
"""
Main experiment runner script.

Usage:
    # Run single experiment on one domain
    python scripts/run_experiment.py --experiment A6_hybrid_rerank --domain fiqa
    
    # Run single experiment on all domains
    python scripts/run_experiment.py --experiment A6_hybrid_rerank --domain all
    
    # Run all experiments on one domain
    python scripts/run_experiment.py --experiment all --domain clapnq
    
    # Dry run (validate configs without execution)
    python scripts/run_experiment.py --experiment A6_hybrid_rerank --domain fiqa --dry-run
"""

import argparse
import logging
from pathlib import Path
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.run import run_pipeline
from utils.config_loader import load_config, merge_configs
from utils.logger import setup_logger
from utils.hf_manager import HFManager


DOMAINS = ["clapnq", "fiqa", "govt", "cloud"]

# List of all available experiments (UPDATED: Match actual config file names)
EXPERIMENTS = [
    # Baselines - Replications (last turn)
    "replication_bm25",
    "replication_bge15",
    "replication_bgem3",
    "replication_splade",
    "replication_voyage",
    # Baselines - Full History
    "A0_baseline_bm25_fullhist",
    "A0_baseline_splade_fullhist",
    "A1_baseline_bgem3_fullhist",
    "A1_baseline_voyage_fullhist",
    # Query processing
    "bm25_r1_condensation",
    "bm25_r2_multi",
    "splade_r1_condensation",
    "splade_r3_hyde",
    "bgem3_r1_condensation",
    "bgem3_r2_multi",
    "voyage_r1_condensation",
    "voyage_r2_multi",
    # Hybrid
    "hybrid_splade_bgem3_norewrite",
    "hybrid_splade_bgem3_r1",
    "hybrid_splade_voyage_norewrite",
    "hybrid_splade_voyage_r1",
    "A7_domain_gated",
    # Iterative
    # "A8_iterative_refinement",  # Future Work
    # Reranking
    "hybrid_cohere_norewrite",
    "hybrid_cohere_r1",
    "bgem3_cohere_norewrite",
    # Fine-tuned
    # "A10_finetuned_reranker",  # Future Work
    # "A11_finetuned_splade",  # Future Work
]

# Mapping of experiments to subdirectories (FIXED: Match actual directory names)
EXPERIMENT_DIRS = {
    # Baselines - Replications
    "replication_bm25": "0-baselines",
    "replication_bge15": "0-baselines",
    "replication_bgem3": "0-baselines",
    "replication_splade": "0-baselines",
    "replication_voyage": "0-baselines",
    # Baselines - Full History
    "A0_baseline_bm25_fullhist": "0-baselines",
    "A0_baseline_splade_fullhist": "0-baselines",
    "A1_baseline_bgem3_fullhist": "0-baselines",
    "A1_baseline_voyage_fullhist": "0-baselines",
    # Query processing
    "bm25_r1_condensation": "01-query",
    "bm25_r2_multi": "01-query",
    "splade_r1_condensation": "01-query",
    "splade_r3_hyde": "01-query",
    "bgem3_r1_condensation": "01-query",
    "bgem3_r2_multi": "01-query",
    "voyage_r1_condensation": "01-query",
    "voyage_r2_multi": "01-query",
    # Hybrid
    "hybrid_splade_bgem3_norewrite": "02-hybrid",
    "hybrid_splade_bgem3_r1": "02-hybrid",
    "hybrid_splade_voyage_norewrite": "02-hybrid",
    "hybrid_splade_voyage_r1": "02-hybrid",
    "A7_domain_gated": "02-hybrid",
    # Iterative
    # "A8_iterative_refinement": "04-iterative",  # Future Work
    # Reranking
    "hybrid_cohere_norewrite": "03-rerank",
    "hybrid_cohere_r1": "03-rerank",
    "bgem3_cohere_norewrite": "03-rerank",
    # Fine-tuned
    # "A10_finetuned_reranker": "05-finetune",  # Future Work
    # "A11_finetuned_splade": "05-finetune",  # Future Work
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RAG benchmark experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--experiment", "-e",
        required=True,
        help=f"Experiment name or 'all'. Options: {', '.join(EXPERIMENTS)}"
    )
    
    parser.add_argument(
        "--domain", "-d",
        required=True,
        help=f"Domain name or 'all'. Options: {', '.join(DOMAINS)}"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running experiment"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments"),
        help="Output directory for results (default: experiments/)"
    )
    
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Configuration directory (default: configs/)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing results"
    )

    parser.add_argument(
        "--baseline-path",
        type=str,
        help="Path to baseline results file for significance testing"
    )
    
    return parser.parse_args()


def resolve_experiments(experiment_name):
    """Resolve experiment name to list of experiments."""
    if experiment_name == "all":
        return EXPERIMENTS
    elif experiment_name in EXPERIMENTS:
        return [experiment_name]
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")


def resolve_domains(domain_name):
    """Resolve domain name to list of domains."""
    if domain_name == "all":
        return DOMAINS
    elif domain_name in DOMAINS:
        return [domain_name]
    else:
        raise ValueError(f"Unknown domain: {domain_name}")


def main():
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger("run_experiment", level=log_level)
    
    # Resolve experiments and domains
    experiments = resolve_experiments(args.experiment)
    domains = resolve_domains(args.domain)
    
    logger.info(f"Running {len(experiments)} experiment(s) on {len(domains)} domain(s)")
    logger.info(f"Experiments: {', '.join(experiments)}")
    logger.info(f"Domains: {', '.join(domains)}")
    
    # Initialize HF Manager
    hf_manager = HFManager()

    # Run experiments
    total_runs = len(experiments) * len(domains)
    current_run = 0
    failed_runs = []
    
    for experiment in experiments:
        for domain in domains:
            current_run += 1
            logger.info(f"\n{'='*80}")
            logger.info(f"Run {current_run}/{total_runs}: {experiment} on {domain}")
            logger.info(f"{'='*80}\n")
            
            try:
                # Find experiment config in subdirectory
                experiment_subdir = EXPERIMENT_DIRS.get(experiment, "")
                experiment_config_path = args.config_dir / "experiments" / experiment_subdir / f"{experiment}.yaml"
                
                # Load and merge configs
                config = merge_configs(
                    base_config=args.config_dir / "base.yaml",
                    domain_config=args.config_dir / "domains" / f"{domain}.yaml",
                    experiment_config=experiment_config_path
                )
                
                # Handle query_file_suffix override for replication experiments
                if "query_file_suffix" in config.get("data", {}):
                    suffix = config["data"]["query_file_suffix"]
                    # Replace 'questions' with the specified suffix in query_file path
                    original_path = config["data"]["query_file"]
                    config["data"]["query_file"] = original_path.replace("questions", suffix)
                    logger.info(f"Using query file: {config['data']['query_file']}")
                
                # FIX: Dynamically adjust index_path if it's hardcoded to clapnq but domain is different
                if "retrieval" in config and "index_path" in config["retrieval"]:
                    index_path = config["retrieval"]["index_path"]
                    if "clapnq" in index_path and domain != "clapnq":
                         new_index_path = index_path.replace("clapnq", domain)
                         config["retrieval"]["index_path"] = new_index_path
                         logger.info(f"Fixed index_path for {domain}: {new_index_path}")

                # Setup output directory
                output_dir = args.output_dir / experiment / domain
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save resolved config
                config_path = args.output_dir / experiment / "config_resolved.yaml"
                if not config_path.exists() or args.force:
                    with open(config_path, "w") as f:
                        import yaml
                        yaml.dump(config, f, default_flow_style=False)
                
                if args.dry_run:
                    logger.info(f"✓ Configuration validated for {experiment}/{domain}")
                    logger.info(f"  Output would be written to: {output_dir}")
                else:
                    # Run pipeline
                    logger.info(f"Starting pipeline execution...")
                    start_time = datetime.now()
                    
                    run_pipeline(
                        config=config,
                        domain=domain,
                        output_dir=output_dir,
                        force=args.force,
                        baseline_path=args.baseline_path
                    )
                    
                    duration = (datetime.now() - start_time).total_seconds()
                    logger.info(f"✓ Completed in {duration:.2f}s")
                    
                    # Upload results to Hugging Face if enabled
                    if hf_manager.enabled:
                        hf_manager.upload_directory(output_dir)
                    
            except Exception as e:
                logger.error(f"✗ Failed: {experiment}/{domain}")
                logger.error(f"  Error: {str(e)}", exc_info=args.verbose)
                failed_runs.append(f"{experiment}/{domain}")
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total runs: {total_runs}")
    logger.info(f"Successful: {total_runs - len(failed_runs)}")
    logger.info(f"Failed: {len(failed_runs)}")
    
    if failed_runs:
        logger.error("\nFailed runs:")
        for run in failed_runs:
            logger.error(f"  - {run}")
        sys.exit(1)
    else:
        logger.info("\n✓ All runs completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
