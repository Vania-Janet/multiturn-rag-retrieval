#!/usr/bin/env python3
"""
Reproduce baseline experiments for paper.

This script runs the canonical baseline experiments (A0-A1) across all domains
to establish lower bounds for comparison.

Usage:
    # Run all baselines on all domains
    python scripts/reproduce_baselines.py
    
    # Run specific baseline
    python scripts/reproduce_baselines.py --baseline A0_baseline_sparse
    
    # Run on specific domains
    python scripts/reproduce_baselines.py --domain clapnq fiqa
    
    # Dry run
    python scripts/reproduce_baselines.py --dry-run
"""

import argparse
import logging
from pathlib import Path
import sys
import subprocess
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.logger import setup_logger


BASELINES = [
    "A0_baseline_sparse",
    "A1_dense_baseline",
]

DOMAINS = ["clapnq", "fiqa", "govt", "cloud"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce baseline experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--baseline", "-b",
        choices=BASELINES,
        nargs="+",
        default=BASELINES,
        help=f"Baseline(s) to run (default: all)"
    )
    
    parser.add_argument(
        "--domain", "-d",
        choices=DOMAINS,
        nargs="+",
        default=DOMAINS,
        help=f"Domain(s) to run on (default: all)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show commands without executing"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing results"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def run_baseline(baseline, domain, force, dry_run, logger):
    """Run a single baseline experiment."""
    logger.info(f"Running: {baseline} on {domain}")
    
    # Construct command
    cmd = [
        "python", "scripts/run_experiment.py",
        "--experiment", baseline,
        "--domain", domain
    ]
    
    if force:
        cmd.append("--force")
    
    if dry_run:
        logger.info(f"  Command: {' '.join(cmd)}")
        return True
    
    try:
        # Run experiment
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.info(f"  ✓ Completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"  ✗ Failed with exit code {e.returncode}")
        logger.error(f"  Error: {e.stderr}")
        return False


def main():
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger("reproduce_baselines", level=log_level)
    
    baselines = args.baseline
    domains = args.domain
    
    logger.info("="*80)
    logger.info("REPRODUCING BASELINE EXPERIMENTS")
    logger.info("="*80)
    logger.info(f"Baselines: {', '.join(baselines)}")
    logger.info(f"Domains: {', '.join(domains)}")
    logger.info(f"Total runs: {len(baselines) * len(domains)}")
    
    if args.dry_run:
        logger.info("DRY RUN - no actual execution")
    
    logger.info("="*80 + "\n")
    
    # Run baselines
    start_time = datetime.now()
    total_runs = len(baselines) * len(domains)
    current_run = 0
    failed_runs = []
    
    for baseline in baselines:
        for domain in domains:
            current_run += 1
            logger.info(f"\n[{current_run}/{total_runs}] {baseline} / {domain}")
            logger.info("-" * 80)
            
            success = run_baseline(
                baseline=baseline,
                domain=domain,
                force=args.force,
                dry_run=args.dry_run,
                logger=logger
            )
            
            if not success:
                failed_runs.append(f"{baseline}/{domain}")
    
    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Total runs: {total_runs}")
    logger.info(f"Successful: {total_runs - len(failed_runs)}")
    logger.info(f"Failed: {len(failed_runs)}")
    logger.info(f"Duration: {duration:.2f}s")
    
    if failed_runs:
        logger.error("\nFailed runs:")
        for run in failed_runs:
            logger.error(f"  - {run}")
        logger.info("\n✗ Some baselines failed")
        sys.exit(1)
    else:
        logger.info("\n✓ All baselines completed successfully!")
        logger.info("\nResults can be found in: experiments/{baseline}/{domain}/")
        sys.exit(0)


if __name__ == "__main__":
    main()
