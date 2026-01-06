#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline.run import run_pipeline
from src.utils.config_loader import merge_configs
from src.utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    setup_logger("run_vllm", level="INFO")
    
    # Paths
    project_root = Path("/workspace/mt-rag-benchmark/task_a_retrieval")
    base_config_path = project_root / "configs/base.yaml"
    domain_config_path = project_root / f"configs/domains/{args.domain}.yaml"
    experiment_config_path = Path(args.config)
    
    if not experiment_config_path.exists():
        experiment_config_path = project_root / args.config
        
    print(f"Merging configs:\n Base: {base_config_path}\n Domain: {domain_config_path}\n Exp: {experiment_config_path}")
    
    config = merge_configs(
        base_config=base_config_path,
        domain_config=domain_config_path,
        experiment_config=experiment_config_path
    )
    
    # Fix output dir
    exp_name = config.get("experiment", {}).get("name", "vllm_experiment")
    output_dir = project_root / "experiments" / exp_name / args.domain
    config["output_dir"] = str(output_dir)
    
    print(f"Output directory: {output_dir}")
    
    run_pipeline(
        config=config,
        output_dir=output_dir,
        domain=args.domain,
        force=args.force
    )

if __name__ == "__main__":
    main()
