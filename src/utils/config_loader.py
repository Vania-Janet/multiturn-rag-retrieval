"""
Configuration loading and merging utilities.

Provides functions to load YAML configurations and merge them hierarchically:
- Base config: Global defaults
- Domain config: Domain-specific overrides
- Experiment config: Experiment-specific overrides
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary containing configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logger.debug(f"Loading config from: {config_path}")
    
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            if config is None:
                config = {}
            logger.debug(f"Loaded config with {len(config)} top-level keys")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML from {config_path}: {e}")
            raise


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.
    
    Values from 'override' take precedence over 'base'.
    Nested dictionaries are merged recursively.
    
    Args:
        base: Base dictionary
        override: Override dictionary
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = deep_merge(result[key], value)
        else:
            # Override value
            result[key] = value
    
    return result


def merge_configs(
    base_config: Union[str, Path],
    domain_config: Union[str, Path],
    experiment_config: Union[str, Path]
) -> Dict[str, Any]:
    """
    Load and merge three configuration files hierarchically.
    
    Merge order (later configs override earlier ones):
    1. base_config: Global defaults
    2. domain_config: Domain-specific settings
    3. experiment_config: Experiment-specific settings
    
    Args:
        base_config: Path to base configuration
        domain_config: Path to domain configuration
        experiment_config: Path to experiment configuration
        
    Returns:
        Merged configuration dictionary
        
    Example:
        >>> config = merge_configs(
        ...     "configs/base.yaml",
        ...     "configs/domains/fiqa.yaml",
        ...     "configs/experiments/baselines/A0_baseline_sparse.yaml"
        ... )
    """
    logger.info(f"Merging configs:")
    logger.info(f"  Base: {base_config}")
    logger.info(f"  Domain: {domain_config}")
    logger.info(f"  Experiment: {experiment_config}")
    
    # Load all configs
    base = load_config(base_config)
    domain = load_config(domain_config)
    experiment = load_config(experiment_config)
    
    # Merge: base <- domain <- experiment
    config = deep_merge(base, domain)
    config = deep_merge(config, experiment)
    
    logger.info(f"✓ Config merged successfully")
    logger.debug(f"Final config keys: {list(config.keys())}")
    
    return config


def validate_config(config: Dict[str, Any], required_keys: list = None) -> None:
    """
    Validate that configuration contains required keys.
    
    Args:
        config: Configuration dictionary to validate
        required_keys: List of required top-level keys
        
    Raises:
        ValueError: If required keys are missing
    """
    if required_keys is None:
        required_keys = ["data", "retrieval", "evaluation"]
    
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        raise ValueError(f"Missing required config keys: {', '.join(missing_keys)}")
    
    logger.debug(f"✓ Config validation passed")


def save_config(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.debug(f"✓ Config saved to: {output_path}")


if __name__ == "__main__":
    # Test config loading
    import sys
    logging.basicConfig(level=logging.DEBUG)
    
    if len(sys.argv) != 4:
        print("Usage: python config_loader.py <base.yaml> <domain.yaml> <experiment.yaml>")
        sys.exit(1)
    
    config = merge_configs(sys.argv[1], sys.argv[2], sys.argv[3])
    print("\n=== MERGED CONFIG ===")
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))
