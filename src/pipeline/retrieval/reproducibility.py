"""
Reproducibility and Statistical Analysis Tools.

This module provides utilities to ensure reproducibility (seeding) and 
perform statistical analysis (significance testing, confidence intervals)
on retrieval results.
"""

import random
import logging
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional

# Try to import scipy, handle if missing
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across random, numpy, and torch.
    
    This ensures that stochastic processes (like initialization or 
    GPU non-determinism) produce consistent results.
    
    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior in CuDNN (sacrifices some performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed} for reproducibility.")

def calculate_wilcoxon_significance(
    baseline_scores: List[float], 
    model_scores: List[float]
) -> Dict[str, Any]:
    """
    Perform Wilcoxon Signed-Rank Test to determine if the difference between
    two models is statistically significant.
    
    This is a non-parametric test suitable for comparing paired samples 
    (scores on the same set of queries).
    
    Args:
        baseline_scores: List of metric scores (e.g., NDCG@10) for the baseline.
        model_scores: List of metric scores for the candidate model.
        
    Returns:
        Dictionary with 'statistic', 'p_value', and 'is_significant' (p < 0.05).
    """
    if not SCIPY_AVAILABLE:
        logger.warning("Scipy not installed. Skipping Wilcoxon test.")
        return {"error": "scipy_not_installed"}

    if len(baseline_scores) != len(model_scores):
        raise ValueError("Score lists must have the same length (paired samples).")

    try:
        statistic, p_value = stats.wilcoxon(baseline_scores, model_scores)
        is_significant = p_value < 0.05
        
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "is_significant": is_significant,
            "verdict": "Significant" if is_significant else "Not Significant"
        }
    except ValueError as e:
        # Wilcoxon fails if all differences are zero
        logger.warning(f"Wilcoxon test failed (likely identical scores): {e}")
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "is_significant": False,
            "verdict": "Identical"
        }

def bootstrap_confidence_interval(
    scores: List[float], 
    num_samples: int = 1000, 
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Calculate Confidence Intervals (CI) using Bootstrapping.
    
    Resamples the scores with replacement to estimate the distribution 
    of the mean metric.
    
    Args:
        scores: List of metric scores per query.
        num_samples: Number of bootstrap iterations.
        confidence_level: Desired confidence level (default 0.95).
        
    Returns:
        Dictionary with 'mean', 'ci_lower', 'ci_upper', 'std_dev'.
    """
    scores_np = np.array(scores)
    means = []
    
    for _ in range(num_samples):
        # Resample with replacement
        sample = np.random.choice(scores_np, size=len(scores_np), replace=True)
        means.append(np.mean(sample))
        
    means = np.array(means)
    
    # Calculate percentiles
    alpha = (1.0 - confidence_level) / 2.0
    lower_p = alpha * 100
    upper_p = (1.0 - alpha) * 100
    
    ci_lower = np.percentile(means, lower_p)
    ci_upper = np.percentile(means, upper_p)
    
    return {
        "mean": float(np.mean(scores_np)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "std_dev": float(np.std(means)),
        "confidence_level": confidence_level
    }

def report_stability(
    runs_metrics: List[float]
) -> Dict[str, Any]:
    """
    Report stability statistics across multiple experimental runs.
    
    Args:
        runs_metrics: List of aggregate scores (e.g., mean NDCG) from multiple runs.
        
    Returns:
        Dictionary with mean and standard deviation formatted as string.
    """
    mean_score = np.mean(runs_metrics)
    std_dev = np.std(runs_metrics)
    
    return {
        "mean": float(mean_score),
        "std": float(std_dev),
        "formatted": f"{mean_score:.4f} ± {std_dev:.4f}"
    }

def apply_bonferroni_correction(p_value: float, num_tests: int) -> Dict[str, Any]:
    """
    Apply Bonferroni correction for multiple hypothesis testing.
    
    Args:
        p_value: Original p-value from a single test.
        num_tests: Total number of tests performed (hypotheses tested).
        
    Returns:
        Dictionary with corrected p-value and significance verdict.
    """
    corrected_p = min(1.0, p_value * num_tests)
    is_significant = corrected_p < 0.05
    
    return {
        "original_p": p_value,
        "corrected_p": corrected_p,
        "num_tests": num_tests,
        "is_significant": is_significant
    }
