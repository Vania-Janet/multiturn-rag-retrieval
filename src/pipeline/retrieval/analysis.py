"""
Analysis tools for Retrieval performance and errors.

Includes:
- Hard Failure Analysis
- Performance by Turn (Late Turn Analysis)
- Latency Monitoring
"""

import time
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class LatencyMonitor:
    """
    Context manager to measure and track retrieval latency.
    
    Usage:
        monitor = LatencyMonitor()
        with monitor:
            retriever.retrieve(query)
        print(f"Avg Latency: {monitor.get_average_latency()}s")
    """
    def __init__(self):
        self.latencies = []
        self._start_time = 0

    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self._start_time
        self.latencies.append(duration)

    def get_average_latency(self) -> float:
        """Get average latency in seconds."""
        return np.mean(self.latencies) if self.latencies else 0.0
    
    def get_p95_latency(self) -> float:
        """Get 95th percentile latency."""
        return np.percentile(self.latencies, 95) if self.latencies else 0.0
    
    def get_p99_latency(self) -> float:
        """Get 99th percentile latency."""
        return np.percentile(self.latencies, 99) if self.latencies else 0.0

    def report(self) -> Dict[str, float]:
        """Return a summary of latency statistics."""
        return {
            "avg_latency_sec": self.get_average_latency(),
            "p95_latency_sec": self.get_p95_latency(),
            "p99_latency_sec": self.get_p99_latency(),
            "total_queries": len(self.latencies)
        }

def analyze_hard_failures(
    results_df: pd.DataFrame, 
    metric_col: str = "ndcg", 
    threshold: float = 0.0,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Identify 'Hard Failures' where the model scored 0.0 (or below threshold).
    
    Args:
        results_df: DataFrame containing query results and metrics.
        metric_col: Column name of the metric to check (e.g., 'ndcg_at_10').
        threshold: Score threshold to consider a failure (default 0.0).
        top_k: Number of failures to return.
        
    Returns:
        DataFrame of top failures.
    """
    # Handle empty DataFrame
    if results_df.empty or metric_col not in results_df.columns:
        logger.warning(f"Empty results or metric column '{metric_col}' not found. Returning empty DataFrame.")
        return pd.DataFrame()
        
    failures = results_df[results_df[metric_col] <= threshold].copy()
    
    logger.info(f"Found {len(failures)} hard failures (score <= {threshold}).")
    
    # Return top K failures (can sort by other metadata if available, or just return head)
    return failures.head(top_k)

def analyze_performance_by_turn(
    results_df: pd.DataFrame,
    metric_col: str = "ndcg",
    turn_col: str = "turn"
) -> pd.DataFrame:
    """
    Analyze performance degradation by conversation turn.
    
    Args:
        results_df: DataFrame containing query results.
        metric_col: Metric to analyze.
        turn_col: Column indicating the turn number.
        
    Returns:
        DataFrame with Mean and Std Dev per turn.
    """
    if turn_col not in results_df.columns:
        logger.warning(f"Turn column '{turn_col}' not found. Attempting to extract from 'id'...")
        # Heuristic: Try to extract turn from ID if it follows format "conv_id::turn_id"
        # This depends on specific ID format of the dataset
        pass

    if turn_col in results_df.columns:
        stats = results_df.groupby(turn_col)[metric_col].agg(['mean', 'std', 'count'])
        return stats
    else:
        raise ValueError(f"Could not find or infer turn column '{turn_col}'.")

def analyze_query_variance(
    results_df: pd.DataFrame,
    group_by_col: str,
    metric_col: str = "ndcg"
) -> pd.DataFrame:
    """
    Calculate variance/std-dev of metrics across different query groups.
    
    Args:
        results_df: DataFrame with results.
        group_by_col: Column to group by (e.g., 'category', 'turn', 'source').
        metric_col: Metric to analyze.
        
    Returns:
        DataFrame with variance statistics.
    """
    if group_by_col not in results_df.columns:
        raise ValueError(f"Group column '{group_by_col}' not found.")
        
    return results_df.groupby(group_by_col)[metric_col].agg(['mean', 'std', 'min', 'max'])

def bootstrap_confidence_interval(
    scores: List[float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Calculate bootstrap confidence interval for a set of scores.
    
    Args:
        scores: List of metric scores.
        n_bootstrap: Number of bootstrap samples.
        confidence_level: Confidence level (default 0.95 for 95% CI).
        
    Returns:
        Dictionary with mean, lower, and upper bounds.
    """
    if not scores:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    
    scores_array = np.array(scores)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores_array, size=len(scores_array), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    return {
        "mean": float(np.mean(scores_array)),
        "lower": float(np.percentile(bootstrap_means, lower_percentile)),
        "upper": float(np.percentile(bootstrap_means, upper_percentile))
    }

def apply_bonferroni_correction(p_value: float, num_tests: int) -> Dict[str, Any]:
    """
    Apply Bonferroni correction for multiple hypothesis testing.
    
    Args:
        p_value: Original p-value.
        num_tests: Number of tests performed.
        
    Returns:
        Dictionary with corrected p-value and significance.
    """
    corrected_p = min(p_value * num_tests, 1.0)
    
    return {
        "original_p_value": p_value,
        "corrected_p_value": corrected_p,
        "num_tests": num_tests,
        "significant_at_0.05": corrected_p < 0.05
    }
