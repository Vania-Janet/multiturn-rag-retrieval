"""
Reproducibility utilities for MT-RAG Benchmark.

This module ensures deterministic behavior across all experiments.
"""

import os
import random
import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Set random seed for Python, NumPy, and PyTorch.
    
    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    logger.info(f"✓ Set global random seed to {seed}")


def enable_deterministic_mode(warn_only: bool = False):
    """
    Enable deterministic algorithms in PyTorch.
    
    Args:
        warn_only: If True, only warn about non-deterministic ops instead of failing
    """
    # PyTorch deterministic mode
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    
    # CuDNN settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For some CUDA operations
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    logger.info("✓ Enabled deterministic mode for PyTorch")


def set_num_threads(num_threads: int = 4):
    """
    Set number of threads for CPU operations.
    
    Args:
        num_threads: Number of threads to use
    """
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)  # For reproducibility
    
    # Set environment variables for other libraries
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
    
    logger.info(f"✓ Set num_threads to {num_threads}")


def configure_reproducibility(
    seed: int = 42,
    deterministic: bool = True,
    num_threads: Optional[int] = 4,
    warn_only: bool = False
):
    """
    Configure all reproducibility settings.
    
    This should be called at the start of every experiment script.
    
    Args:
        seed: Random seed
        deterministic: Enable deterministic algorithms
        num_threads: Number of CPU threads (None to skip)
        warn_only: Only warn about non-deterministic ops
    
    Example:
        ```python
        from src.utils.reproducibility import configure_reproducibility
        
        configure_reproducibility(seed=42, deterministic=True)
        ```
    """
    logger.info("=" * 60)
    logger.info("Configuring reproducibility settings")
    logger.info("=" * 60)
    
    set_seed(seed)
    
    if deterministic:
        enable_deterministic_mode(warn_only=warn_only)
    
    if num_threads is not None:
        set_num_threads(num_threads)
    
    # Log device info
    if torch.cuda.is_available():
        logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"✓ CUDA version: {torch.version.cuda}")
    else:
        logger.info("✓ Running on CPU")
    
    logger.info("=" * 60)


def get_worker_init_fn(base_seed: int = 42):
    """
    Get worker initialization function for DataLoader.
    
    Ensures each DataLoader worker has a different but deterministic seed.
    
    Args:
        base_seed: Base random seed
        
    Returns:
        Worker init function
        
    Example:
        ```python
        from src.utils.reproducibility import get_worker_init_fn
        
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            num_workers=4,
            worker_init_fn=get_worker_init_fn(42)
        )
        ```
    """
    def worker_init_fn(worker_id):
        worker_seed = base_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    return worker_init_fn


def log_environment_info():
    """Log key environment information for reproducibility."""
    import platform
    import sys
    
    logger.info("Environment Information:")
    logger.info(f"  Python version: {sys.version}")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  PyTorch version: {torch.__version__}")
    logger.info(f"  NumPy version: {np.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"  CUDA available: True")
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  cuDNN version: {torch.backends.cudnn.version()}")
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info(f"  CUDA available: False")
