import logging
import sys
from pathlib import Path

def setup_logger(name: str, log_file: Path = None, level=logging.INFO):
    """
    Sets up a logger that writes to console and optionally to a file.
    
    Args:
        name: Name of the logger
        log_file: Path to the log file (optional)
        level: Logging level
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Check if handlers already exist to avoid duplicates
    if logger.handlers:
        return logger
        
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler (if path provided)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
