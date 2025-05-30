"""
Logging utilities
"""

import os
import logging
from datetime import datetime
from typing import Optional


def setup_logging(name: str = "dl_arch", 
                 log_file: Optional[str] = None,
                 log_level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        name: Logger name
        log_file: Log file path (optional)
        log_level: Logging level

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_file is provided)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get existing logger by name."""
    return logging.getLogger(name)
