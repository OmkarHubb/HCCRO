"""
Custom Colorized Logging Module
===============================
Provides formatted color logger for terminal output across HCCRO stages.
"""

import logging
import sys

def get_logger(name: str = "HCCRO") -> logging.Logger:
    """Returns a configured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
