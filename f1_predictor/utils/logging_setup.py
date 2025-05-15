"""Logging configuration for F1 predictor."""

import os
import logging
from typing import Dict, Any


def setup_logging(config: Dict[str, Any] = None, log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging for the application.
    
    Args:
        config: Configuration dictionary.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        
    Returns:
        Logger instance.
    """
    # Create logger
    logger = logging.getLogger('f1_predictor')
    logger.setLevel(getattr(logging, log_level))
    
    # Remove existing handlers to avoid duplicates on multiple calls
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    # Add file handler if log file is specified in config
    if config and 'project_paths' in config and 'log_file' in config['project_paths']:
        log_file = config['project_paths']['log_file']
        
        # Create directory for log file if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 