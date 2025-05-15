"""Configuration loader module for F1 predictor."""

import os
import yaml
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file. If None, uses default.
    
    Returns:
        Dict containing configuration settings.
    
    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config is invalid.
    """
    if config_path is None:
        # Default config path relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, 'config', 'config.yaml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Basic validation
    _validate_config(config)
    
    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration structure.
    
    Args:
        config: Configuration dictionary to validate.
    
    Raises:
        ValueError: If configuration is invalid.
    """
    required_sections = ['project_paths', 'data_sources', 'feature_engineering', 
                         'models', 'ensemble', 'evaluation']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate models section
    if not any(model for model in config['models'] if config['models'].get(model, {}).get('type')):
        raise ValueError("At least one model must be enabled in configuration")
    
    # Validate ensemble method
    valid_ensemble_methods = ['weighted_average', 'learning_to_rank', 'borda_count']
    if config['ensemble']['method'] not in valid_ensemble_methods:
        raise ValueError(f"Invalid ensemble method. Must be one of: {valid_ensemble_methods}")
    
    # If using weighted_average, validate weights
    if config['ensemble']['method'] == 'weighted_average':
        if 'weights' not in config['ensemble']:
            raise ValueError("Weights must be specified for weighted_average ensemble method") 