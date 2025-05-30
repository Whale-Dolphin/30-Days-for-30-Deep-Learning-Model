"""
Configuration utilities
"""

import yaml
import os
from typing import Any, Dict, Optional, Union


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config or {}


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def merge_configs(base_config: Dict[str, Any], 
                 override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.

    Args:
        base_config: Base configuration
        override_config: Configuration to override base

    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_config(config: Dict[str, Any], 
                   required_keys: Optional[list] = None) -> bool:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration to validate
        required_keys: List of required keys

    Returns:
        True if valid, False otherwise
    """
    if required_keys is None:
        required_keys = []

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Required configuration key missing: {key}")

    return True


def get_config_value(config: Dict[str, Any], 
                    key_path: str, 
                    default: Any = None) -> Any:
    """
    Get nested configuration value using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., "model.hidden_size")
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def set_config_value(config: Dict[str, Any], 
                    key_path: str, 
                    value: Any) -> None:
    """
    Set nested configuration value using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., "model.hidden_size")
        value: Value to set
    """
    keys = key_path.split('.')
    current = config
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value
