"""
Common utility functions
"""

import torch
import random
import numpy as np
import os
from typing import Any, Dict, List, Optional, Union


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: torch.nn.Module) -> Dict[str, int]:
    """
    Get model size information.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model size information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params
    }


def create_dir(path: str) -> None:
    """
    Create directory if it doesn't exist.

    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get PyTorch device.

    Args:
        device: Device string ("cpu", "cuda", "cuda:0", etc.)

    Returns:
        PyTorch device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return torch.device(device)


def move_to_device(data: Any, device: torch.device) -> Any:
    """
    Move data to specified device.

    Args:
        data: Data to move (tensor, dict, list, etc.)
        device: Target device

    Returns:
        Data moved to device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    else:
        return data


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """
    Unflatten dictionary with nested keys.

    Args:
        d: Dictionary to unflatten
        sep: Separator for nested keys

    Returns:
        Unflattened dictionary
    """
    result = {}
    for key, value in d.items():
        keys = key.split(sep)
        current = result
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    return result


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes to human readable string.

    Args:
        bytes_value: Size in bytes

    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024
    return f"{bytes_value:.2f} PB"


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage.

    Returns:
        Dictionary with memory usage information
    """
    import psutil
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    usage = {
        "rss": memory_info.rss / 1024 / 1024,  # MB
        "vms": memory_info.vms / 1024 / 1024,  # MB
    }
    
    if torch.cuda.is_available():
        usage["gpu_allocated"] = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        usage["gpu_cached"] = torch.cuda.memory_reserved() / 1024 / 1024  # MB
    
    return usage


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer:
    """Simple timer utility."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        import time
        self.start_time = time.time()

    def stop(self):
        import time
        self.end_time = time.time()

    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        end_time = self.end_time if self.end_time is not None else time.time()
        return end_time - self.start_time
