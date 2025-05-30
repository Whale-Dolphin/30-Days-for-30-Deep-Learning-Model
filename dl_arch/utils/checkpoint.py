"""
Checkpoint utilities
"""

import os
import torch
from typing import Any, Dict, Optional


def save_checkpoint(checkpoint: Dict[str, Any], 
                   checkpoint_path: str) -> None:
    """
    Save model checkpoint.

    Args:
        checkpoint: Checkpoint dictionary
        checkpoint_path: Path to save checkpoint
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path: str, 
                   device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on

    Returns:
        Checkpoint dictionary
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def create_checkpoint(model: torch.nn.Module,
                     optimizer: torch.optim.Optimizer,
                     epoch: int,
                     metrics: Optional[Dict[str, float]] = None,
                     config: Optional[Dict[str, Any]] = None,
                     scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Dict[str, Any]:
    """
    Create checkpoint dictionary.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Training metrics
        config: Configuration dictionary
        scheduler: Learning rate scheduler

    Returns:
        Checkpoint dictionary
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics or {},
        'config': config or {}
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    return checkpoint


def load_model_from_checkpoint(model: torch.nn.Module,
                              checkpoint_path: str,
                              device: Optional[torch.device] = None,
                              strict: bool = True) -> Dict[str, Any]:
    """
    Load model weights from checkpoint.

    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on
        strict: Whether to strictly enforce state dict keys match

    Returns:
        Checkpoint dictionary
    """
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    return checkpoint


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Get path to the latest checkpoint in directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to latest checkpoint or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir)
        if f.endswith('.pth') or f.endswith('.pt')
    ]

    if not checkpoint_files:
        return None

    # Sort by modification time
    checkpoint_files.sort(
        key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)),
        reverse=True
    )

    return os.path.join(checkpoint_dir, checkpoint_files[0])


def cleanup_checkpoints(checkpoint_dir: str, 
                       keep_best: bool = True,
                       keep_latest: int = 5) -> None:
    """
    Clean up old checkpoints, keeping only the best and latest.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_best: Whether to keep best checkpoint
        keep_latest: Number of latest checkpoints to keep
    """
    if not os.path.exists(checkpoint_dir):
        return

    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir)
        if (f.endswith('.pth') or f.endswith('.pt')) and f != 'best_model.pth'
    ]

    if len(checkpoint_files) <= keep_latest:
        return

    # Sort by modification time (newest first)
    checkpoint_files.sort(
        key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)),
        reverse=True
    )

    # Remove old checkpoints
    for checkpoint_file in checkpoint_files[keep_latest:]:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        os.remove(checkpoint_path)
