"""
DL-Arch: A Universal Deep Learning Architecture Framework

This framework provides a modular and extensible platform for implementing
various deep learning architectures including CNN, Transformer, MLP, etc.

Key components:
- Data processing pipeline
- Model architecture definitions
- Training and evaluation loops
- Configuration management
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .data import BaseDataset, DataLoader
from .models import (
    BaseModel, PositionalEncoding, MultiHeadAttention,
    TransformerEncoderLayer, TransformerDecoderLayer
)
from .training import Trainer
from .evaluation import Evaluator, MetricTracker
from .utils import setup_logging, load_config

__all__ = [
    "BaseDataset",
    "DataLoader",
    "BaseModel",
    "PositionalEncoding",
    "MultiHeadAttention",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    "Trainer",
    "Evaluator",
    "MetricTracker",
    "setup_logging",
    "load_config",
]
