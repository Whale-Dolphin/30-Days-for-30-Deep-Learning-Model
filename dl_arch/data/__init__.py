"""
Data processing module initialization.
"""

from .dataset import (
    BaseDataset, ImageDataset, TextDataset, 
    TabularDataset, AudioDataset
)
from .dataloader import DataLoader, create_dataloader, create_train_val_dataloaders
from .preprocessor import (
    BasePreprocessor, PREPROCESSORS, register_preprocessor,
    ImagePreprocessor, TextPreprocessor, TabularPreprocessor,
    AudioPreprocessor, MNISTPreprocessor
)

# Import datasets to register them
from .datasets import mnist

__all__ = [
    "BaseDataset",
    "ImageDataset", 
    "TextDataset",
    "TabularDataset",
    "AudioDataset",
    "DataLoader",
    "create_dataloader",
    "create_train_val_dataloaders",
    "BasePreprocessor",
    "PREPROCESSORS",
    "register_preprocessor",
    "ImagePreprocessor",
    "TextPreprocessor", 
    "TabularPreprocessor",
    "AudioPreprocessor",
    "MNISTPreprocessor",
]
