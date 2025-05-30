"""
Base dataset classes and interfaces
"""

import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class BaseDataset(Dataset, ABC):
    """
    Abstract base class for all datasets in the framework.

    This class defines the interface that all datasets must implement.
    It provides common functionality for data loading and preprocessing.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dataset with configuration.

        Args:
            config: Dictionary containing dataset configuration
        """
        self.config = config
        self.data_path = config.get("data_path", "")
        self.split = config.get("split", "train")
        self.transform = config.get("transform", None)

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (input, target)
        """
        pass

    @abstractmethod
    def preprocess(self, data: Any) -> torch.Tensor:
        """
        Preprocess raw data into tensor format.

        Args:
            data: Raw data to preprocess

        Returns:
            Preprocessed tensor
        """
        pass

    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.

        Returns:
            Dictionary containing dataset information
        """
        return {
            "dataset_size": len(self),
            "split": self.split,
            "data_path": self.data_path,
        }
