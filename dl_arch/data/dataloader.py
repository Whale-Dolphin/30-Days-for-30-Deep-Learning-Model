"""
Data loading utilities
"""

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from typing import Any, Dict
from .dataset import BaseDataset


class DataLoader:
    """
    Wrapper around PyTorch DataLoader with additional functionality.
    """

    def __init__(self, dataset: BaseDataset, config: Dict[str, Any]):
        """
        Initialize the data loader.

        Args:
            dataset: The dataset to load data from
            config: Configuration dictionary for data loading
        """
        self.dataset = dataset
        self.config = config

        # Extract DataLoader parameters from config
        self.batch_size = config.get("batch_size", 32)
        self.shuffle = config.get("shuffle", True)
        self.num_workers = config.get("num_workers", 4)
        self.pin_memory = config.get("pin_memory", True)
        self.drop_last = config.get("drop_last", False)

        self.dataloader = TorchDataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        """
        Custom collate function for batching samples.
        Can be overridden by subclasses for custom batching logic.

        Args:
            batch: List of samples from the dataset

        Returns:
            Batched data
        """
        inputs, targets = zip(*batch)
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)
        return inputs, targets

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


def create_dataloader(dataset_type: str, config: Dict[str, Any]) -> DataLoader:
    """
    Factory function to create appropriate dataset and dataloader.

    Args:
        dataset_type: Type of dataset to create
        config: Configuration dictionary

    Returns:
        DataLoader instance
    """
    # This is a placeholder - users should implement their own datasets
    # and register them here or use the factory pattern
    raise NotImplementedError(
        f"Dataset type '{dataset_type}' not implemented. "
        "Please implement your own dataset class inheriting from BaseDataset "
        "and create the dataloader manually."
    )
