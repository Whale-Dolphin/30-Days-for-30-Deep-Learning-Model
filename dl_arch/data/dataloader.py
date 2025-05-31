"""
Data loading utilities
"""

import torch
from torch.utils.data import DataLoader as TorchDataLoader, IterableDataset
from typing import Any, Dict, Union
from .dataset import BaseDataset


class DataLoader:
    """
    Wrapper around PyTorch DataLoader with additional functionality.
    Supports both map-style and iterable datasets.
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
        self.num_workers = config.get("num_workers", 4)
        self.pin_memory = config.get("pin_memory", True)

        # For map-style datasets
        self.shuffle = config.get("shuffle", True)
        self.drop_last = config.get("drop_last", False)

        # Determine if dataset is iterable
        self.is_iterable = isinstance(dataset, IterableDataset)

        # Create appropriate DataLoader
        if self.is_iterable:
            # For iterable datasets, shuffle and drop_last are not applicable
            self.dataloader = TorchDataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self.collate_fn
            )
        else:
            # For map-style datasets
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
        Handles different data formats automatically.

        Args:
            batch: List of samples from the dataset

        Returns:
            Batched data
        """
        if not batch:
            return None

        # Check the type of the first sample
        first_sample = batch[0]

        if isinstance(first_sample, dict):
            # Handle dictionary format (e.g., for transformers)
            batched = {}
            for key in first_sample.keys():
                values = [sample[key] for sample in batch]
                if isinstance(values[0], torch.Tensor):
                    batched[key] = torch.stack(values)
                else:
                    batched[key] = torch.tensor(values)
            return batched
        elif isinstance(first_sample, (tuple, list)):
            # Handle tuple/list format (input, target)
            inputs, targets = zip(*batch)

            # Stack inputs
            if isinstance(inputs[0], torch.Tensor):
                inputs = torch.stack(inputs)
            else:
                inputs = torch.tensor(inputs)

            # Stack targets
            if isinstance(targets[0], torch.Tensor):
                targets = torch.stack(targets)
            else:
                targets = torch.tensor(targets)

            return inputs, targets
        else:
            # Handle single tensor format
            return torch.stack(batch)

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        if self.is_iterable:
            # For iterable datasets, length might not be available
            return getattr(self.dataloader, '__len__', lambda: None)()
        return len(self.dataloader)


def create_dataloader(dataset: BaseDataset, config: Dict[str, Any], mode: str = "train") -> DataLoader:
    """
    Factory function to create appropriate dataloader.

    Args:
        dataset: Dataset instance
        config: Configuration dictionary
        mode: Mode ("train", "val", "test") - affects default settings

    Returns:
        DataLoader instance
    """
    # Create mode-specific config
    dataloader_config = config.get("dataloader", {}).copy()

    # Apply mode-specific defaults
    if mode == "train":
        dataloader_config.setdefault("shuffle", True)
        dataloader_config.setdefault("drop_last", True)
    else:
        dataloader_config.setdefault("shuffle", False)
        dataloader_config.setdefault("drop_last", False)

    return DataLoader(dataset, dataloader_config)
