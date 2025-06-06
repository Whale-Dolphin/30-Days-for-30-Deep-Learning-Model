"""
Enhanced data loading utilities with improved preprocessing support.
"""

import torch
from torch.utils.data import DataLoader as TorchDataLoader, IterableDataset
from typing import Any, Dict, Union, List, Optional
from loguru import logger
from .dataset import BaseDataset


class DataLoader:
    """
    Enhanced wrapper around PyTorch DataLoader with intelligent batch processing.
    
    Supports both map-style and iterable datasets with automatic collation
    based on data type detection and preprocessing output.
    """

    def __init__(self, dataset: BaseDataset, config: Dict[str, Any]):
        """
        Initialize the enhanced data loader.

        Args:
            dataset: The dataset to load data from
            config: Configuration dictionary for data loading
        """
        self.dataset = dataset
        self.config = config

        logger.debug("Initializing DataLoader for dataset: {}", 
                    dataset.__class__.__name__)

        # Extract DataLoader parameters from config
        self.batch_size = config.get("batch_size", 32)
        self.num_workers = config.get("num_workers", 0)  # Default to 0 for better compatibility
        self.pin_memory = config.get("pin_memory", torch.cuda.is_available())

        # For map-style datasets
        self.shuffle = config.get("shuffle", True)
        self.drop_last = config.get("drop_last", False)

        # Determine if dataset is iterable
        self.is_iterable = isinstance(dataset, IterableDataset)

        # Determine data type from dataset info
        self.data_info = dataset.get_data_info()
        logger.debug("Dataset info: {}", self.data_info)

        # Create appropriate DataLoader
        if self.is_iterable:
            # For iterable datasets, shuffle and drop_last are not applicable
            self.dataloader = TorchDataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self.smart_collate_fn
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
                collate_fn=self.smart_collate_fn
            )

        logger.info("DataLoader created - batch_size: {}, num_workers: {}, "
                   "is_iterable: {}, shuffle: {}", 
                   self.batch_size, self.num_workers, self.is_iterable, self.shuffle)

    def smart_collate_fn(self, batch: List[Any]) -> Union[torch.Tensor, Dict[str, torch.Tensor], tuple]:
        """
        Intelligent collate function that handles different data formats.
        
        Automatically detects data format and applies appropriate batching:
        - Dictionary format (e.g., transformers): Stacks values by key
        - Tuple format (input, target): Stacks inputs and targets separately
        - Single tensor format: Simple stacking
        
        Args:
            batch: List of samples from the dataset

        Returns:
            Appropriately batched data
        """
        if not batch:
            logger.warning("Empty batch received")
            return None

        logger.debug("Collating batch of size: {}", len(batch))
        
        # Check the type of the first sample
        first_sample = batch[0]
        logger.debug("First sample type: {}, structure: {}", 
                    type(first_sample), 
                    type(first_sample) if not hasattr(first_sample, 'keys') else list(first_sample.keys()) if hasattr(first_sample, 'keys') else str(first_sample)[:50])

        try:
            if isinstance(first_sample, dict):
                # Handle dictionary format (e.g., for transformers)
                return self._collate_dict_batch(batch)
            elif isinstance(first_sample, (tuple, list)):
                # Handle tuple/list format (input, target)
                return self._collate_tuple_batch(batch)
            else:
                # Handle single tensor format
                return self._collate_tensor_batch(batch)
        except Exception as e:
            logger.error("Failed to collate batch: {}", e)
            # Fallback to default collation
            return torch.utils.data.dataloader.default_collate(batch)

    def _collate_dict_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate dictionary format batches."""
        logger.debug("Collating dictionary batch")
        
        batched = {}
        first_sample = batch[0]
        
        for key in first_sample.keys():
            values = [sample[key] for sample in batch]
            
            # Handle different value types
            if isinstance(values[0], torch.Tensor):
                # Stack tensors
                try:
                    batched[key] = torch.stack(values)
                    logger.debug("Stacked key '{}' with shape: {}", key, batched[key].shape)
                except Exception as e:
                    logger.warning("Failed to stack key '{}': {}, trying cat", key, e)
                    batched[key] = torch.cat(values, dim=0)
            else:
                # Convert to tensor
                try:
                    batched[key] = torch.tensor(values)
                    logger.debug("Converted key '{}' to tensor with shape: {}", key, batched[key].shape)
                except Exception as e:
                    logger.warning("Failed to convert key '{}' to tensor: {}", key, e)
                    batched[key] = values  # Keep as list if conversion fails
        
        return batched

    def _collate_tuple_batch(self, batch: List[tuple]) -> tuple:
        """Collate tuple format batches (input, target)."""
        logger.debug("Collating tuple batch")
        
        # Separate inputs and targets
        inputs, targets = zip(*batch)
        
        # Stack inputs
        try:
            if isinstance(inputs[0], torch.Tensor):
                batched_inputs = torch.stack(inputs)
            else:
                batched_inputs = torch.tensor(inputs)
            logger.debug("Batched inputs shape: {}", batched_inputs.shape)
        except Exception as e:
            logger.warning("Failed to batch inputs: {}", e)
            batched_inputs = inputs

        # Stack targets
        try:
            if isinstance(targets[0], torch.Tensor):
                batched_targets = torch.stack(targets)
            else:
                batched_targets = torch.tensor(targets)
            logger.debug("Batched targets shape: {}", batched_targets.shape)
        except Exception as e:
            logger.warning("Failed to batch targets: {}", e)
            batched_targets = targets

        return batched_inputs, batched_targets

    def _collate_tensor_batch(self, batch: List[torch.Tensor]) -> torch.Tensor:
        """Collate single tensor format batches."""
        logger.debug("Collating tensor batch")
        
        try:
            batched = torch.stack(batch)
            logger.debug("Batched tensor shape: {}", batched.shape)
            return batched
        except Exception as e:
            logger.warning("Failed to stack tensors: {}", e)
            return batch

    def __iter__(self):
        """Iterate over batches."""
        return iter(self.dataloader)

    def __len__(self):
        """Get number of batches."""
        if self.is_iterable:
            # For iterable datasets, length might not be available
            try:
                return len(self.dataloader)
            except TypeError:
                logger.warning("Length not available for iterable dataset")
                return None
        return len(self.dataloader)

    def get_batch_info(self) -> Dict[str, Any]:
        """Get information about batching configuration."""
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "is_iterable": self.is_iterable,
            "dataset_type": self.dataset.__class__.__name__,
            "estimated_batches": len(self) if len(self) is not None else "unknown"
        }

    def get_sample_batch(self, num_samples: int = 1) -> Any:
        """
        Get a sample batch for inspection.
        
        Args:
            num_samples: Number of batches to sample
            
        Returns:
            Sample batch(es)
        """
        logger.info("Getting {} sample batch(es)", num_samples)
        
        samples = []
        for i, batch in enumerate(self):
            samples.append(batch)
            if i + 1 >= num_samples:
                break
        
        if len(samples) == 1:
            return samples[0]
        return samples


def create_dataloader(dataset: BaseDataset, config: Dict[str, Any], mode: str = "train") -> DataLoader:
    """
    Factory function to create appropriate dataloader with mode-specific settings.

    Args:
        dataset: Dataset instance
        config: Configuration dictionary
        mode: Mode ("train", "val", "test") - affects default settings

    Returns:
        Enhanced DataLoader instance
    """
    logger.debug("Creating dataloader for mode: {}", mode)
    
    # Create mode-specific config
    dataloader_config = config.get("dataloader", {}).copy()

    # Apply mode-specific defaults
    if mode == "train":
        dataloader_config.setdefault("shuffle", True)
        dataloader_config.setdefault("drop_last", True)
        logger.debug("Applied training mode defaults")
    else:
        dataloader_config.setdefault("shuffle", False)
        dataloader_config.setdefault("drop_last", False)
        logger.debug("Applied evaluation mode defaults")

    # Override config with any mode-specific settings
    mode_specific_config = config.get(f"{mode}_dataloader", {})
    dataloader_config.update(mode_specific_config)

    return DataLoader(dataset, dataloader_config)


def create_train_val_dataloaders(train_dataset: BaseDataset, 
                                val_dataset: BaseDataset, 
                                config: Dict[str, Any]) -> tuple:
    """
    Convenience function to create both training and validation dataloaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    logger.info("Creating training and validation dataloaders")
    
    train_loader = create_dataloader(train_dataset, config, mode="train")
    val_loader = create_dataloader(val_dataset, config, mode="val")
    
    logger.info("Created dataloaders - train batches: {}, val batches: {}", 
               len(train_loader) if len(train_loader) is not None else "unknown",
               len(val_loader) if len(val_loader) is not None else "unknown")
    
    return train_loader, val_loader
