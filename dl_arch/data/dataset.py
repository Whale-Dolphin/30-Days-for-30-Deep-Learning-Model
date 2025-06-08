"""
Base dataset classes and interfaces
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union, Optional

import torch
from torch.utils.data import Dataset, IterableDataset
from loguru import logger

from .preprocessor import PREPROCESSORS, BasePreprocessor


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

        # Setup preprocessor
        preprocessor_config = config.get("preprocessor", {})
        preprocessor_name = preprocessor_config.get("name", None)

        if preprocessor_name:
            logger.debug("Creating preprocessor: {}", preprocessor_name)
            self.preprocessor = PREPROCESSORS.create(
                preprocessor_name, preprocessor_config)
        else:
            logger.warning("No preprocessor specified in config")
            self.preprocessor = None

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (input, target) or dictionary with processed data
        """
        pass

    @abstractmethod
    def _load_raw_data(self, idx: int) -> Any:
        """
        Load raw data without preprocessing.

        Args:
            idx: Index of the sample

        Returns:
            Raw data sample
        """
        pass

    def preprocess(self, data: Any) -> torch.Tensor:
        """
        Preprocess raw data using the registered preprocessor.

        Args:
            data: Raw data to preprocess

        Returns:
            Preprocessed tensor
        """
        if self.preprocessor is None:
            logger.warning("No preprocessor available, returning data as-is")
            if isinstance(data, torch.Tensor):
                return data
            return torch.tensor(data)

        return self.preprocessor.preprocess(data)

    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.

        Returns:
            Dictionary containing dataset information
        """
        info = {
            "dataset_size": len(self),
            "split": self.split,
            "data_path": self.data_path,
        }

        if self.preprocessor:
            info["preprocessor"] = self.preprocessor.__class__.__name__
            info["output_shape"] = self.preprocessor.get_output_shape()

        return info

    def get_sample_info(self, idx: int = 0) -> Dict[str, Any]:
        """
        Get information about a sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with sample information
        """
        sample = self[idx]

        if isinstance(sample, tuple):
            input_data, target = sample
            return {
                "input_shape": input_data.shape if hasattr(input_data, 'shape') else None,
                "input_dtype": input_data.dtype if hasattr(input_data, 'dtype') else type(input_data),
                "target_shape": target.shape if hasattr(target, 'shape') else None,
                "target_dtype": target.dtype if hasattr(target, 'dtype') else type(target),
            }
        elif isinstance(sample, dict):
            info = {}
            for key, value in sample.items():
                info[f"{key}_shape"] = value.shape if hasattr(
                    value, 'shape') else None
                info[f"{key}_dtype"] = value.dtype if hasattr(
                    value, 'dtype') else type(value)
            return info
        else:
            return {
                "sample_shape": sample.shape if hasattr(sample, 'shape') else None,
                "sample_dtype": sample.dtype if hasattr(sample, 'dtype') else type(sample),
            }


class BaseIterableDataset(IterableDataset, ABC):
    """
    Abstract base class for all iterable datasets in the framework.

    This class defines the interface that all iterable datasets must implement.
    It provides common functionality for streaming data loading and preprocessing.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the iterable dataset with configuration.

        Args:
            config: Dictionary containing dataset configuration
        """
        self.config = config
        self.data_path = config.get("data_path", "")
        self.split = config.get("split", "train")

        # Setup preprocessor
        preprocessor_config = config.get("preprocessor", {})
        preprocessor_name = preprocessor_config.get("name", None)

        if preprocessor_name:
            logger.debug("Creating preprocessor: {}", preprocessor_name)
            self.preprocessor = PREPROCESSORS.create(
                preprocessor_name, preprocessor_config)
        else:
            logger.warning("No preprocessor specified in config")
            self.preprocessor = None

    @abstractmethod
    def __iter__(self):
        """
        Return an iterator over the dataset.

        Returns:
            Iterator yielding (input, target) tuples or dictionaries with processed data
        """
        pass

    @abstractmethod
    def _load_data_stream(self):
        """
        Load data as a stream without preprocessing.

        Returns:
            Generator yielding raw data samples
        """
        pass

    def preprocess(self, data: Any) -> torch.Tensor:
        """
        Preprocess raw data using the registered preprocessor.

        Args:
            data: Raw data to preprocess

        Returns:
            Preprocessed tensor
        """
        if self.preprocessor is None:
            logger.warning("No preprocessor available, returning data as-is")
            if isinstance(data, torch.Tensor):
                return data
            return torch.tensor(data)

        return self.preprocessor.preprocess(data)

    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.

        Returns:
            Dictionary containing dataset information
        """
        info = {
            "split": self.split,
            "data_path": self.data_path,
            "dataset_type": "iterable",
        }

        if self.preprocessor:
            info["preprocessor"] = self.preprocessor.__class__.__name__
            info["output_shape"] = self.preprocessor.get_output_shape()

        return info

    def get_sample_info(self, sample: Any = None) -> Dict[str, Any]:
        """
        Get information about a sample.

        Args:
            sample: Sample to analyze (if None, will get first sample from iterator)

        Returns:
            Dictionary with sample information
        """
        if sample is None:
            # Get the first sample from the iterator
            iterator = iter(self)
            try:
                sample = next(iterator)
            except StopIteration:
                return {"error": "Dataset is empty"}

        if isinstance(sample, tuple):
            input_data, target = sample
            return {
                "input_shape": input_data.shape if hasattr(input_data, 'shape') else None,
                "input_dtype": input_data.dtype if hasattr(input_data, 'dtype') else type(input_data),
                "target_shape": target.shape if hasattr(target, 'shape') else None,
                "target_dtype": target.dtype if hasattr(target, 'dtype') else type(target),
            }
        elif isinstance(sample, dict):
            info = {}
            for key, value in sample.items():
                info[f"{key}_shape"] = value.shape if hasattr(
                    value, 'shape') else None
                info[f"{key}_dtype"] = value.dtype if hasattr(
                    value, 'dtype') else type(value)
            return info
        else:
            return {
                "sample_shape": sample.shape if hasattr(sample, 'shape') else None,
                "sample_dtype": sample.dtype if hasattr(sample, 'dtype') else type(sample),
            }
