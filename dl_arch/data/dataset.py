"""
Base dataset classes and interfaces
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union, Optional

import torch
from torch.utils.data import Dataset
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


class ImageDataset(BaseDataset):
    """Base class for image datasets."""

    def __init__(self, config: Dict[str, Any]):
        # Set default preprocessor for images
        if "preprocessor" not in config:
            config["preprocessor"] = {"name": "image"}
        super().__init__(config)

        self.num_classes = config.get("num_classes", None)
        self.image_size = config.get("image_size", (3, 224, 224))

    def get_num_classes(self) -> Optional[int]:
        """Get number of classes for classification tasks."""
        return self.num_classes

    def get_image_size(self) -> Tuple[int, ...]:
        """Get expected image size."""
        return self.image_size


class TextDataset(BaseDataset):
    """Base class for text datasets."""

    def __init__(self, config: Dict[str, Any]):
        # Set default preprocessor for text
        if "preprocessor" not in config:
            config["preprocessor"] = {"name": "text"}
        super().__init__(config)

        self.vocab_size = config.get("vocab_size", None)
        self.max_length = config.get("max_length", 512)
        self.num_classes = config.get("num_classes", None)

    def get_vocab_size(self) -> Optional[int]:
        """Get vocabulary size."""
        return self.vocab_size

    def get_max_length(self) -> int:
        """Get maximum sequence length."""
        return self.max_length

    def get_num_classes(self) -> Optional[int]:
        """Get number of classes for classification tasks."""
        return self.num_classes


class TabularDataset(BaseDataset):
    """Base class for tabular datasets."""

    def __init__(self, config: Dict[str, Any]):
        # Set default preprocessor for tabular data
        if "preprocessor" not in config:
            config["preprocessor"] = {"name": "tabular"}
        super().__init__(config)

        self.num_features = config.get("num_features", None)
        self.num_classes = config.get("num_classes", None)

    def get_num_features(self) -> Optional[int]:
        """Get number of input features."""
        return self.num_features

    def get_num_classes(self) -> Optional[int]:
        """Get number of classes for classification tasks."""
        return self.num_classes


class AudioDataset(BaseDataset):
    """Base class for audio datasets."""

    def __init__(self, config: Dict[str, Any]):
        # Set default preprocessor for audio
        if "preprocessor" not in config:
            config["preprocessor"] = {"name": "audio"}
        super().__init__(config)

        self.sample_rate = config.get("sample_rate", 16000)
        self.num_classes = config.get("num_classes", None)

    def get_sample_rate(self) -> int:
        """Get audio sample rate."""
        return self.sample_rate

    def get_num_classes(self) -> Optional[int]:
        """Get number of classes for classification tasks."""
        return self.num_classes
