"""
Data preprocessing system with registry support.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union, Tuple, Optional

import torch
import torchvision.transforms as transforms
import numpy as np
from loguru import logger

from dl_arch.registry import register_preprocess


class BasePreprocessor(ABC):
    """
    Abstract base class for all data preprocessors.

    Preprocessors handle data transformation and normalization
    for different data types (image, text, audio, tabular).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor with configuration.

        Args:
            config: Preprocessor configuration dictionary
        """
        self.config = config

    @abstractmethod
    def preprocess(self, data: Any) -> torch.Tensor:
        """
        Preprocess raw data into tensor format.

        Args:
            data: Raw input data

        Returns:
            Preprocessed tensor
        """
        pass

    @abstractmethod
    def get_output_shape(self) -> Tuple[int, ...]:
        """
        Get expected output shape after preprocessing.

        Returns:
            Output tensor shape
        """
        pass

    def postprocess(self, data: torch.Tensor) -> Any:
        """
        Optional postprocessing for model outputs.

        Args:
            data: Model output tensor

        Returns:
            Postprocessed data
        """
        return data


@register_preprocessor("image")
class ImagePreprocessor(BasePreprocessor):
    """Preprocessor for image data."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Image parameters
        self.image_size = tuple(config.get("image_size", (3, 224, 224)))
        self.normalize = config.get("normalize", True)
        self.augment = config.get("augment", False)

        # Setup transforms
        self._setup_transforms()

    def _setup_transforms(self):
        """Setup image transformation pipeline."""
        transform_list = []

        # Resize if specified
        if len(self.image_size) == 3:
            height, width = self.image_size[1], self.image_size[2]
            transform_list.append(transforms.Resize((height, width)))

        # Data augmentation for training
        if self.augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2)
            ])

        # Convert to tensor
        transform_list.append(transforms.ToTensor())

        # Normalization
        if self.normalize:
            # ImageNet normalization by default
            mean = self.config.get("mean", [0.485, 0.456, 0.406])
            std = self.config.get("std", [0.229, 0.224, 0.225])
            transform_list.append(transforms.Normalize(mean=mean, std=std))

        self.transform = transforms.Compose(transform_list)

    def preprocess(self, data: Any) -> torch.Tensor:
        """
        Preprocess image data to tensor format.

        Args:
            data: Input image data (PIL Image, numpy array, or tensor)

        Returns:
            Preprocessed image tensor with shape (C, H, W)
        """
        # logger.debug("Image preprocessing - input type: {}", type(data))

        if isinstance(data, torch.Tensor):
            # Ensure correct shape and dtype
            if data.dim() == 4:  # Batch dimension present
                data = data.squeeze(0)
            if data.dtype != torch.float32:
                data = data.float()

            # Apply normalization if needed
            if self.normalize and data.max() > 1.0:
                data = data / 255.0

            # logger.debug("Image preprocessing - output shape: {}", data.shape)
            return data

        elif isinstance(data, np.ndarray):
            # Convert numpy array to PIL or tensor
            if data.ndim == 3 and data.shape[0] == 3:  # CHW format
                data = torch.from_numpy(data).float()
            elif data.ndim == 3 and data.shape[2] == 3:  # HWC format
                data = torch.from_numpy(data).permute(2, 0, 1).float()
            else:
                raise ValueError(
                    f"Unsupported numpy array shape: {data.shape}")

            if data.max() > 1.0:
                data = data / 255.0

            # logger.debug("Image preprocessing - output shape: {}", data.shape)
            return data

        else:
            # Use torchvision transforms for PIL images
            processed = self.transform(data)
            # logger.debug("Image preprocessing - output shape: {}",
            #              processed.shape)
            return processed

    def get_output_shape(self) -> Tuple[int, ...]:
        """Get output tensor shape."""
        return self.image_size


@register_preprocessor("text")
class TextPreprocessor(BasePreprocessor):
    """Preprocessor for text sequence data."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Text parameters
        self.vocab_size = config.get("vocab_size", 10000)
        self.max_length = config.get("max_length", 512)
        self.padding_value = config.get("padding_value", 0)
        self.truncation = config.get("truncation", True)

    def preprocess(self, data: Any) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Preprocess text data to tensor format.

        Args:
            data: Input text data (token IDs, dictionary with input_ids/attention_mask)

        Returns:
            Preprocessed text tensor or dictionary with input_ids and attention_mask
        """
        # logger.debug("Text preprocessing - input type: {}", type(data))

        if isinstance(data, dict):
            # Handle dictionary format (e.g., from tokenizers)
            processed = {}

            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    tensor = value
                else:
                    tensor = torch.tensor(value, dtype=torch.long)

                # Apply padding/truncation
                if key in ["input_ids", "attention_mask"] and tensor.dim() == 1:
                    tensor = self._pad_truncate(
                        tensor, key == "attention_mask")

                processed[key] = tensor

            # logger.debug("Text preprocessing - output keys: {}",
            #              list(processed.keys()))
            return processed

        elif isinstance(data, (list, np.ndarray)):
            # Convert to tensor
            tensor = torch.tensor(data, dtype=torch.long)
            if tensor.dim() == 1:
                tensor = self._pad_truncate(tensor, is_mask=False)

            # logger.debug("Text preprocessing - output shape: {}", tensor.shape)
            return tensor

        elif isinstance(data, torch.Tensor):
            # Ensure correct dtype and apply padding/truncation
            tensor = data.long()
            if tensor.dim() == 1:
                tensor = self._pad_truncate(tensor, is_mask=False)

            # logger.debug("Text preprocessing - output shape: {}", tensor.shape)
            return tensor

        else:
            raise ValueError(f"Unsupported text data type: {type(data)}")

    def _pad_truncate(self, tensor: torch.Tensor, is_mask: bool = False) -> torch.Tensor:
        """Apply padding or truncation to sequence."""
        current_length = tensor.size(0)

        if current_length > self.max_length and self.truncation:
            # Truncate
            tensor = tensor[:self.max_length]
        elif current_length < self.max_length:
            # Pad
            pad_value = 1 if is_mask else self.padding_value
            padding = torch.full(
                (self.max_length - current_length,), pad_value, dtype=tensor.dtype)
            tensor = torch.cat([tensor, padding])

        return tensor

    def get_output_shape(self) -> Tuple[int, ...]:
        """Get output tensor shape."""
        return (self.max_length,)


@register_preprocessor("tabular")
class TabularPreprocessor(BasePreprocessor):
    """Preprocessor for tabular/structured data."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Tabular parameters
        self.num_features = config.get("num_features", None)
        self.normalize = config.get("normalize", True)
        self.feature_mean = config.get("feature_mean", None)
        self.feature_std = config.get("feature_std", None)

    def preprocess(self, data: Any) -> torch.Tensor:
        """
        Preprocess tabular data to tensor format.

        Args:
            data: Input tabular data (numpy array, list, or tensor)

        Returns:
            Preprocessed tabular tensor with shape (num_features,)
        """
        # logger.debug("Tabular preprocessing - input type: {}", type(data))

        if isinstance(data, torch.Tensor):
            tensor = data.float()
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).float()
        elif isinstance(data, (list, tuple)):
            tensor = torch.tensor(data, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported tabular data type: {type(data)}")

        # Flatten if needed
        if tensor.dim() > 1:
            tensor = tensor.view(-1)

        # Normalization
        if self.normalize and self.feature_mean is not None and self.feature_std is not None:
            mean = torch.tensor(self.feature_mean, dtype=tensor.dtype)
            std = torch.tensor(self.feature_std, dtype=tensor.dtype)
            tensor = (tensor - mean) / (std + 1e-8)

        # logger.debug("Tabular preprocessing - output shape: {}", tensor.shape)
        return tensor

    def get_output_shape(self) -> Tuple[int, ...]:
        """Get output tensor shape."""
        return (self.num_features,) if self.num_features else (-1,)


@register_preprocessor("audio")
class AudioPreprocessor(BasePreprocessor):
    """Preprocessor for audio data."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Audio parameters
        self.sample_rate = config.get("sample_rate", 16000)
        self.n_mels = config.get("n_mels", 80)
        self.n_fft = config.get("n_fft", 512)
        self.hop_length = config.get("hop_length", 256)
        self.max_length = config.get("max_length", None)

    def preprocess(self, data: Any) -> torch.Tensor:
        """
        Preprocess audio data to tensor format.

        Args:
            data: Input audio data (raw waveform or spectrogram)

        Returns:
            Preprocessed audio tensor
        """
        # logger.debug("Audio preprocessing - input type: {}", type(data))

        if isinstance(data, torch.Tensor):
            tensor = data.float()
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).float()
        else:
            tensor = torch.tensor(data, dtype=torch.float32)

        # Ensure tensor is 1D (raw waveform) or 2D (spectrogram)
        if tensor.dim() == 1:
            # Raw waveform - convert to mel spectrogram
            # Note: This is a simplified version, in practice you'd use torchaudio
            tensor = tensor.unsqueeze(0)  # Add channel dimension

        # Apply length constraints
        if self.max_length and tensor.size(-1) > self.max_length:
            tensor = tensor[..., :self.max_length]

        # logger.debug("Audio preprocessing - output shape: {}", tensor.shape)
        return tensor

    def get_output_shape(self) -> Tuple[int, ...]:
        """Get output tensor shape."""
        if self.max_length:
            return (1, self.max_length)
        return (1, -1)


@register_preprocessor("mnist")
class MNISTPreprocessor(BasePreprocessor):
    """Specialized preprocessor for MNIST dataset."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # MNIST specific parameters
        self.image_size = (1, 28, 28)  # MNIST is grayscale 28x28
        self.normalize = config.get("normalize", True)
        self.flatten = config.get("flatten", False)  # For MLP models

    def preprocess(self, data: Any) -> torch.Tensor:
        """
        Preprocess MNIST data to tensor format.

        Args:
            data: Input MNIST data (PIL Image, numpy array, or tensor)

        Returns:
            Preprocessed MNIST tensor
        """
        # logger.debug("MNIST preprocessing - input type: {}", type(data))

        if isinstance(data, torch.Tensor):
            tensor = data.float()
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).float()
        else:
            # Handle PIL Images
            import torchvision.transforms.functional as F
            tensor = F.to_tensor(data)

        # Ensure correct shape
        if tensor.dim() == 2:  # Add channel dimension
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 3 and tensor.size(0) != 1:  # Convert RGB to grayscale
            tensor = tensor.mean(dim=0, keepdim=True)

        # Ensure 28x28 size
        if tensor.shape[-2:] != (28, 28):
            import torch.nn.functional as F
            tensor = F.interpolate(tensor.unsqueeze(
                0), size=(28, 28), mode='bilinear').squeeze(0)

        # Normalization (MNIST pixel values are 0-255 or 0-1)
        if self.normalize:
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            # MNIST normalization
            tensor = (tensor - 0.1307) / 0.3081

        # Flatten for MLP models
        if self.flatten:
            tensor = tensor.view(-1)  # 784 features

        # logger.debug("MNIST preprocessing - output shape: {}", tensor.shape)
        return tensor

    def get_output_shape(self) -> Tuple[int, ...]:
        """Get output tensor shape."""
        if self.flatten:
            return (784,)
        return self.image_size
