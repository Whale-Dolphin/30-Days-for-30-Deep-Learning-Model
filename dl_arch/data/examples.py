"""
Example datasets demonstrating the registry system.
"""

from typing import Dict, Any

import torch
from torch.utils.data import IterableDataset

from dl_arch.data import BaseDataset
from dl_arch import register_dataset


@register_dataset("dummy_image", dataset_type="map")
class DummyImageDataset(BaseDataset):
    """Dummy image dataset for testing CNN models."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config

        # Extract parameters from config
        self.num_samples = config.get("num_samples", 1000)
        self.image_size = config.get("image_size", (3, 32, 32))
        self.num_classes = config.get("num_classes", 10)

        # Generate dummy data
        self.images = torch.randn(self.num_samples, *self.image_size)
        self.labels = torch.randint(0, self.num_classes, (self.num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def preprocess(self, data):
        """Preprocess image data to tensor format."""
        if isinstance(data, torch.Tensor):
            return data
        return torch.tensor(data, dtype=torch.float32)

    def get_sample_shape(self):
        return self.image_size

    def get_num_classes(self):
        return self.num_classes


@register_dataset("dummy_text", dataset_type="map")
class DummyTextDataset(BaseDataset):
    """Dummy text dataset for testing Transformer models."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config

        # Extract parameters from config
        self.num_samples = config.get("num_samples", 1000)
        self.seq_len = config.get("seq_len", 128)
        self.vocab_size = config.get("vocab_size", 10000)
        self.num_classes = config.get("num_classes", 2)

        # Generate dummy data
        self.sequences = torch.randint(
            0, self.vocab_size, (self.num_samples, self.seq_len)
        )
        self.labels = torch.randint(0, self.num_classes, (self.num_samples,))

        # Generate attention masks (random padding)
        self.masks = torch.ones_like(self.sequences, dtype=torch.bool)
        for i in range(self.num_samples):
            # Random sequence length
            actual_len = torch.randint(
                self.seq_len // 2, self.seq_len + 1, (1,)).item()
            self.masks[i, actual_len:] = False

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "input_ids": self.sequences[idx],
            "attention_mask": self.masks[idx],
            "labels": self.labels[idx]
        }

    def preprocess(self, data):
        """Preprocess text data to tensor format."""
        if isinstance(data, dict):
            # Handle dictionary format
            processed = {}
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    processed[key] = value
                else:
                    processed[key] = torch.tensor(value)
            return processed
        elif isinstance(data, torch.Tensor):
            return data
        return torch.tensor(data, dtype=torch.long)

    def get_vocab_size(self):
        return self.vocab_size

    def get_num_classes(self):
        return self.num_classes


@register_dataset("dummy_tabular", dataset_type="map")
class DummyTabularDataset(BaseDataset):
    """Dummy tabular dataset for testing MLP models."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config

        # Extract parameters from config
        self.num_samples = config.get("num_samples", 1000)
        self.num_features = config.get("num_features", 20)
        self.num_classes = config.get("num_classes", 2)

        # Generate dummy data
        self.features = torch.randn(self.num_samples, self.num_features)
        self.labels = torch.randint(0, self.num_classes, (self.num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def preprocess(self, data):
        """Preprocess tabular data to tensor format."""
        if isinstance(data, torch.Tensor):
            return data
        return torch.tensor(data, dtype=torch.float32)

    def get_num_features(self):
        return self.num_features

    def get_num_classes(self):
        return self.num_classes


@register_dataset("streaming_text", dataset_type="iterable")
class StreamingTextDataset(IterableDataset, BaseDataset):
    """Streaming text dataset for large-scale data."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config

        # Extract parameters from config
        self.num_samples = config.get("num_samples", 10000)
        self.seq_len = config.get("seq_len", 128)
        self.vocab_size = config.get("vocab_size", 10000)
        self.num_classes = config.get("num_classes", 2)

    def __iter__(self):
        for _ in range(self.num_samples):
            # Generate sample on-the-fly
            sequence = torch.randint(0, self.vocab_size, (self.seq_len,))
            label = torch.randint(0, self.num_classes, (1,)).item()

            # Generate attention mask
            actual_len = torch.randint(
                self.seq_len // 2, self.seq_len + 1, (1,)).item()
            mask = torch.ones(self.seq_len, dtype=torch.bool)
            mask[actual_len:] = False

            yield {
                "input_ids": sequence,
                "attention_mask": mask,
                "labels": label
            }

    def preprocess(self, data):
        """Preprocess streaming text data to tensor format."""
        if isinstance(data, dict):
            processed = {}
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    processed[key] = value
                else:
                    processed[key] = torch.tensor(value)
            return processed
        elif isinstance(data, torch.Tensor):
            return data
        return torch.tensor(data, dtype=torch.long)

    def get_vocab_size(self):
        return self.vocab_size

    def get_num_classes(self):
        return self.num_classes


@register_dataset("streaming_image", dataset_type="iterable")
class StreamingImageDataset(IterableDataset, BaseDataset):
    """Streaming image dataset for large-scale data."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config

        # Extract parameters from config
        self.num_samples = config.get("num_samples", 10000)
        self.image_size = config.get("image_size", (3, 32, 32))
        self.num_classes = config.get("num_classes", 10)

    def __iter__(self):
        for _ in range(self.num_samples):
            # Generate sample on-the-fly
            image = torch.randn(*self.image_size)
            label = torch.randint(0, self.num_classes, (1,)).item()
            yield image, label

    def preprocess(self, data):
        """Preprocess streaming image data to tensor format."""
        if isinstance(data, torch.Tensor):
            return data
        return torch.tensor(data, dtype=torch.float32)

    def get_sample_shape(self):
        return self.image_size

    def get_num_classes(self):
        return self.num_classes
