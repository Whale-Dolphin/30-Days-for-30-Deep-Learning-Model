"""
MNIST dataset implementation.
"""

import torch
import torchvision.datasets as datasets
from typing import Dict, Any, Tuple
from pathlib import Path
from loguru import logger

from ..dataset import ImageDataset
from ...registry import register_dataset


@register_dataset("mnist", dataset_type="map")
class MNISTDataset(ImageDataset):
    """
    MNIST handwritten digit dataset.

    Automatically downloads and loads the MNIST dataset using torchvision.
    Applies specialized MNIST preprocessing.
    """

    def __init__(self, config: Dict[str, Any]):
        # Inherit preprocessor config from parent or set default
        if "preprocessor" not in config:
            # Try to get from data root config or set default
            config["preprocessor"] = {
                "name": "mnist",
                "flatten": config.get("flatten", False),
                "normalize": config.get("normalize", True)
            }

        if config["preprocessor"] is None:
            logger.error("Preprocessor configuration is missing. Exiting...")
            raise ValueError("Preprocessor configuration is required.")

        # Initialize base dataset
        super().__init__(config)

        # MNIST specific parameters
        self.data_path = Path(config.get("data_path", "./data"))
        self.download = config.get("download", True)
        self.split = config.get("split", "train")

        # Validate split
        if self.split not in ["train", "test"]:
            raise ValueError(
                f"Invalid split: {self.split}. Must be 'train' or 'test'")

        # Create data directory
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Load MNIST dataset
        train = self.split == "train"
        self.dataset = datasets.MNIST(
            root=str(self.data_path),
            train=train,
            download=self.download,
            transform=None  # We'll use our preprocessor
        )

        # Set number of classes
        self.num_classes = 10

        logger.info("MNIST dataset loaded: {} samples in {} split",
                    len(self.dataset), self.split)
        logger.debug("MNIST dataset path: {}", self.data_path)
        logger.debug("Preprocessor config: {}", config.get("preprocessor", {}))

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (processed_image, label)
        """
        # Get raw sample from torchvision dataset
        image, label = self.dataset[idx]

        # Apply preprocessing
        processed_image = self.preprocessor.preprocess(image)

        logger.debug("Sample {} - original shape: {}, processed shape: {}",
                     idx, image.size if hasattr(image, 'size') else 'PIL', processed_image.shape)

        return processed_image, label

    def _load_raw_data(self, idx: int) -> Any:
        """
        Load raw data without preprocessing.

        Args:
            idx: Index of the sample

        Returns:
            Raw PIL image from MNIST dataset
        """
        image, _ = self.dataset[idx]
        return image

    def get_sample_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of a preprocessed sample.

        Returns:
            Tuple representing the shape of preprocessed data
        """
        # Get a sample to determine shape
        sample_image, _ = self[0]
        return sample_image.shape

    def get_num_classes(self) -> int:
        """
        Get number of classes.

        Returns:
            Number of classes (10 for MNIST)
        """
        return self.num_classes

    def get_class_names(self):
        """
        Get class names for MNIST digits.

        Returns:
            List of class names (digits 0-9)
        """
        return [str(i) for i in range(10)]

    def visualize_samples(self, num_samples: int = 8, save_path: str = None):
        """
        Visualize random samples from the dataset.

        Args:
            num_samples: Number of samples to visualize
            save_path: Path to save the visualization (optional)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for visualization")
            return

        # Get random samples
        indices = torch.randperm(len(self))[:num_samples]

        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()

        for i, idx in enumerate(indices):
            if i >= len(axes):
                break

            image, label = self[idx.item()]

            # Convert back to displayable format
            if image.dim() == 1:  # Flattened
                image = image.view(28, 28)
            elif image.dim() == 3:  # CHW format
                image = image.squeeze(
                    0) if image.shape[0] == 1 else image.permute(1, 2, 0)

            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f'Label: {label}')
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Sample visualization saved to: {}", save_path)
        else:
            plt.show()

    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset.

        Returns:
            Dictionary containing dataset statistics
        """
        # Sample a few items to compute statistics
        sample_size = min(1000, len(self))
        sample_indices = torch.randperm(len(self))[:sample_size]

        images = []
        labels = []

        for idx in sample_indices:
            image, label = self[idx.item()]
            images.append(image)
            labels.append(label)

        images = torch.stack(images)
        labels = torch.tensor(labels)

        # Compute statistics
        stats = {
            "num_samples": len(self),
            "num_classes": self.num_classes,
            "sample_shape": self.get_sample_shape(),
            "data_mean": images.mean().item(),
            "data_std": images.std().item(),
            "label_distribution": {i: (labels == i).sum().item() for i in range(10)}
        }

        return stats
