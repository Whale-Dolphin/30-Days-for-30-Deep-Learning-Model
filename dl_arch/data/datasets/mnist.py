"""
MNIST dataset implementation.
"""

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
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
        # Set MNIST-specific preprocessor if not specified
        if "preprocessor" not in config:
            logger.error("Preprocessor configuration is missing. Exiting...")
            raise ValueError("Preprocessor configuration is required.")

        # Set MNIST-specific defaults
        config.setdefault("num_classes", 10)
        config.setdefault("image_size", (1, 28, 28))

        super().__init__(config)

        # MNIST specific parameters
        self.data_dir = Path(config.get("data_path", "./data"))
        self.download = config.get("download", True)
        self.split = config.get("split", "train")

        logger.info("Initializing MNIST dataset - split: {}, data_dir: {}",
                    self.split, self.data_dir)

        # Load MNIST data
        self._load_mnist_data()

    def _load_mnist_data(self):
        """
        Load MNIST data using torchvision datasets.

        Downloads the dataset if not already present.
        """
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Determine if this is training or test split
        is_train = self.split in ["train", "training"]

        try:
            # Load MNIST dataset
            self.mnist_dataset = datasets.MNIST(
                root=str(self.data_dir),
                train=is_train,
                download=self.download,
                transform=None  # We'll handle preprocessing manually
            )

            # Extract data and targets
            self.data = self.mnist_dataset.data
            self.targets = self.mnist_dataset.targets

            logger.info("Loaded MNIST {} set: {} samples",
                        "train" if is_train else "test", len(self.data))
            logger.debug("Data shape: {}, Targets shape: {}",
                         self.data.shape, self.targets.shape)

        except Exception as e:
            logger.error("Failed to load MNIST dataset: {}", e)
            raise

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.data)

    def _load_raw_data(self, idx: int) -> torch.Tensor:
        """
        Load raw MNIST image data.

        Args:
            idx: Index of the sample

        Returns:
            Raw MNIST image tensor (28x28)
        """
        # Get raw image data (28x28 uint8)
        raw_image = self.data[idx]

        # Convert to float tensor
        if isinstance(raw_image, torch.Tensor):
            return raw_image.float()
        else:
            return torch.tensor(raw_image, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single MNIST sample.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (preprocessed_image, label)
        """
        logger.debug("Getting MNIST sample {}", idx)

        # Load raw image data
        raw_image = self._load_raw_data(idx)

        # Apply preprocessing
        processed_image = self.preprocess(raw_image)

        # Get target label
        target = self.targets[idx]
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.long)

        logger.debug("MNIST sample {} - image shape: {}, target: {}",
                     idx, processed_image.shape, target.item())

        return processed_image, target

    def get_sample_shape(self) -> Tuple[int, ...]:
        """Get the shape of preprocessed samples."""
        if self.preprocessor and hasattr(self.preprocessor, 'flatten') and self.preprocessor.flatten:
            return (784,)  # Flattened for MLP
        return (1, 28, 28)  # Original MNIST shape

    def get_data_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information."""
        info = super().get_data_info()
        info.update({
            "dataset_name": "MNIST",
            "num_classes": 10,
            "image_size": (1, 28, 28),
            "sample_shape": self.get_sample_shape(),
            "data_dir": str(self.data_dir),
            "split": self.split,
            "is_flattened": (self.preprocessor and
                             hasattr(self.preprocessor, 'flatten') and
                             self.preprocessor.flatten)
        })
        return info

    @staticmethod
    def get_class_names() -> list:
        """Get MNIST class names (digits 0-9)."""
        return [str(i) for i in range(10)]

    def visualize_sample(self, idx: int = 0, save_path: str = None):
        """
        Visualize a sample from the dataset.

        Args:
            idx: Sample index to visualize
            save_path: Optional path to save the visualization
        """
        try:
            import matplotlib.pyplot as plt

            # Get raw and processed samples
            raw_image = self._load_raw_data(idx)
            processed_image, target = self[idx]

            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            # Plot raw image
            axes[0].imshow(raw_image.squeeze(), cmap='gray')
            axes[0].set_title(f'Raw Image - Label: {target.item()}')
            axes[0].axis('off')

            # Plot processed image
            if processed_image.dim() == 1:  # Flattened
                processed_display = processed_image.view(28, 28)
            else:
                processed_display = processed_image.squeeze()

            axes[1].imshow(processed_display, cmap='gray')
            axes[1].set_title(
                f'Processed Image - Shape: {processed_image.shape}')
            axes[1].axis('off')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                logger.info("Sample visualization saved to: {}", save_path)
            else:
                plt.show()

        except ImportError:
            logger.warning("matplotlib not available for visualization")
        except Exception as e:
            logger.error("Failed to visualize sample: {}", e)
