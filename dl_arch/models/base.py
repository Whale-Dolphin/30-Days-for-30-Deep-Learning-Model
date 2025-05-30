"""
Base model class
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models in the framework.

    This class defines the interface that all models must implement.
    It provides common functionality for model initialization and forward pass.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model with configuration.

        Args:
            config: Dictionary containing model configuration
        """
        super().__init__()
        self.config = config
        self.input_dim = config.get("input_dim", None)
        self.output_dim = config.get("output_dim", None)
        self.hidden_dim = config.get("hidden_dim", 512)
        self.dropout = config.get("dropout", 0.1)

    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor
            *args, **kwargs: Additional arguments for specific models

        Returns:
            Output tensor
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel()
                               for p in self.parameters() if p.requires_grad)

        return {
            "model_name": self.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }

    def freeze_layers(self, layer_names: list):
        """
        Freeze specified layers.

        Args:
            layer_names: List of layer names to freeze
        """
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False

    def unfreeze_layers(self, layer_names: list):
        """
        Unfreeze specified layers.

        Args:
            layer_names: List of layer names to unfreeze
        """
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
