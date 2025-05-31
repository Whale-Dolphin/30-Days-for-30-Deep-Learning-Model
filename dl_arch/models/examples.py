"""
Example models demonstrating the registry system.
"""

import torch
import torch.nn as nn
from dl_arch.models import BaseModel
from dl_arch.models.transformer import TransformerEncoderLayer
from dl_arch import register_model


@register_model("simple_cnn")
class SimpleCNN(BaseModel):
    """Simple CNN model for image classification."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Extract parameters from config
        in_channels = config.get("in_channels", 3)
        num_classes = config.get("num_classes", 10)
        hidden_channels = config.get("hidden_channels", [32, 64, 128])

        # Build CNN layers
        layers = []
        prev_channels = in_channels

        for channels in hidden_channels:
            layers.extend([
                nn.Conv2d(prev_channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ])
            prev_channels = channels

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(prev_channels, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_model_info(self):
        return {
            "model_type": "CNN",
            "parameters": sum(p.numel() for p in self.parameters()),
            "config": self.config
        }


@register_model("simple_mlp")
class SimpleMLP(BaseModel):
    """Simple MLP model."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Extract parameters from config
        input_size = config.get("input_size", 784)
        hidden_sizes = config.get("hidden_sizes", [256, 128])
        output_size = config.get("output_size", 10)
        dropout = config.get("dropout", 0.2)

        # Build MLP layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)

    def get_model_info(self):
        return {
            "model_type": "MLP",
            "parameters": sum(p.numel() for p in self.parameters()),
            "config": self.config
        }


@register_model("simple_transformer")
class SimpleTransformer(BaseModel):
    """Simple Transformer model."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Extract parameters from config
        vocab_size = config.get("vocab_size", 10000)
        d_model = config.get("d_model", 512)
        num_heads = config.get("num_heads", 8)
        num_layers = config.get("num_layers", 6)
        d_ff = config.get("d_ff", 2048)
        max_seq_len = config.get("max_seq_len", 512)
        dropout = config.get("dropout", 0.1)
        num_classes = config.get("num_classes", vocab_size)

        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(
            torch.randn(max_seq_len, d_model) * 0.02
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len)
        seq_len = x.size(1)

        # Embedding + positional encoding
        x = self.embedding(x) * (self.config.get("d_model", 512) ** 0.5)
        x = x + self.pos_encoding[:seq_len]
        x = self.dropout(x)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)

        # Classification (use mean pooling)
        if mask is not None:
            # Masked mean pooling
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            # Simple mean pooling
            x = x.mean(dim=1)

        return self.classifier(x)

    def get_model_info(self):
        return {
            "model_type": "Transformer",
            "parameters": sum(p.numel() for p in self.parameters()),
            "config": self.config
        }
