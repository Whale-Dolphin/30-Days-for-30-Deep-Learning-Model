#!/usr/bin/env python3
"""
Main entry point for the deep learning architecture framework.

This script provides a unified interface for training and evaluating
different deep learning models using the modular framework.

Usage:
    python main.py --config configs/cnn_config.yaml --mode train
    python main.py --config configs/transformer_config.yaml --mode eval
    python main.py --config configs/mlp_config.yaml --mode train --resume checkpoint.pt
"""

from dl_arch.utils import set_seed, get_device
from dl_arch import (
    BaseDataset, DataLoader, BaseModel, Trainer, Evaluator,
    setup_logging, load_config
)
import argparse
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deep Learning Architecture Framework")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "both"],
        default="train",
        help="Mode: train, eval, or both"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override data path from config"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed from config"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device from config (cpu, cuda, auto)"
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config from {args.config}: {e}")
        sys.exit(1)

    # Override config with command line arguments
    if args.data_path:
        config.setdefault("data", {})["data_path"] = args.data_path

    if args.output_dir:
        config.setdefault("experiment", {})["output_dir"] = args.output_dir

    if args.seed:
        config.setdefault("experiment", {})["seed"] = args.seed

    if args.device:
        config.setdefault("experiment", {})["device"] = args.device

    # Set up experiment
    experiment_config = config.get("experiment", {})

    # Set random seed
    seed = experiment_config.get("seed", 42)
    set_seed(seed)

    # Set up logging
    output_dir = experiment_config.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logging(
        name="main",
        log_file=os.path.join(output_dir, "experiment.log")
    )

    logger.info(f"Starting experiment with config: {args.config}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output directory: {output_dir}")

    print("=" * 60)
    print("DL-Arch Framework")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Mode: {args.mode}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Since we don't have factory functions anymore,
    # users need to implement their own model and dataset classes
    print("\nTo use this framework:")
    print("1. Create your own model class that inherits from BaseModel")
    print("2. Create your own dataset class that inherits from BaseDataset")
    print("3. Instantiate them in your training script")
    print("4. Use the Trainer and Evaluator classes from the framework")
    print("\nExample usage:")
    print("""
from dl_arch import BaseModel, BaseDataset, Trainer, Evaluator
from dl_arch.models import TransformerEncoderLayer  # Use framework components

class MyModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        # Your model implementation
        
    def forward(self, x):
        # Your forward pass
        return x
        
class MyDataset(BaseDataset):
    def __init__(self, config):
        super().__init__()
        # Your dataset implementation
        
    def __getitem__(self, idx):
        # Return your data
        pass

# In your training script:
model = MyModel(config)
dataset = MyDataset(config)
dataloader = DataLoader(dataset, config)
trainer = Trainer(model, dataloader, config)
trainer.train()
""")

    print("\nFramework provides these ready-to-use components:")
    print("- BaseModel: Abstract base class for models")
    print("- BaseDataset: Abstract base class for datasets")
    print("- Trainer: Universal training loop")
    print("- Evaluator: Comprehensive evaluation metrics")
    print("- Transformer components: PositionalEncoding, MultiHeadAttention, etc.")
    print("- Utilities: Logging, checkpointing, configuration management")

    logger.info("Framework overview completed")


if __name__ == "__main__":
    main()
