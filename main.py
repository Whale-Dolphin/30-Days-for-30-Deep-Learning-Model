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

import dl_arch.data.examples
import dl_arch.models.examples
from dl_arch.utils import set_seed, get_device
from dl_arch import (
    DataLoader, Trainer, Evaluator,
    setup_logging, load_config,
    create_model, create_dataset,
    list_models, list_datasets
)
import argparse
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Import all examples to register them


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

    try:
        # Extract model and dataset configurations
        model_config = config.get("model", {})
        data_config = config.get("data", {})
        training_config = config.get("training", {})

        model_name = model_config.get("name")
        dataset_name = data_config.get("name")

        if not model_name:
            logger.error("Model name not specified in config")
            print(f"Available models: {list_models()}")
            sys.exit(1)

        if not dataset_name:
            logger.error("Dataset name not specified in config")
            print(f"Available datasets: {list_datasets()}")
            sys.exit(1)

        print(f"Creating model: {model_name}")
        print(f"Creating dataset: {dataset_name}")

        # Create model and datasets
        model = create_model(model_name, model_config)
        train_dataset = create_dataset(
            dataset_name, data_config.get("train", data_config))

        # Create validation dataset if specified
        val_dataset = None
        if "validation" in data_config:
            val_dataset = create_dataset(
                dataset_name, data_config["validation"])

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config.get("batch_size", 32),
            shuffle=training_config.get("shuffle", True),
            num_workers=training_config.get("num_workers", 0)
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=training_config.get("batch_size", 32),
                shuffle=False,
                num_workers=training_config.get("num_workers", 0)
            )

        # Update training config with experiment settings
        training_config.update(experiment_config)

        print(f"Model: {model.__class__.__name__}")
        print(f"Training samples: {len(train_dataset)}")
        if val_dataset:
            print(f"Validation samples: {len(val_dataset)}")

        # Training mode
        if args.mode in ["train", "both"]:
            print("\nStarting training...")
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=training_config
            )

            if args.resume:
                print(f"Resuming from checkpoint: {args.resume}")
                trainer.resume_training(args.resume)
            else:
                trainer.train()

        # Evaluation mode
        if args.mode in ["eval", "both"]:
            print("\nStarting evaluation...")
            eval_loader = val_loader if val_loader else train_loader
            evaluator = Evaluator(model, eval_loader, training_config)

            if args.resume:
                print(f"Loading checkpoint for evaluation: {args.resume}")
                trainer = Trainer(model, train_loader,
                                  val_loader, training_config)
                trainer.load_checkpoint(args.resume)

            metrics = evaluator.evaluate()
            print("\nEvaluation Results:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

    except Exception as e:
        logger.error(f"Error during execution: {e}")
        print(f"Error: {e}")
        print(f"\nAvailable models: {list_models()}")
        print(f"Available datasets: {list_datasets()}")
        sys.exit(1)

    logger.info("Experiment completed successfully")


if __name__ == "__main__":
    main()
