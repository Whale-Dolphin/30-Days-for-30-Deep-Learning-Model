#!/usr/bin/env python3
"""
Main entry point for the deep learning architecture framework.

This script provides a unified interface for training and evaluating
different deep learning models using the modular framework.

Usage:
    python main.py --config configs/config.yaml --mode train
    python main.py --config configs/config.yaml --mode eval --checkpoint checkpoint.pt
"""

import sys
from pathlib import Path

import click
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

import dl_arch.data.examples
import dl_arch.models.examples
from dl_arch.utils import set_seed
from dl_arch import (
    DataLoader, Trainer, Evaluator,
    setup_logging, load_config,
    create_model, create_dataset,
    list_models, list_datasets
)

# Set up basic logging first to control import messages
logger.remove()  # Remove default handler
# INFO and above during imports to hide DEBUG
logger.add(sys.stderr, level="INFO")

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Now import after basic logging setup

# Import all examples to register them


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to configuration YAML file"
)
@click.option(
    "--mode",
    type=click.Choice(["train", "eval"]),
    default="train",
    help="Mode: train (default with periodic eval) or eval (test only)"
)
@click.option(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to checkpoint to resume training from or load for evaluation"
)
@click.option(
    "--data-path",
    type=str,
    default=None,
    help="Override data path from config"
)
@click.option(
    "--output-dir",
    type=str,
    default=None,
    help="Override output directory from config"
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Override random seed from config"
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Override device from config (cpu, cuda, auto)"
)
@click.option(
    "--eval-interval",
    type=int,
    default=None,
    help="Override evaluation interval during training (epochs)"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable verbose logging (DEBUG level)"
)
def main(config, mode, checkpoint, data_path, output_dir, seed, device, eval_interval, verbose):
    """Deep Learning Architecture Framework main entry point."""

    # Set up logging level based on verbose flag
    log_level = "DEBUG" if verbose else "INFO"

    # Configure loguru to filter based on log level
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    # Load configuration
    try:
        config_dict = load_config(config)
    except Exception as e:
        logger.error("Error loading config from {}: {}", config, e)
        click.echo(f"Error loading config from {config}: {e}", err=True)
        sys.exit(1)

    # Override config with command line arguments
    if data_path:
        config_dict.setdefault("data", {})["data_path"] = data_path

    if output_dir:
        config_dict.setdefault("experiment", {})["output_dir"] = output_dir

    if seed:
        config_dict.setdefault("experiment", {})["seed"] = seed

    if device:
        config_dict.setdefault("experiment", {})["device"] = device

    if eval_interval:
        config_dict.setdefault("training", {})["eval_interval"] = eval_interval

    # Set up experiment
    experiment_config = config_dict.get("experiment", {})

    # Set random seed
    seed_value = experiment_config.get("seed", 42)
    set_seed(seed_value)

    # Set up output directory and logging
    output_dir_path = experiment_config.get("output_dir", "./outputs")
    config_name = Path(config).stem
    experiment_name = f"{config_name}_experiment"
    full_output_dir = Path(output_dir_path) / experiment_name
    full_output_dir.mkdir(parents=True, exist_ok=True)

    # Set up file logging with same level as console
    log_file = full_output_dir / "experiment.log"
    logger.add(str(log_file), rotation="10 MB", level=log_level)

    logger.info("Starting experiment with config: {}", config)
    logger.info("Mode: {}", mode)
    logger.info("Verbose mode: {}", verbose)
    logger.info("Log level: {}", log_level)
    logger.info("Output directory: {}", full_output_dir)

    click.echo("=" * 60)
    click.echo("DL-Arch Framework")
    click.echo("=" * 60)
    click.echo(f"Config: {config}")
    click.echo(f"Mode: {mode}")
    click.echo(f"Verbose: {verbose}")
    click.echo(f"Output: {full_output_dir}")
    click.echo("=" * 60)

    try:
        # Extract configurations
        model_config = config_dict.get("model", {})
        data_config = config_dict.get("data", {})
        training_config = config_dict.get("training", {})

        model_name = model_config.get("name")
        dataset_name = data_config.get("name")

        if not model_name:
            logger.error("Model name not specified in config")
            click.echo(f"Available models: {list_models()}", err=True)
            sys.exit(1)

        if not dataset_name:
            logger.error("Dataset name not specified in config")
            click.echo(f"Available datasets: {list_datasets()}", err=True)
            sys.exit(1)

        click.echo(f"Creating model: {model_name}")
        click.echo(f"Creating dataset: {dataset_name}")

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
        dataloader_config = config_dict.get("dataloader", {})
        train_loader_config = {
            "batch_size": dataloader_config.get("batch_size", training_config.get("batch_size", 32)),
            "shuffle": dataloader_config.get("shuffle", True),
            "num_workers": dataloader_config.get("num_workers", 0),
            "pin_memory": dataloader_config.get("pin_memory", True)
        }
        train_loader = DataLoader(train_dataset, train_loader_config)

        val_loader = None
        if val_dataset:
            val_loader_config = {
                "batch_size": dataloader_config.get("batch_size", training_config.get("batch_size", 32)),
                "shuffle": False,
                "num_workers": dataloader_config.get("num_workers", 0),
                "pin_memory": dataloader_config.get("pin_memory", True)
            }
            val_loader = DataLoader(val_dataset, val_loader_config)

        # Update training config with experiment settings
        training_config.update(experiment_config)
        training_config["output_dir"] = str(full_output_dir)

        click.echo(f"Model: {model.__class__.__name__}")
        click.echo(f"Training samples: {len(train_dataset)}")
        if val_dataset:
            click.echo(f"Validation samples: {len(val_dataset)}")

        # Set up TensorBoard
        tensorboard_dir = full_output_dir / "tensorboard"
        writer = SummaryWriter(log_dir=str(tensorboard_dir))
        logger.info("TensorBoard logging to: {}", tensorboard_dir)

        if mode == "train":
            # Default mode: Training with periodic evaluation
            click.echo("\nStarting training with periodic evaluation...")

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=training_config
            )

            # Set up trainer with tensorboard writer
            trainer.tensorboard_writer = writer

            # Set up evaluator for periodic evaluation
            eval_loader = val_loader if val_loader else train_loader
            evaluator = Evaluator(
                model=model,
                test_loader=eval_loader,
                config=training_config,
                tensorboard_writer=writer
            )

            # Set trainer's evaluator for periodic evaluation
            trainer.evaluator = evaluator

            if checkpoint:
                click.echo(f"Resuming from checkpoint: {checkpoint}")
                trainer.resume_training(checkpoint)
            else:
                trainer.train()

        elif mode == "eval":
            # Pure evaluation mode: Test model performance only
            click.echo("\nStarting pure evaluation mode...")

            if not checkpoint:
                logger.error("Checkpoint required for evaluation mode")
                click.echo(
                    "Error: --checkpoint required for evaluation mode", err=True)
                sys.exit(1)

            # Use validation set if available, otherwise test set
            eval_loader = val_loader if val_loader else train_loader
            evaluator = Evaluator(
                model=model,
                test_loader=eval_loader,
                config=training_config,
                tensorboard_writer=writer
            )

            # Load checkpoint
            click.echo(f"Loading checkpoint for evaluation: {checkpoint}")
            trainer = Trainer(model, train_loader, val_loader, training_config)
            trainer.load_checkpoint(checkpoint)

            # Run evaluation
            metrics = evaluator.evaluate(epoch=0, prefix="test")

            click.echo("\nEvaluation Results:")
            click.echo("=" * 40)
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    click.echo(f"  {metric}: {value:.4f}")
                else:
                    click.echo(f"  {metric}: {value}")

            # Save evaluation results
            results_file = full_output_dir / "evaluation_results.pt"
            evaluator.save_results(str(results_file))
            click.echo(f"\nEvaluation results saved to: {results_file}")

        # Close tensorboard writer
        writer.close()
        logger.info("TensorBoard writer closed")

    except Exception as e:
        import traceback
        logger.error("Error during execution: {}", e)
        logger.error("Traceback: {}", traceback.format_exc())
        click.echo(f"Error: {e}", err=True)
        if verbose:
            click.echo(f"Traceback: {traceback.format_exc()}", err=True)
        click.echo(f"\nAvailable models: {list_models()}")
        click.echo(f"Available datasets: {list_datasets()}")
        sys.exit(1)

    logger.info("Experiment completed successfully")
    click.echo(f"\nTensorBoard logs available at: {tensorboard_dir}")
    click.echo(f"Run: tensorboard --logdir {tensorboard_dir}")


if __name__ == "__main__":
    main()
