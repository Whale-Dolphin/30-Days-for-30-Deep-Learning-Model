"""
Training utilities and trainer class
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, Optional, Callable
import os
import time
from tqdm import tqdm
import logging

from ..data import DataLoader
from ..models import BaseModel
from ..utils import setup_logging, save_checkpoint, load_checkpoint


class Trainer:
    """
    Universal trainer class for deep learning models.

    This class provides a common training interface that can be used
    with any model and dataset that follows the framework's interface.
    """

    def __init__(self,
                 model: BaseModel,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 config: Dict[str, Any] = None):
        """
        Initialize the trainer.

        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}

        # Training parameters
        self.num_epochs = self.config.get("num_epochs", 100)
        self.learning_rate = self.config.get("learning_rate", 1e-3)
        self.weight_decay = self.config.get("weight_decay", 1e-4)
        self.gradient_clip = self.config.get("gradient_clip", None)
        self.save_interval = self.config.get("save_interval", 10)
        self.eval_interval = self.config.get("eval_interval", 1)
        self.log_interval = self.config.get("log_interval", 10)

        # Device configuration
        self.device = torch.device(
            self.config.get(
                "device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        # Optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()

        # Loss function
        self.criterion = self._get_loss_function()

        # Logging and checkpointing
        self.output_dir = self.config.get("output_dir", "./output")
        os.makedirs(self.output_dir, exist_ok=True)

        self.use_tensorboard = self.config.get("use_tensorboard", True)
        if self.use_tensorboard:
            self.writer = SummaryWriter(
                log_dir=os.path.join(self.output_dir, "tensorboard")
            )

        # Set up logging
        self.logger = setup_logging(
            name="trainer",
            log_file=os.path.join(self.output_dir, "training.log")
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')

    def _setup_optimizer(self):
        """Setup optimizer based on configuration."""
        optimizer_name = self.config.get("optimizer", "adam").lower()

        if optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=self.config.get("momentum", 0.9),
                weight_decay=self.weight_decay
            )
        elif optimizer_name == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _setup_scheduler(self):
        """Setup learning rate scheduler based on configuration."""
        scheduler_name = self.config.get("scheduler", None)

        if scheduler_name is None:
            self.scheduler = None
        elif scheduler_name == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get("step_size", 30),
                gamma=self.config.get("gamma", 0.1)
            )
        elif scheduler_name == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs
            )
        elif scheduler_name == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.get("factor", 0.5),
                patience=self.config.get("patience", 10)
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    def _get_loss_function(self):
        """Get loss function based on configuration."""
        loss_name = self.config.get("loss", "cross_entropy").lower()

        if loss_name == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif loss_name == "mse":
            return nn.MSELoss()
        elif loss_name == "mae":
            return nn.L1Loss()
        elif loss_name == "bce":
            return nn.BCELoss()
        elif loss_name == "bce_with_logits":
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs}"
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self.model.prepare_batch(batch, self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch["input"])
            loss = self.criterion(outputs, batch["target"])

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Logging
            if batch_idx % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss / num_batches:.4f}',
                    'lr': f'{current_lr:.6f}'
                })

                if self.use_tensorboard:
                    self.writer.add_scalar(
                        'Train/Loss', loss.item(), self.global_step)
                    self.writer.add_scalar(
                        'Train/LR', current_lr, self.global_step)

        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}

    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = self.model.prepare_batch(batch, self.device)

                # Forward pass
                outputs = self.model(batch["input"])
                loss = self.criterion(outputs, batch["target"])

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}

    def train(self) -> None:
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            self.logger.info(
                f"Validation samples: {len(self.val_loader.dataset)}")

        start_time = time.time()

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # Training
            train_metrics = self.train_epoch()

            # Validation
            val_metrics = {}
            if epoch % self.eval_interval == 0:
                val_metrics = self.validate()

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if val_metrics and "loss" in val_metrics:
                        self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Logging
            log_msg = f"Epoch {epoch + 1}/{self.num_epochs} - "
            log_msg += f"Train Loss: {train_metrics['loss']:.4f}"
            if val_metrics:
                log_msg += f" - Val Loss: {val_metrics['loss']:.4f}"

            self.logger.info(log_msg)

            # TensorBoard logging
            if self.use_tensorboard:
                self.writer.add_scalar(
                    'Epoch/Train_Loss', train_metrics['loss'], epoch)
                if val_metrics:
                    self.writer.add_scalar(
                        'Epoch/Val_Loss', val_metrics['loss'], epoch)

            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(epoch + 1)

            # Save best model
            if val_metrics and val_metrics['loss'] < self.best_metric:
                self.best_metric = val_metrics['loss']
                self.save_checkpoint(epoch + 1, is_best=True)

        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")

        if self.use_tensorboard:
            self.writer.close()

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_metric': self.best_metric
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.output_dir, f"checkpoint_epoch_{epoch}.pth")
        save_checkpoint(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.output_dir, "best_model.pth")
            save_checkpoint(checkpoint, best_path)
            self.logger.info(f"Saved best model at epoch {epoch}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = load_checkpoint(checkpoint_path, self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint.get('best_metric', float('inf'))

        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def resume_training(self, checkpoint_path: str) -> None:
        """Resume training from checkpoint."""
        self.load_checkpoint(checkpoint_path)

        # Continue training from the next epoch
        remaining_epochs = self.num_epochs - self.current_epoch
        if remaining_epochs > 0:
            self.num_epochs = remaining_epochs
            self.train()
        else:
            self.logger.info("Training already completed")
