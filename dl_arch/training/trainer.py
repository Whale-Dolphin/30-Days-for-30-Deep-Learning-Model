"""
Training utilities and trainer class
"""

import os
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from loguru import logger

from ..data import DataLoader
from ..models import BaseModel
from ..utils import setup_logging, save_checkpoint, load_checkpoint, get_device


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
        self.num_epochs = int(self.config.get("num_epochs", 100))
        self.learning_rate = float(self.config.get("learning_rate", 1e-3))
        self.weight_decay = float(self.config.get("weight_decay", 1e-4))
        self.gradient_clip = self.config.get("gradient_clip", None)
        if self.gradient_clip is not None:
            self.gradient_clip = float(self.gradient_clip)
        self.save_interval = int(self.config.get("save_interval", 10))
        self.eval_interval = int(self.config.get("eval_interval", 1))
        self.log_interval = int(self.config.get("log_interval", 10))

        # Device configuration
        self.device = get_device(self.config.get("device", "auto"))
        self.model.to(self.device)

        # Optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()

        # Loss function
        self.criterion = self._get_loss_function()

        # Logging and checkpointing
        self.output_dir = self.config.get("output_dir", "./output")
        os.makedirs(self.output_dir, exist_ok=True)

        # TensorBoard setup - can be overridden from main.py
        self.tensorboard_writer = None
        self.use_tensorboard = self.config.get("use_tensorboard", True)
        if self.use_tensorboard and not hasattr(self, 'tensorboard_writer'):
            self.tensorboard_writer = SummaryWriter(
                log_dir=os.path.join(self.output_dir, "tensorboard")
            )

        # Evaluator for periodic evaluation - can be set from main.py
        self.evaluator = None

        # Set up logging
        logger.add(os.path.join(self.output_dir,
                   "training.log"), rotation="10 MB")
        logger.info("Trainer initialized with device: {}", self.device)

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
                momentum=float(self.config.get("momentum", 0.9)),
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
                step_size=int(self.config.get("step_size", 30)),
                gamma=float(self.config.get("gamma", 0.1))
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
                factor=float(self.config.get("factor", 0.5)),
                patience=int(self.config.get("patience", 10))
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

    def _process_batch(self, batch, training=True):
        """
        Input: batch - Dictionary with keys (input_ids, labels, attention_mask) for transformers
                      or Tuple (inputs, targets) for CNN/MLP models
               training (bool) - Whether model is in training mode
        Output: inputs (batch_size, ...) - Input tensor moved to device
                targets (batch_size, ...) - Target tensor moved to device  
                loss (scalar) - Computed loss value for the batch

        Purpose: Process different batch formats (dict/tuple), move tensors to device,
                perform forward pass, compute loss, and optionally perform backward pass
                with gradient clipping during training.

        Mathematical operations:
            Forward: outputs = model(inputs) or model(inputs, mask)
            Loss: loss = criterion(outputs, targets)
            Backward (if training): ∇loss w.r.t. parameters, gradient clipping, optimizer step

        Tensor flow:
            batch -> inputs: (B, ...), targets: (B, ...) -> device
            -> forward -> outputs: (B, C) -> loss: scalar
            -> backward (if training) -> parameter updates
        Where B=batch_size, C=num_classes or output_dim
        """
        logger.debug("Processing batch, training mode: {}", training)

        if isinstance(batch, dict):
            # Handle dictionary format (e.g., transformer data)
            inputs = batch.get("input_ids", batch.get("input", None))
            targets = batch.get("labels", batch.get("target", None))

            if inputs is None or targets is None:
                raise ValueError(
                    "Dict batch must contain 'input_ids'/'input' and 'labels'/'target' keys")

            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            logger.debug(
                "Dictionary batch - inputs shape: {}, targets shape: {}", inputs.shape, targets.shape)

            # Check if we need attention mask
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
                outputs = self.model(inputs, mask=attention_mask)
            else:
                outputs = self.model(inputs)

        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            # Handle tuple format (inputs, targets)
            inputs, targets = batch

            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            logger.debug(
                "Tuple batch - inputs shape: {}, targets shape: {}", inputs.shape, targets.shape)

            # Forward pass
            outputs = self.model(inputs)

        else:
            raise ValueError(f"Unsupported batch format: {type(batch)}")

        # Calculate loss
        loss = self.criterion(outputs, targets)
        logger.debug("Loss computed: {:.4f}", loss.item())

        # Backward pass and optimization (only if training)
        if training:
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping if specified
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip)
                logger.debug("Gradients clipped to: {}", self.gradient_clip)

            self.optimizer.step()
            self.global_step += 1

        return inputs, targets, loss

    def train_epoch(self) -> Dict[str, float]:
        """
        Input: None (uses self.train_loader)
        Output: metrics (Dict[str, float]) - Training metrics for the epoch including:
               - train_loss: Average loss across all batches
               - train_accuracy: Average accuracy (for classification tasks)
               - learning_rate: Current learning rate

        Purpose: Execute one complete training epoch, processing all batches in train_loader.
                Computes and tracks training metrics, logs progress with tqdm.

        Mathematical operations:
            For each batch: loss_i = criterion(model(x_i), y_i)
            Average loss: L = (1/N) * Σ(loss_i) where N = number of batches
            Accuracy: acc = (1/N) * Σ(correct_predictions_i / batch_size_i)

        Tensor operations:
            Iterates through train_loader: [(B, D), (B,)] where B=batch_size, D=input_dim
            -> forward pass -> [(B, C)] where C=num_classes
            -> loss computation -> scalar per batch
            -> aggregation -> epoch metrics
        """
        logger.info("Starting training epoch {}", self.current_epoch + 1)

        self.model.train()
        total_loss = 0.0
        num_batches = 0
        correct_predictions = 0
        total_predictions = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs}",
            leave=False
        )

        for batch_idx, batch in enumerate(progress_bar):
            inputs, targets, loss = self._process_batch(batch, training=True)

            # Calculate accuracy for classification tasks
            if len(targets.shape) == 1 or (len(targets.shape) == 2 and targets.shape[1] == 1):
                # Classification task
                outputs = self.model(inputs)
                if outputs.dim() == 2 and outputs.size(1) > 1:  # Multi-class
                    predictions = torch.argmax(outputs, dim=1)
                    correct = (predictions == targets).sum().item()
                else:  # Binary classification
                    predictions = (torch.sigmoid(outputs)
                                   > 0.5).long().squeeze()
                    correct = (predictions == targets).sum().item()

                correct_predictions += correct
                total_predictions += targets.size(0)

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            current_avg_loss = total_loss / num_batches
            progress_bar.set_postfix({
                'loss': f'{current_avg_loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

            # Log to TensorBoard
            if self.tensorboard_writer and batch_idx % self.log_interval == 0:
                self.tensorboard_writer.add_scalar(
                    'train/batch_loss', loss.item(), self.global_step)
                self.tensorboard_writer.add_scalar(
                    'train/learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)

                # Log batch accuracy if available
                if total_predictions > 0:
                    current_accuracy = correct_predictions / total_predictions
                    self.tensorboard_writer.add_scalar(
                        'train/batch_accuracy', current_accuracy, self.global_step)

        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        metrics = {
            "train_loss": avg_loss,
            "learning_rate": self.optimizer.param_groups[0]['lr']
        }

        # Add accuracy if classification task
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            metrics["train_accuracy"] = accuracy

        # Log epoch metrics to TensorBoard
        if self.tensorboard_writer:
            for key, value in metrics.items():
                self.tensorboard_writer.add_scalar(
                    f'epoch/{key}', value, self.current_epoch)

            # Log gradient norms for debugging
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            self.tensorboard_writer.add_scalar(
                'train/gradient_norm', total_norm, self.current_epoch)

            # Log parameter histograms occasionally
            if (self.current_epoch + 1) % 5 == 0:
                for name, param in self.model.named_parameters():
                    self.tensorboard_writer.add_histogram(
                        f'parameters/{name}', param, self.current_epoch)

        logger.info("Training epoch {} completed. Metrics: {}",
                    self.current_epoch + 1,
                    {k: f"{v:.4f}" for k, v in metrics.items()})

        return metrics

    def validate(self) -> Dict[str, float]:
        """
        Input: None (uses self.val_loader if available)
        Output: metrics (Dict[str, float]) - Validation metrics including:
               - val_loss: Average validation loss
               - val_accuracy: Average validation accuracy (for classification)

        Purpose: Execute validation pass over validation set without gradient updates.
                Used for monitoring training progress and early stopping.

        Mathematical operations:
            For each batch: loss_i = criterion(model(x_i), y_i) [no gradients]
            Average loss: L_val = (1/N) * Σ(loss_i)
            Accuracy: acc_val = (1/N) * Σ(correct_predictions_i / batch_size_i)

        Tensor operations:
            Same as train_epoch but with torch.no_grad() context
            and self.model.eval() mode (disables dropout, batch norm updates)
        """
        if self.val_loader is None:
            return {}

        logger.debug("Starting validation")

        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                inputs, targets, loss = self._process_batch(
                    batch, training=False)

                # Calculate accuracy for classification tasks
                if len(targets.shape) == 1 or (len(targets.shape) == 2 and targets.shape[1] == 1):
                    # Classification task
                    outputs = self.model(inputs)
                    if outputs.dim() == 2 and outputs.size(1) > 1:  # Multi-class
                        predictions = torch.argmax(outputs, dim=1)
                        correct = (predictions == targets).sum().item()
                    else:  # Binary classification
                        predictions = (torch.sigmoid(
                            outputs) > 0.5).long().squeeze()
                        correct = (predictions == targets).sum().item()

                    correct_predictions += correct
                    total_predictions += targets.size(0)

                total_loss += loss.item()
                num_batches += 1

        # Calculate validation metrics
        avg_loss = total_loss / num_batches
        metrics = {"val_loss": avg_loss}

        # Add accuracy if classification task
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            metrics["val_accuracy"] = accuracy

        # Log validation metrics to TensorBoard
        if self.tensorboard_writer:
            for key, value in metrics.items():
                self.tensorboard_writer.add_scalar(
                    f'epoch/{key}', value, self.current_epoch)

        logger.debug("Validation completed. Metrics: {}",
                     {k: f"{v:.4f}" for k, v in metrics.items()})

        return metrics

    def _run_evaluation(self) -> Dict[str, float]:
        """
        Run evaluation using the evaluator if available.

        Returns:
            Dictionary containing evaluation metrics
        """
        if self.evaluator is None:
            return {}

        logger.info("Running evaluation at epoch {}", self.current_epoch + 1)

        try:
            eval_metrics = self.evaluator.evaluate(
                epoch=self.current_epoch,
                prefix="eval"
            )
            logger.info("Evaluation completed. Metrics: {}",
                        {k: f"{v:.4f}" for k, v in eval_metrics.items() if isinstance(v, (int, float))})
            return eval_metrics
        except Exception as e:
            logger.error("Error during evaluation: {}", e)
            return {}

    def train(self) -> None:
        """
        Input: None (uses instance variables and loaders)
        Output: None (saves checkpoints and logs to file/TensorBoard)

        Purpose: Execute complete training loop across all epochs.
                Handles training/validation cycles, checkpointing, learning rate scheduling,
                early stopping, and progress monitoring.

        Training flow:
            For each epoch:
                1. Execute training pass (self.train_epoch())
                2. Execute validation pass (self.validate()) 
                3. Update learning rate scheduler
                4. Run evaluation if evaluator is available
                5. Save checkpoint if best model or at save interval
                6. Check early stopping conditions
        """
        logger.info("Starting training for {} epochs", self.num_epochs)

        start_time = time.time()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Training phase
            train_metrics = self.train_epoch()

            # Validation phase
            val_metrics = validate_metrics = self.validate()

            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}

            # Learning rate scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    # Use validation loss for plateau scheduler
                    metric_value = val_metrics.get(
                        "val_loss", train_metrics.get("train_loss", 0))
                    self.scheduler.step(metric_value)
                else:
                    self.scheduler.step()

            # Run evaluation if evaluator is available and it's time for evaluation
            if self.evaluator is not None and (epoch + 1) % self.eval_interval == 0:
                eval_metrics = self._run_evaluation()
                all_metrics.update(eval_metrics)

            # Check if this is the best model
            current_metric = val_metrics.get(
                "val_loss", train_metrics.get("train_loss", float('inf')))
            is_best = current_metric < self.best_metric
            if is_best:
                self.best_metric = current_metric

            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

            # Log epoch summary
            epoch_time = time.time() - epoch_start_time
            logger.info("Epoch {}/{} completed in {:.2f}s. Metrics: {}",
                        epoch + 1, self.num_epochs, epoch_time,
                        {k: f"{v:.4f}" for k, v in all_metrics.items() if isinstance(v, (int, float))})

        total_time = time.time() - start_time
        logger.info("Training completed in {:.2f}s. Best metric: {:.4f}",
                    total_time, self.best_metric)

        # Close TensorBoard writer if we created it
        if self.tensorboard_writer and self.use_tensorboard:
            self.tensorboard_writer.close()

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save(checkpoint, checkpoint_path)
        logger.info("Checkpoint saved: {}", checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.output_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            logger.info("Best model saved: {}", best_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint.get('best_metric', float('inf'))

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info("Checkpoint loaded from: {}", checkpoint_path)

    def resume_training(self, checkpoint_path: str) -> None:
        """
        Resume training from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.load_checkpoint(checkpoint_path)
        logger.info("Resuming training from epoch {}", self.current_epoch + 1)
        self.train()
