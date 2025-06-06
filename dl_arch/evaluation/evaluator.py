"""
Evaluation utilities and metrics
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from loguru import logger

from ..data import DataLoader
from ..models import BaseModel


class Evaluator:
    """
    Universal evaluator class for deep learning models.

    This class provides comprehensive evaluation metrics and visualization
    tools for different types of tasks (classification, regression, etc.).
    """

    def __init__(self,
                 model: BaseModel,
                 test_loader: DataLoader,
                 config: Dict[str, Any] = None,
                 tensorboard_writer: Optional[SummaryWriter] = None):
        """
        Initialize the evaluator.

        Args:
            model: The model to evaluate
            test_loader: Test data loader
            config: Evaluation configuration
            tensorboard_writer: Optional tensorboard writer for logging metrics
        """
        self.model = model
        self.test_loader = test_loader
        self.config = config or {}
        self.tensorboard_writer = tensorboard_writer

        # Setup device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Task type
        self.task_type = self.config.get("task_type", "classification")

        # Results storage
        self.predictions = []
        self.targets = []
        self.losses = []

        logger.debug("Evaluator initialized with device: {}, task_type: {}",
                     self.device, self.task_type)

    def evaluate(self, epoch: Optional[int] = None, prefix: str = "eval") -> Dict[str, Any]:
        """
        Evaluate the model on test data.

        Args:
            epoch: Current training epoch (for tensorboard logging)
            prefix: Prefix for metric names (e.g., "eval", "test")

        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Starting evaluation with prefix: {}", prefix)

        self.model.eval()
        self.predictions = []
        self.targets = []
        self.losses = []

        criterion = self._get_loss_function()

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f"{prefix.capitalize()}"):
                # Process batch based on format
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

                    # Forward pass
                    outputs = self.model(inputs)

                else:
                    raise ValueError(
                        f"Unsupported batch format: {type(batch)}")

                # Calculate loss
                loss = criterion(outputs, targets)

                # Store results
                self.predictions.append(outputs.cpu())
                self.targets.append(targets.cpu())
                self.losses.append(loss.item())

        # Concatenate all predictions and targets
        self.predictions = torch.cat(self.predictions, dim=0)
        self.targets = torch.cat(self.targets, dim=0)

        # Calculate metrics based on task type
        if self.task_type == "classification":
            metrics = self._calculate_classification_metrics()
        elif self.task_type == "regression":
            metrics = self._calculate_regression_metrics()
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        # Log metrics to tensorboard if writer is available
        if self.tensorboard_writer is not None and epoch is not None:
            self._log_metrics_to_tensorboard(metrics, epoch, prefix)

        logger.info("Evaluation completed. {} metrics: {}",
                    prefix.capitalize(),
                    {k: f"{v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))})

        return metrics

    def _log_metrics_to_tensorboard(self, metrics: Dict[str, Any], epoch: int, prefix: str):
        """
        Log evaluation metrics to tensorboard.

        Args:
            metrics: Dictionary of metrics to log
            epoch: Current epoch number
            prefix: Prefix for metric names
        """
        logger.debug(
            "Logging {} metrics to tensorboard for epoch {}", prefix, epoch)

        # Log all scalar metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_name = f"{prefix}/{key}"
                self.tensorboard_writer.add_scalar(metric_name, value, epoch)
                logger.debug("Logged {}: {:.4f}", metric_name, value)

        # Log confusion matrix for classification tasks
        if self.task_type == "classification" and hasattr(self, 'predictions') and hasattr(self, 'targets'):
            try:
                self._log_confusion_matrix_to_tensorboard(epoch, prefix)
            except Exception as e:
                logger.warning("Failed to log confusion matrix: {}", e)

        # Log additional analysis for classification
        if self.task_type == "classification" and hasattr(self, 'predictions') and hasattr(self, 'targets'):
            try:
                self._log_classification_analysis(epoch, prefix)
            except Exception as e:
                logger.warning("Failed to log classification analysis: {}", e)

    def _log_confusion_matrix_to_tensorboard(self, epoch: int, prefix: str):
        """
        Log confusion matrix to tensorboard for classification tasks.

        Args:
            epoch: Current epoch number  
            prefix: Prefix for metric names
        """
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        import io
        from PIL import Image

        # Convert predictions to class labels
        if self.predictions.dim() == 2:  # Multi-class
            pred_labels = torch.argmax(self.predictions, dim=1)
        else:  # Binary
            pred_labels = (torch.sigmoid(self.predictions) > 0.5).long()

        # Convert to numpy
        pred_labels = pred_labels.numpy()
        true_labels = self.targets.numpy()

        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title(f'{prefix.capitalize()} Confusion Matrix - Epoch {epoch}')

        # Convert plot to image and log to tensorboard
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        image_array = np.array(image)

        # Convert RGBA to RGB if necessary
        if image_array.shape[-1] == 4:
            image_array = image_array[:, :, :3]

        # Log to tensorboard (HWC format)
        self.tensorboard_writer.add_image(
            f'{prefix}/confusion_matrix',
            image_array,
            epoch,
            dataformats='HWC'
        )

        plt.close(fig)
        buf.close()
        logger.debug("Logged confusion matrix to tensorboard")

    def _log_classification_analysis(self, epoch: int, prefix: str):
        """
        Log additional classification analysis to tensorboard.

        Args:
            epoch: Current epoch number
            prefix: Prefix for metric names
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import classification_report, confusion_matrix

        # Convert predictions to class labels
        if self.predictions.dim() == 2:  # Multi-class
            pred_labels = torch.argmax(self.predictions, dim=1)
            pred_probs = torch.softmax(self.predictions, dim=1)
        else:  # Binary
            pred_labels = (torch.sigmoid(self.predictions)
                           > 0.5).long().squeeze()
            pred_probs = torch.sigmoid(self.predictions)

        # Convert to numpy
        pred_labels = pred_labels.numpy()
        true_labels = self.targets.numpy()

        if self.predictions.dim() == 2:
            pred_probs = pred_probs.numpy()

            # Log prediction confidence distribution
            confidence_scores = np.max(pred_probs, axis=1)
            self.tensorboard_writer.add_histogram(
                f'{prefix}/prediction_confidence',
                confidence_scores,
                epoch
            )

            # Log per-class prediction probabilities
            for class_idx in range(pred_probs.shape[1]):
                class_probs = pred_probs[:, class_idx]
                self.tensorboard_writer.add_histogram(
                    f'{prefix}/class_{class_idx}_probabilities',
                    class_probs,
                    epoch
                )

        # Log class distribution
        unique, counts = np.unique(true_labels, return_counts=True)
        for class_idx, count in zip(unique, counts):
            self.tensorboard_writer.add_scalar(
                f'{prefix}/class_{class_idx}_count',
                count,
                epoch
            )

        # Log prediction distribution
        unique_pred, counts_pred = np.unique(pred_labels, return_counts=True)
        for class_idx, count in zip(unique_pred, counts_pred):
            self.tensorboard_writer.add_scalar(
                f'{prefix}/predicted_class_{class_idx}_count',
                count,
                epoch
            )

        logger.debug("Logged classification analysis to tensorboard")

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

    def _calculate_classification_metrics(self) -> Dict[str, Any]:
        """Calculate classification metrics."""
        # Convert predictions to class labels
        if self.predictions.dim() == 2:  # Multi-class
            pred_labels = torch.argmax(self.predictions, dim=1)
        else:  # Binary
            pred_labels = (torch.sigmoid(self.predictions) > 0.5).long()

        # Convert to numpy
        pred_labels = pred_labels.numpy()
        true_labels = self.targets.numpy()

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(true_labels, pred_labels),
            "precision": precision_score(true_labels, pred_labels, average="weighted", zero_division=0),
            "recall": recall_score(true_labels, pred_labels, average="weighted", zero_division=0),
            "f1_score": f1_score(true_labels, pred_labels, average="weighted", zero_division=0),
            "test_loss": np.mean(self.losses)
        }

        # Per-class metrics for multi-class
        if self.predictions.dim() == 2 and self.predictions.size(1) > 2:
            num_classes = self.predictions.size(1)
            for i in range(num_classes):
                precision_per_class = precision_score(true_labels, pred_labels, labels=[
                                                      i], average=None, zero_division=0)
                recall_per_class = recall_score(true_labels, pred_labels, labels=[
                                                i], average=None, zero_division=0)
                f1_per_class = f1_score(true_labels, pred_labels, labels=[
                                        i], average=None, zero_division=0)

                if len(precision_per_class) > 0:
                    metrics[f"precision_class_{i}"] = precision_per_class[0]
                if len(recall_per_class) > 0:
                    metrics[f"recall_class_{i}"] = recall_per_class[0]
                if len(f1_per_class) > 0:
                    metrics[f"f1_class_{i}"] = f1_per_class[0]

        return metrics

    def _calculate_regression_metrics(self) -> Dict[str, Any]:
        """Calculate regression metrics."""
        pred_values = self.predictions.numpy()
        true_values = self.targets.numpy()

        metrics = {
            "mse": mean_squared_error(true_values, pred_values),
            "mae": mean_absolute_error(true_values, pred_values),
            "rmse": np.sqrt(mean_squared_error(true_values, pred_values)),
            "test_loss": np.mean(self.losses)
        }

        # R-squared
        ss_res = np.sum((true_values - pred_values) ** 2)
        ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
        if ss_tot != 0:
            metrics["r2_score"] = 1 - (ss_res / ss_tot)
        else:
            metrics["r2_score"] = 0.0

        return metrics

    def plot_confusion_matrix(self, save_path: Optional[str] = None):
        """
        Plot confusion matrix for classification tasks.

        Args:
            save_path: Path to save the plot (optional)
        """
        if self.task_type != "classification":
            raise ValueError(
                "Confusion matrix is only available for classification tasks")

        try:
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning(
                "matplotlib and/or seaborn not available for plotting")
            return

        # Convert predictions to class labels
        if self.predictions.dim() == 2:  # Multi-class
            pred_labels = torch.argmax(self.predictions, dim=1)
        else:  # Binary
            pred_labels = (torch.sigmoid(self.predictions) > 0.5).long()

        # Convert to numpy
        pred_labels = pred_labels.numpy()
        true_labels = self.targets.numpy()

        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)

        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Confusion matrix saved to: {}", save_path)
        else:
            plt.show()

    def plot_predictions_vs_targets(self, save_path: Optional[str] = None):
        """
        Plot predictions vs targets for regression tasks.

        Args:
            save_path: Path to save the plot (optional)
        """
        if self.task_type != "regression":
            raise ValueError(
                "Predictions vs targets plot is only available for regression tasks")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return

        pred_values = self.predictions.numpy()
        true_values = self.targets.numpy()

        plt.figure(figsize=(8, 6))
        plt.scatter(true_values, pred_values, alpha=0.6)
        plt.plot([true_values.min(), true_values.max()],
                 [true_values.min(), true_values.max()], 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs True Values')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Predictions vs targets plot saved to: {}", save_path)
        else:
            plt.show()

    def get_predictions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get model predictions and targets.

        Returns:
            Tuple of (predictions, targets)
        """
        return self.predictions, self.targets

    def save_results(self, save_path: str):
        """
        Save evaluation results to file.

        Args:
            save_path: Path to save results
        """
        results = {
            "predictions": self.predictions.numpy(),
            "targets": self.targets.numpy(),
            "losses": self.losses
        }
        torch.save(results, save_path)
        logger.info("Evaluation results saved to: {}", save_path)

    def load_results(self, load_path: str):
        """
        Load evaluation results from file.

        Args:
            load_path: Path to load results from
        """
        results = torch.load(load_path)
        self.predictions = torch.from_numpy(results["predictions"])
        self.targets = torch.from_numpy(results["targets"])
        self.losses = results["losses"]
        logger.info("Evaluation results loaded from: {}", load_path)


class MetricTracker:
    """Track metrics across training epochs."""

    def __init__(self):
        self.metrics = {}

    def update(self, metrics: Dict[str, float]):
        """Update metrics with new values."""
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)

    def get_metric(self, name: str) -> Optional[float]:
        """Get the latest value of a metric."""
        if name in self.metrics and self.metrics[name]:
            return self.metrics[name][-1]
        return None

    def get_history(self, name: str) -> Optional[List[float]]:
        """Get the full history of a metric."""
        return self.metrics.get(name, None)

    def get_best(self, name: str, mode: str = "min") -> Tuple[float, int]:
        """
        Get the best value and epoch for a metric.

        Args:
            name: Metric name
            mode: "min" for lowest value, "max" for highest value

        Returns:
            Tuple of (best_value, epoch)
        """
        if name not in self.metrics or not self.metrics[name]:
            raise ValueError(f"Metric '{name}' not found")

        values = self.metrics[name]
        if mode == "min":
            best_value = min(values)
            best_epoch = values.index(best_value)
        elif mode == "max":
            best_value = max(values)
            best_epoch = values.index(best_value)
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'min' or 'max'")

        return best_value, best_epoch

    def plot_metrics(self, metrics: List[str], save_path: Optional[str] = None):
        """
        Plot metric histories.

        Args:
            metrics: List of metric names to plot
            save_path: Path to save the plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return

        fig, axes = plt.subplots(
            len(metrics), 1, figsize=(10, 3 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            if metric in self.metrics:
                axes[i].plot(self.metrics[metric])
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric)
                axes[i].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Metrics plot saved to: {}", save_path)
        else:
            plt.show()

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}

    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all metrics.

        Returns:
            Dictionary with summary stats for each metric
        """
        summary = {}
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    "latest": values[-1],
                    "best": min(values),
                    "worst": max(values),
                    "mean": sum(values) / len(values)
                }
        return summary
