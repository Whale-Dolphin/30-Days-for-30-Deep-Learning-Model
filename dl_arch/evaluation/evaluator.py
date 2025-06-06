"""
Evaluation utilities and metrics
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

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
                 config: Dict[str, Any] = None):
        """
        Initialize the evaluator.

        Args:
            model: The model to evaluate
            test_loader: Test data loader
            config: Evaluation configuration
        """
        self.model = model
        self.test_loader = test_loader
        self.config = config or {}

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

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model on test data.

        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        self.predictions = []
        self.targets = []
        self.losses = []

        criterion = self._get_loss_function()

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
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
            return self._calculate_classification_metrics()
        elif self.task_type == "regression":
            return self._calculate_regression_metrics()
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

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
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        metrics["r2_score"] = r2

        return metrics

    def plot_confusion_matrix(self, save_path: Optional[str] = None):
        """Plot confusion matrix for classification tasks."""
        if self.task_type != "classification":
            raise ValueError(
                "Confusion matrix is only available for classification tasks")

        from sklearn.metrics import confusion_matrix

        # Convert predictions to class labels
        if self.predictions.dim() == 2:  # Multi-class
            pred_labels = torch.argmax(self.predictions, dim=1)
        else:  # Binary
            pred_labels = (torch.sigmoid(self.predictions) > 0.5).long()

        pred_labels = pred_labels.numpy()
        true_labels = self.targets.numpy()

        cm = confusion_matrix(true_labels, pred_labels)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_predictions_vs_targets(self, save_path: Optional[str] = None):
        """Plot predictions vs targets for regression tasks."""
        if self.task_type != "regression":
            raise ValueError(
                "Predictions vs targets plot is only available for regression tasks")

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
            plt.savefig(save_path)
        else:
            plt.show()

    def get_predictions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get raw predictions and targets.

        Returns:
            Tuple of (predictions, targets)
        """
        return self.predictions, self.targets

    def save_results(self, save_path: str):
        """Save evaluation results to file."""
        results = {
            "predictions": self.predictions.numpy(),
            "targets": self.targets.numpy(),
            "losses": self.losses,
            "config": self.config
        }

        np.savez(save_path, **results)

    def load_results(self, load_path: str):
        """Load evaluation results from file."""
        data = np.load(load_path, allow_pickle=True)

        self.predictions = torch.from_numpy(data["predictions"])
        self.targets = torch.from_numpy(data["targets"])
        self.losses = data["losses"].tolist()
        self.config = data["config"].item()


class MetricTracker:
    """
    Helper class to track metrics during training/evaluation.
    """

    def __init__(self):
        self.metrics = {}
        self.history = {}

    def update(self, metrics: Dict[str, float]):
        """Update metrics with new values."""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
            self.metrics[key] = value

    def get_metric(self, name: str) -> Optional[float]:
        """Get current value of a metric."""
        return self.metrics.get(name)

    def get_history(self, name: str) -> Optional[List[float]]:
        """Get history of a metric."""
        return self.history.get(name)

    def get_best(self, name: str, mode: str = "min") -> Tuple[float, int]:
        """
        Get best value and epoch for a metric.

        Args:
            name: Metric name
            mode: "min" or "max"

        Returns:
            Tuple of (best_value, best_epoch)
        """
        if name not in self.history:
            raise ValueError(f"Metric {name} not found")

        history = self.history[name]
        if mode == "min":
            best_idx = np.argmin(history)
        else:
            best_idx = np.argmax(history)

        return history[best_idx], best_idx

    def plot_metrics(self, metrics: List[str], save_path: Optional[str] = None):
        """Plot metric history."""
        fig, axes = plt.subplots(
            len(metrics), 1, figsize=(10, 4 * len(metrics)))

        if len(metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            if metric in self.history:
                axes[i].plot(self.history[metric])
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric)
                axes[i].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def reset(self):
        """Reset all metrics and history."""
        self.metrics = {}
        self.history = {}

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        summary = {}
        for name, history in self.history.items():
            if history:
                summary[name] = {
                    "current": history[-1],
                    "best": min(history),
                    "worst": max(history),
                    "mean": np.mean(history),
                    "std": np.std(history)
                }
        return summary
