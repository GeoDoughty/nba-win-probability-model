"""Functions for generating and plotting confusion matrices."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
) -> plt.Figure:
    """
    Calculate and plot confusion matrix.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted probabilities for the positive class
        threshold: Classification threshold (default: 0.5)

    Returns:
        Matplotlib figure object for MLflow logging
    """
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot confusion matrix as heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar=True,
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )

    # Formatting
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix (threshold={threshold})")

    plt.tight_layout()

    return fig
