"""Functions for generating and plotting ROC curves."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_roc_curve(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    """
    Calculate and plot ROC curve.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted probabilities for the positive class

    Returns:
        Matplotlib figure object for MLflow logging
    """
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot ROC curve
    ax.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
    )
    ax.plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier"
    )

    # Formatting
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    return fig
