"""Functions for generating and plotting ROC curves."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc as sklearn_auc


def plot_precision_recall_curve(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    """
    Calculate and plot Precision-Recall curve.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted probabilities for the positive class

    Returns:
        Matplotlib figure object for MLflow logging
    """

    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = sklearn_auc(recall, precision)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot precision-recall curve
    ax.plot(
        recall,
        precision,
        color="darkorange",
        lw=2,
        label=f"PR curve (AUC = {pr_auc:.3f})",
    )

    # Add baseline (no skill classifier)
    baseline = np.sum(y_true) / len(y_true)
    ax.axhline(
        y=baseline,
        color="navy",
        lw=2,
        linestyle="--",
        label=f"Baseline ({baseline:.3f})",
    )

    # Formatting
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    return fig
