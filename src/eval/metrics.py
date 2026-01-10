"""Common eval metrics for classifiers"""

from typing import Any
import polars as pl
from sklearn.metrics import (
    mean_squared_error,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def calculate_accuracy_metrics(df: pl.DataFrame, pred_col: str) -> dict[str, Any]:
    """Calculate accuracy metrics for the predictions."""
    df = df.with_columns(
        [
            (pl.col(pred_col) > 0.5).alias("predicted_home_win"),
            (pl.col("home_win") == 1).alias("actual_home_win"),
        ]
    )

    return {
        "mse": mean_squared_error(df["actual_home_win"], df[pred_col]),
        "accuracy": (df["predicted_home_win"] == df["actual_home_win"]).mean(),
        "brier_score": brier_score_loss(df["actual_home_win"], df[pred_col]),
        "log_loss": log_loss(df["actual_home_win"], df[pred_col]),
        "roc_auc": roc_auc_score(df["actual_home_win"], df[pred_col]),
    }
