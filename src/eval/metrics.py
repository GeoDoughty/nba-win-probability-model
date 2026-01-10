"""Common eval metrics for classifiers"""

from typing import Any
import polars as pl
from sklearn.metrics import mean_squared_error


def calculate_accuracy_metrics(df: pl.DataFrame, pred_col: str) -> dict[str, Any]:
    """Calculate accuracy metrics for the predictions."""
    df = df.with_columns(
        [
            (pl.col(pred_col) > 0.5).alias("predicted_home_win"),
            (pl.col("home_win") == 1).alias("actual_home_win"),
        ]
    )

    mse = mean_squared_error(df["actual_home_win"], df[pred_col])

    return {
        "mse": mse,
        "accuracy": (df["predicted_home_win"] == df["actual_home_win"]).mean(),
    }
