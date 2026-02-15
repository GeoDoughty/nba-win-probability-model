"""Basic XGBoost implementation.

Next time"""

import mlflow
from pathlib import Path
import polars as pl
import numpy as np
from rich import print
from xgboost import XGBClassifier

from src.eval.plots.roc_curve import plot_roc_curve
from src.eval.plots.pr_curve import plot_precision_recall_curve
from src.eval.plots.confusion_matrix import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV

from src.eval.metrics import calculate_accuracy_metrics

mlflow.end_run()
mlflow.xgboost.autolog()
mlflow.set_experiment("XGBoost")


### ---- User Inputs ---- ###
output_path = Path("data/processed/xgboost/")

train_path = Path("data/processed/resampled_train_22.parquet")
test_path = Path("data/processed/resampled_test_23.parquet")

X_COLS = [
    "gametime_elapsed",
    "score_diff",
    "HOME_ENTRY_W_PCT",
    "AWAY_ENTRY_W_PCT",
    "AWAY_FG3A_SEASON_AVG",
    "HOME_FG3A_SEASON_AVG",
    "AWAY_PF_LAST_5_AVG",
    "HOME_PF_LAST_5_AVG",
]
Y_COL = "home_win"
### --------------------- ###

INDEX_COLUMNS = [
    "evt",
    "opt1",
    "opt2",
    "opt3",
    "opt4",
    "mtype",
    "etype",
    "opid",
    "tid",
    "pid",
    "hs",
    "vs",
    "epid",
    "oftid",
    "ord",
    "HOME_Team_ID",
    "AWAY_Team_ID",
    # "PERIOD",
]


def prepare_data(df: pl.DataFrame) -> pl.DataFrame:
    """Prepare the data by converting columns to appropriate types."""
    return (
        df.with_columns(
            pl.col("gametime_elapsed").sub(pl.datetime(1970, 1, 1)).dt.total_seconds()
        )
        .select(pl.col(pl.NUMERIC_DTYPES))
        .drop(INDEX_COLUMNS)
    ).drop_nulls()


def split_train_test_cols(df: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray]:
    """Split the dataframe into features and target variable."""
    X = df.select(pl.exclude(Y_COL))
    y = df[Y_COL].to_numpy()

    return X, y


train_df = pl.read_parquet(train_path)
test_df = pl.read_parquet(test_path)


clean_train_df = prepare_data(train_df)
clean_test_df = prepare_data(test_df)


train_X, train_y = split_train_test_cols(clean_train_df)
# train_X = train_X.select(X_COLS)
test_X, test_y = split_train_test_cols(clean_test_df)
# test_X = test_X.select(X_COLS)

# Use this next time: https://xgboost.readthedocs.io/en/stable/parameter.html
# https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html
# https://mlflow.org/docs/latest/ml/traditional-ml/xgboost/#grid-search
param_grid = {
    "max_depth": [2, 3, 5],
    "subsample": [0.6, 0.9],
    "colsample_bytree": [0.6, 0.9],
    "learning_rate": [0.1, 0.3],
    "scale_pos_weight": [1, 0.72],
}

with mlflow.start_run() as run:
    # Define model features
    # create model instance
    xgb = XGBClassifier(
        n_estimators=100, objective="binary:logistic", random_state=42, n_jobs=-1
    )
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, verbose=2)
    grid_search.fit(train_X, train_y)

    # Print best parameters
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")

    y_pred = grid_search.predict_proba(train_X)[:, 1]

    roc_curve_fig = plot_roc_curve(train_y, y_pred)
    pr_curve_fig = plot_precision_recall_curve(train_y, y_pred)
    conf_mat_fig = plot_confusion_matrix(train_y, y_pred)

    mlflow.log_figure(roc_curve_fig, "train/roc_curve.png")
    mlflow.log_figure(pr_curve_fig, "train/precision_recall_curve.png")
    mlflow.log_figure(conf_mat_fig, "train/confusion_matrix.png")

    export_train_df = train_df.drop_nulls().with_columns(home_win_prob=y_pred)
    train_metrics = calculate_accuracy_metrics(export_train_df, "home_win_prob")

    print("Metrics on train data:")
    for metric, val in train_metrics.items():
        print(f"{metric}: {val}")

    # Run on test data
    y_pred = grid_search.predict_proba(test_X)[:, 1]
    export_test_df = test_df.drop_nulls().with_columns(home_win_prob=y_pred)
    test_metrics = calculate_accuracy_metrics(export_test_df, "home_win_prob")

    print("Metrics on test data:")
    for metric, val in test_metrics.items():
        print(f"{metric}: {val}")

    mlflow.log_metrics(
        {"train_" + k: v for k, v in train_metrics.items()}
        | {"test_" + k: v for k, v in test_metrics.items()}
    )

    roc_curve_fig = plot_roc_curve(test_y, y_pred)
    pr_curve_fig = plot_precision_recall_curve(test_y, y_pred)
    conf_mat_fig = plot_confusion_matrix(test_y, y_pred)

    mlflow.log_figure(roc_curve_fig, "test/roc_curve.png")
    mlflow.log_figure(pr_curve_fig, "test/precision_recall_curve.png")
    mlflow.log_figure(conf_mat_fig, "test/confusion_matrix.png")

output_path.mkdir(parents=True, exist_ok=True)

export_train_df.write_parquet(output_path / train_path.name)
export_test_df.write_parquet(output_path / test_path.name)
