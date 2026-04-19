"""XGBoost implementation with proper model tracking and storage.

Uses MLflow for experiment tracking and model registry.
Saves best model from GridSearchCV for reproducibility.
"""

import mlflow
from pathlib import Path
import polars as pl
import numpy as np
import json
from rich import print
from xgboost import XGBClassifier
import pickle

from src.eval.plots.roc_curve import plot_roc_curve
from src.eval.plots.pr_curve import plot_precision_recall_curve
from src.eval.plots.confusion_matrix import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV

from src.eval.metrics import calculate_accuracy_metrics
from sklearn.model_selection import StratifiedKFold

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

# ===== CLASS BALANCE ANALYSIS =====
n_away_train = (train_y == 0).sum()
n_home_train = (train_y == 1).sum()
n_away_test = (test_y == 0).sum()
n_home_test = (test_y == 1).sum()

train_home_pct = (n_home_train / len(train_y)) * 100
test_home_pct = (n_home_test / len(test_y)) * 100

# Calculate optimal scale_pos_weight: weight rare class more
# scale_pos_weight = n_negative_class / n_positive_class
optimal_scale_pos_weight = n_away_train / n_home_train
baseline_accuracy = max(n_home_train / len(train_y), n_away_train / len(train_y))

print(f"\n{'=' * 50}")
print("CLASS BALANCE ANALYSIS")
print(f"{'=' * 50}")
print(
    f"Train set: {n_home_train} home wins ({train_home_pct:.1f}%), {n_away_train} away wins ({100 - train_home_pct:.1f}%)"
)
print(
    f"Test set:  {n_home_test} home wins ({test_home_pct:.1f}%), {n_away_test} away wins ({100 - test_home_pct:.1f}%)"
)
print(f"Class ratio (home/away): {n_home_train / n_away_train:.3f}")
print(f"→ Optimal scale_pos_weight: {optimal_scale_pos_weight:.3f}")
print(f"→ Baseline accuracy (always predict majority): {baseline_accuracy:.3f}")
print(f"{'=' * 50}\n")
# ===== END CLASS BALANCE =====

# Use this next time: https://xgboost.readthedocs.io/en/stable/parameter.html
# https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html
# https://mlflow.org/docs/latest/ml/traditional-ml/xgboost/#grid-search
param_grid = {
    "max_depth": [2, 3, 5],
    "subsample": [0.6, 0.9],
    "colsample_bytree": [0.6, 0.9],
    "learning_rate": [0.1, 0.3],
    "scale_pos_weight": [optimal_scale_pos_weight, optimal_scale_pos_weight * 0.9],
}

with mlflow.start_run() as run:
    # Define model features
    # create model instance
    xgb = XGBClassifier(
        n_estimators=100, objective="binary:logistic", random_state=42, n_jobs=-1
    )
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        verbose=2,
        n_jobs=-1,
    )
    grid_search.fit(train_X, train_y)

    # Print best parameters
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict_proba(train_X)[:, 1]

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
    y_pred_test = best_model.predict_proba(test_X)[:, 1]
    export_test_df = test_df.drop_nulls().with_columns(home_win_prob=y_pred_test)
    test_metrics = calculate_accuracy_metrics(export_test_df, "home_win_prob")

    print("Metrics on test data:")
    for metric, val in test_metrics.items():
        print(f"{metric}: {val}")

    mlflow.log_metrics(
        {"train_" + k: v for k, v in train_metrics.items()}
        | {"test_" + k: v for k, v in test_metrics.items()}
    )

    roc_curve_fig = plot_roc_curve(test_y, y_pred_test)
    pr_curve_fig = plot_precision_recall_curve(test_y, y_pred_test)
    conf_mat_fig = plot_confusion_matrix(test_y, y_pred_test)

    mlflow.log_figure(roc_curve_fig, "test/roc_curve.png")
    mlflow.log_figure(pr_curve_fig, "test/precision_recall_curve.png")
    mlflow.log_figure(conf_mat_fig, "test/confusion_matrix.png")

    # Log model hyperparameters
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("train_size", len(train_X))
    mlflow.log_param("test_size", len(test_X))

    # Log class balance metrics
    mlflow.log_metric("train_home_win_pct", train_home_pct)
    mlflow.log_metric("test_home_win_pct", test_home_pct)
    mlflow.log_metric("baseline_accuracy", baseline_accuracy)
    mlflow.log_metric("n_home_train", n_home_train)
    mlflow.log_metric("n_away_train", n_away_train)

    # Log feature names for reproducibility
    feature_names = train_X.columns
    mlflow.log_param("n_features", len(feature_names))
    mlflow.log_dict({"feature_names": feature_names}, "feature_names.json")

    # Save and log the best model
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    # Log model artifact
    mlflow.log_artifact(str(model_path), artifact_path="models")

    # Register model in MLflow
    mlflow.xgboost.log_model(
        best_model,
        artifact_path="xgb_model",
        registered_model_name="nba-win-probability-xgb",
    )

    # Save model metadata
    model_metadata = {
        "best_params": grid_search.best_params_,
        "best_cv_score": float(grid_search.best_score_),
        "n_estimators": best_model.n_estimators,
        "feature_names": feature_names,
        "train_metrics": {k: float(v) for k, v in train_metrics.items()},
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "train_size": len(train_X),
        "test_size": len(test_X),
        "class_balance": {
            "train_home_win_pct": train_home_pct,
            "test_home_win_pct": test_home_pct,
            "n_home_train": int(n_home_train),
            "n_away_train": int(n_away_train),
            "optimal_scale_pos_weight": float(optimal_scale_pos_weight),
            "baseline_accuracy": float(baseline_accuracy),
        },
    }

    with open(output_path / "model_metadata.json", "w") as f:
        json.dump(model_metadata, f, indent=4)

    # Export predictions
    export_train_df.write_parquet(output_path / train_path.name)
    export_test_df.write_parquet(output_path / test_path.name)

    print(f"\n✓ Model saved to: {model_path}")
    print(f"✓ Artifacts saved to: {output_path}")
    print(f"✓ MLflow run ID: {run.info.run_id}")
    print("\n📊 CLASS BALANCE SUMMARY:")
    print(f"   Baseline accuracy: {baseline_accuracy:.3f}")
    print(f"   Test accuracy: {test_metrics.get('accuracy', 'N/A')}")
    print(
        f"   → Improvement over baseline: {(test_metrics.get('accuracy', 0) - baseline_accuracy):.3f}"
    )
