"""Basic XGBoost implementation.

Next time"""

import mlflow
from pathlib import Path
import polars as pl
import numpy as np
from rich import print
from xgboost import XGBClassifier

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

with mlflow.start_run() as run:
    # Define model features
    # create model instance
    bst = XGBClassifier(
        n_estimators=2, max_depth=2, learning_rate=1, objective="binary:logistic"
    )
    # fit model
    bst.fit(train_X, train_y)

    X_pred = bst.predict_proba(train_X)[:, 1]

    export_train_df = train_df.drop_nulls().with_columns(home_win_prob=X_pred)
    train_metrics = calculate_accuracy_metrics(export_train_df, "home_win_prob")

    print("Metrics on train data:")
    for metric, val in train_metrics.items():
        print(f"{metric}: {val}")

    # Run on test data
    export_test_df = test_df.drop_nulls().with_columns(
        home_win_prob=bst.predict_proba(test_X)[:, 1]
    )
    test_metrics = calculate_accuracy_metrics(export_test_df, "home_win_prob")

    print("Metrics on test data:")
    for metric, val in test_metrics.items():
        print(f"{metric}: {val}")

    mlflow.log_metrics(
        {"train_" + k: v for k, v in train_metrics.items()}
        | {"test_" + k: v for k, v in test_metrics.items()}
    )

output_path.mkdir(parents=True, exist_ok=True)

export_train_df.write_parquet(output_path / train_path.name)
export_test_df.write_parquet(output_path / test_path.name)
