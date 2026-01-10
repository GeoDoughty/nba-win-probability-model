"""Basic RFECV implementation.

Next time"""

from pathlib import Path
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from rich import print

from src.eval.metrics import calculate_accuracy_metrics

### ---- User Inputs ---- ###
output_path = Path("data/processed/rfecv/")

train_path = Path("data/processed/resampled_train_22.parquet")
test_path = Path("data/processed/resampled_test_23.parquet")

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
train_X = train_X.select(
    [
        "gametime_elapsed",
        "score_diff",
        "HOME_ENTRY_W_PCT",
        "AWAY_ENTRY_W_PCT",
        "AWAY_FG3A_SEASON_AVG",
        "HOME_FG3A_SEASON_AVG",
        "AWAY_PF_LAST_5_AVG",
        "HOME_PF_LAST_5_AVG",
    ]
)
test_X, test_y = split_train_test_cols(clean_test_df)
test_X = test_X.select(
    [
        "gametime_elapsed",
        "score_diff",
        "HOME_ENTRY_W_PCT",
        "AWAY_ENTRY_W_PCT",
        "AWAY_FG3A_SEASON_AVG",
        "HOME_FG3A_SEASON_AVG",
        "AWAY_PF_LAST_5_AVG",
        "HOME_PF_LAST_5_AVG",
    ]
)

# Define model features
min_features_to_select = 1  # Minimum number of features to consider
clf = LogisticRegression()
cv = StratifiedKFold(5)

pipe = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        (
            "rfecv",
            RFECV(
                estimator=clf,
                step=1,
                cv=cv,
                scoring="accuracy",
                min_features_to_select=1,
                n_jobs=2,
                verbose=1,
            ),
        ),
    ]
)

pipe.fit(train_X.to_numpy(), train_y)
print(f"Optimal number of features: {pipe.named_steps['rfecv'].n_features_}")
print(
    f"Selected Features: {pipe.named_steps['rfecv'].get_feature_names_out(input_features=train_X.columns)}"
)
print(f"Train Accuracy: {pipe.score(train_X.to_numpy(), train_y)}")
print(f"Test Accuracy: {pipe.score(test_X.to_numpy(), test_y)}")

X_pred = pipe.predict_proba(train_X)[:, 1]
print(np.unique(X_pred, return_counts=True))

## Next time
# - figure out how to export the pipe
# - run on whole dataset


export_train_df = train_df.drop_nulls().with_columns(home_win_prob=X_pred)
train_metrics = calculate_accuracy_metrics(export_train_df, "home_win_prob")

print("Metrics on train data:")
for metric, val in train_metrics.items():
    print(f"{metric}: {val}")

# Run on test data
export_test_df = test_df.drop_nulls().with_columns(
    home_win_prob=pipe.predict_proba(test_X)[:, 1]
)
test_metrics = calculate_accuracy_metrics(export_test_df, "home_win_prob")

print("Metrics on test data:")
for metric, val in test_metrics.items():
    print(f"{metric}: {val}")


output_path.mkdir(parents=True, exist_ok=True)

export_train_df.write_parquet(output_path / train_path.name)
export_test_df.write_parquet(output_path / test_path.name)
