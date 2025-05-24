"""THIS IS IT BABYYY

Train a basic logistic regression model on the data."""

import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np


def prepare_data(df: pl.DataFrame) -> pl.DataFrame:
    """Prepare the data by converting columns to appropriate types."""
    return (
        df.with_columns(
            pl.col("gametime_elapsed").sub(pl.datetime(1970, 1, 1)).dt.total_seconds()
        )
        .select(pl.col(pl.NUMERIC_DTYPES))
        .drop(INDEX_COLUMNS)
    ).drop_nulls()


train_df = pl.read_parquet("data/processed/resampled_train_22.parquet")
test_df = pl.read_parquet("data/processed/resampled_test_23.parquet")

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
    # "PERIOD",
]

clean_train_df = prepare_data(train_df)
clean_test_df = prepare_data(test_df)

test_cols = [
    "home_win",
    "gametime_elapsed",
    "score_diff",
    "HOME_ENTRY_W_PCT",
    "AWAY_ENTRY_W_PCT",
]

reduced_train_df = clean_train_df.select(test_cols)
X = reduced_train_df.drop("home_win")
y = reduced_train_df["home_win"].to_numpy()

model = LogisticRegression().fit(X=X, y=y)

model.classes_
X_pred = model.predict_proba(X)[:, 1]
np.unique(X_pred, return_counts=True)

clean_train_df = clean_train_df.with_columns(home_win_prob=X_pred)
avg_accuracy = (
    (clean_train_df["home_win_prob"] > 0.5) == (clean_train_df["home_win"] == 1)
).mean()

print(f"Average accuracy on training set: {avg_accuracy:.2%}")

# Run on test data
reduced_test_df = clean_test_df.select(test_cols)
X_test = reduced_test_df.drop("home_win")
y = reduced_test_df["home_win"].to_numpy()
clean_test_df = clean_test_df.with_columns(
    home_win_prob=model.predict_proba(X_test)[:, 1]
)

test_avg_accuracy = (
    (clean_test_df["home_win_prob"] > 0.5) == (clean_test_df["home_win"] == 1)
).mean()

print(f"Average accuracy on test set: {test_avg_accuracy:.2%}")
print("you did it chief")
# Check todo.md for more info
