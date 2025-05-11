"""THIS IS IT BABYYY

Train a basic logistic regression model on the data."""

import polars as pl
from sklearn.linear_model import LogisticRegression

train_df = pl.read_parquet("data/processed/resampled_train_22.parquet")
# test_df = pd.read_parquet("data/processed/test_23.parquet")

index_columns = [
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


clean_train_df = (
    train_df.with_columns(
        pl.col("gametime_elapsed").sub(pl.datetime(1970, 1, 1)).dt.total_seconds()
    )
    .select(pl.col(pl.NUMERIC_DTYPES))
    .drop(index_columns)
).drop_nulls()

X = clean_train_df.drop("home_win")
y = clean_train_df["home_win"].to_numpy()

# resampled_train_df.sort(["GAME_ID", "gametime_elapsed"])

model = LogisticRegression().fit(X=X, y=y)
model

# Check todo.md for more info
