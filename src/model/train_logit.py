"""THIS IS IT BABYYY

Train a basic logistic regression model on the data."""

import polars as pl
from sklearn.linear_model import LogisticRegression

train_df = pl.read_parquet("data/processed/train_22.parquet")
# test_df = pd.read_parquet("data/processed/test_23.parquet")
train_df["GAME_ID"]

resampled_train_df = (
    train_df.with_columns(pl.col("gametime_elapsed").add(pl.datetime(1970, 1, 1)))
    .sort("gametime_elapsed")
    .upsample(
        time_column="gametime_elapsed",
        every="5s",
        group_by=["GAME_ID"],
        maintain_order=True,
    )
    # .sort(["GAME_ID", "gametime_elapsed"])
    .fill_null(strategy="forward")
)
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
    "PERIOD",
]

resampled_train_df.row(40_000, named=True)


clean_train_df = (
    resampled_train_df.with_columns(
        pl.col("gametime_elapsed").sub(pl.datetime(1970, 1, 1)).dt.total_seconds()
    )
    .select(pl.col(pl.NUMERIC_DTYPES))
    .drop(index_columns)
).drop_nulls()

X = clean_train_df.drop("home_win")
y = clean_train_df["home_win"].to_numpy()

# resampled_train_df.sort(["GAME_ID", "gametime_elapsed"])

train_df["home_win"]
model = LogisticRegression().fit(X=X, y=y)
model

# Check todo.md for more info
