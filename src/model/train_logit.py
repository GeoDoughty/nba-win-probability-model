"""THIS IS IT BABYYY

Train a basic logistic regression model on the data."""

from pathlib import Path
import polars as pl
from sklearn.linear_model import LogisticRegression
import numpy as np


### ---- User Inputs ---- ###
output_path = Path("data/processed/logit/")

train_path = Path("data/processed/resampled_train_22.parquet")
test_path = Path("data/processed/resampled_test_23.parquet")

FEATURES = [
    "gametime_elapsed",
    "score_diff",
    "HOME_ENTRY_W_PCT",
    "AWAY_ENTRY_W_PCT",
]
Y_COL = "home_win"
### --------------------- ###


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
    X = df.select(FEATURES)
    y = df[Y_COL].to_numpy()

    return X, y


train_df = pl.read_parquet(train_path)
test_df = pl.read_parquet(test_path)

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


train_X, train_y = split_train_test_cols(clean_train_df)
test_X, test_y = split_train_test_cols(clean_test_df)

model = LogisticRegression().fit(X=train_X, y=train_y)

X_pred = model.predict_proba(train_X)[:, 1]
np.unique(X_pred, return_counts=True)

export_train_df = train_df.drop_nulls().with_columns(home_win_prob=X_pred)
avg_accuracy = (
    (export_train_df["home_win_prob"] > 0.5) == (export_train_df[Y_COL] == 1)
).mean()

print(f"Average accuracy on training set: {avg_accuracy:.2%}")

# Run on test data
export_test_df = test_df.drop_nulls().with_columns(
    home_win_prob=model.predict_proba(test_X)[:, 1]
)

test_avg_accuracy = (
    (clean_test_df["home_win_prob"] > 0.5) == (clean_test_df[Y_COL] == 1)
).mean()

print(f"Average accuracy on test set: {test_avg_accuracy:.2%}")
print("you did it chief")

output_path.mkdir(parents=True, exist_ok=True)

export_train_df.write_parquet(output_path / train_path.name)
export_test_df.write_parquet(output_path / test_path.name)
# Check todo.md for more info
