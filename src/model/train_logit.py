"""THIS IS IT BABYYY

Train a basic logistic regression model on the data."""

import polars as pl
from sklearn.linear_model import LogisticRegression

train_df = pl.read_parquet("data/processed/train_22.parquet")
# test_df = pd.read_parquet("data/processed/test_23.parquet")

train_df

index_columns = []

train_df["home_win"]
model = LogisticRegression().fit()

# Check todo.md for more info
