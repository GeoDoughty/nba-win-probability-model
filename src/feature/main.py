import polars as pl
import numpy as np

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


def split_train_test_cols(
    df: pl.DataFrame, y_col: str
) -> tuple[pl.DataFrame, np.ndarray]:
    """Split the dataframe into features and target variable."""
    X = df.select(pl.exclude(y_col))
    y = df[y_col].to_numpy()

    return X, y


if __name__ == "__main__":
    train_df = prepare_data(
        pl.read_parquet("data/processed/resampled_train_22.parquet")
    )
