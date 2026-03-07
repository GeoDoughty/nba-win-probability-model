import polars as pl

data_df = pl.read_parquet("data/raw/pbp/parquet/datanba_2023.tar.parquet")
data_df = data_df.filter(pl.col("etype").eq(6))
data_df = data_df.with_columns(
    pl.col("de").str.extract(r"Foul:\s([^\(]+)", 1).alias("foul_type")
)

# Looks like all null ones are techs which don't count towards the bonus
data_df.filter(pl.col("foul_type").is_null())

# Next time:
# 1. Create a rule for if it's a team foul or not (defensive + loose ball)
# 2. Cumsum over quarters per team
# 3. Pivot to get total teams fouls
# 4. Add flag for when the team went in the bonus (this is different for OT)
