import polars as pl

FOUL_LIST = [
    "Away From Play ",
    "Personal ",
    "Transition Take ",
    "Non-Unsportsmanlike Technical Foul - Defensive Three Seconds ",
    "Clear Path ",
    "Loose Ball ",
    "Shooting ",
    "Personal Take ",
]


data_df = pl.read_parquet("data/raw/pbp/parquet/datanba_2023.tar.parquet")
data_df = data_df.filter(pl.col("etype").eq(6))
data_df = data_df.with_columns(
    pl.col("de").str.extract(r"Foul:\s([^\(]+)", 1).alias("foul_type")
)

# Looks like all null ones are techs which don't count towards the bonus
data_df.filter(pl.col("foul_type").is_null())

# 1. Create a rule for if it's a team foul or not (defensive + loose ball)
data_df = data_df.with_columns(
    data_df["foul_type"].is_in(FOUL_LIST).alias("is_team_foul")
)

# 2. Cumsum over quarters per team
data_df = data_df.with_columns(
    ("00:" + pl.col("cl"))
    .str.to_time()
    .sub(pl.time(0, 0, 0))
    .alias("quarter_duration"),
).with_columns(
    (
        (
            pl.col("quarter_duration")
            + ((pl.col("PERIOD") - 1) * pl.duration(minutes=12))
        ).alias("game_duration")
    )
)

data_df = data_df.sort(
    "GAME_ID", "PERIOD", "quarter_duration", descending=[False, False, True]
).with_columns(
    pl.cum_sum("is_team_foul").over("GAME_ID", "PERIOD", "tid").alias("team_fouls")
)
data_df = data_df.with_columns(
    pl.when(pl.col("PERIOD") < 5)
    .then(pl.col("team_fouls") >= 5)
    .otherwise(pl.col("team_fouls") >= 4)
    .alias("in_bonus")
)
# 4. Add flag for when the team went in the bonus (this is different for OT)
