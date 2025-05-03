"""Build train and test data set

This needs at a minimum:
1. The winner of each game
2. The score differential at X second intervals
3. Some other basic features

For info coming into the game:
- leaguedashteamstats (stats about the last N games for a team)
- Need a way to find out which game we on and get the season at that point"""

import re
import pandas as pd
from nba_api.stats.endpoints.teamgamelog import TeamGameLog

### Data loading params ###
TRAIN_DATA_PATHS = [
    # "data/raw/pbp/parquet/datanba_2021.tar.parquet",  # ! Doesn't contain wallclk just clk
    "data/raw/pbp/parquet/datanba_2022.tar.parquet",
]
TEST_DATA_PATHS = [
    "data/raw/pbp/parquet/datanba_2023.tar.parquet",
]
### ------------------- ###

SEASON_REGEX = r"data/raw/pbp/parquet/datanba_(\d{4})\.tar\.parquet"


def load_datanba_parquet(path: str) -> tuple[pd.DataFrame, str]:
    """Load datanab parquet file from datanba."""
    raw_pbp_df = pd.read_parquet(path)
    season = re.match(SEASON_REGEX, path).group(1)

    # Add in score differential and winner
    raw_pbp_df["score_diff"] = raw_pbp_df["hs"] - raw_pbp_df["vs"]
    home_win_series = (
        raw_pbp_df.sort_values("wallclk").groupby("GAME_ID")["score_diff"].last() > 0
    )
    home_win_series = home_win_series.astype(int).rename("home_win")
    raw_pbp_df = raw_pbp_df.merge(home_win_series, left_on="GAME_ID", right_index=True)

    # Clean to match the NBA api
    raw_pbp_df["GAME_ID"] = "00" + raw_pbp_df["GAME_ID"].astype(str)
    return raw_pbp_df, season


def get_team_cum_stats(team_id: str, season: str) -> pd.DataFrame:
    df = TeamGameLog(team_id=team_id, season=season).get_data_frames()[0]
    team_game_log = (
        df.sort_values("Game_ID", ascending=True)
        .drop(columns=["MIN"])
        .reset_index(drop=True)
    )
    team_game_log["GAME_DATE"] = pd.to_datetime(team_game_log["GAME_DATE"])

    info_cols = ["Team_ID", "Game_ID", "GAME_DATE", "MATCHUP"]
    shift_cols = ["WL", "W", "L", "W_PCT"]
    agg_cols = team_game_log.columns.difference(info_cols + shift_cols)

    return pd.concat(
        [
            team_game_log[info_cols],
            team_game_log[shift_cols].shift(1).add_prefix("ENTRY_"),
            team_game_log[agg_cols]
            .shift(1)
            .expanding()
            .mean()
            .add_suffix("_SEASON_AVG"),
            team_game_log[agg_cols].rolling(5).mean().add_suffix("_LAST_5_AVG"),
        ],
        axis=1,
    )


def get_game_time_rolling_team_stats(team_id_list: list, season: str) -> pd.DataFrame:
    """Get teams rolling stats entering the game, split by home and away."""
    team_stat_list = []
    for team_id in team_id_list:
        if team_id == 0:
            continue
        team_stat_list.append(get_team_cum_stats(team_id, season))

    team_stat_df = pd.concat(team_stat_list)

    # Flatten team stats to be by game, home and away
    team_stat_df["is_home_team"] = team_stat_df["MATCHUP"].str.contains("vs.")
    index_cols = ["Game_ID", "GAME_DATE"]

    home_df = (
        team_stat_df[team_stat_df["is_home_team"]]
        .rename(
            columns={
                v: f"HOME_{v}" if v not in index_cols else v
                for v in team_stat_df.columns
            }
        )
        .drop(columns=["HOME_is_home_team"])
    )
    away_df = (
        team_stat_df[~team_stat_df["is_home_team"]]
        .rename(
            columns={
                v: f"AWAY_{v}" if v not in index_cols else v
                for v in team_stat_df.columns
            }
        )
        .drop(columns=["AWAY_is_home_team"])
    )

    return home_df.merge(away_df, on=index_cols)


def load_and_process_single_season(path: str) -> pd.DataFrame:
    """Load and process raw data."""
    raw_pbp_df, season = load_datanba_parquet(path)
    team_list = raw_pbp_df[["tid", "oftid"]].melt()["value"].unique()
    game_stat_df = get_game_time_rolling_team_stats(team_list, season)

    full_pbp_df = raw_pbp_df.merge(
        game_stat_df, left_on=["GAME_ID"], right_on=["Game_ID"]
    )
    return full_pbp_df


def create_train_test_datasets() -> None:
    """Create train and test datasets."""
    train_df = pd.concat(
        [load_and_process_single_season(path) for path in TRAIN_DATA_PATHS]
    )
    test_df = pd.concat(
        [load_and_process_single_season(path) for path in TEST_DATA_PATHS]
    )

    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = create_train_test_datasets()
    train_df.to_parquet("data/processed/train_22.parquet")
    test_df.to_parquet("data/processed/test_23.parquet")

# Comments for next week:
# - ~~Add in score differential~~
# - ~~Merge team stats with pbp data (may need to set a home and away merge as this isn't provided in the PBP)~~
# - Split into train and test
# - Export
