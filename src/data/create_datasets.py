"""Build train and test data set

This needs at a minimum:
1. The winner of each game
2. The score differential at X second intervals
3. Some other basic features

For info coming into the game:
- leaguedashteamstats (stats about the last N games for a team)
- Need a way to find out which game we on and get the season at that point"""

import pandas as pd
from nba_api.stats.endpoints.teamgamelog import TeamGameLog


raw_pbp_df = pd.read_parquet(r"data\raw\pbp\parquet\datanba_2022.tar.parquet")

# ! This will break in the 90s
season = f"20{raw_pbp_df['GAME_ID'].iloc[0]}"[:4]


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


# Add in score differential and winner
raw_pbp_df["score_diff"] = raw_pbp_df["hs"] - raw_pbp_df["vs"]
raw_pbp_df

team_list = raw_pbp_df[["tid", "oftid"]].melt()["value"].unique()
game_stat_df = get_game_time_rolling_team_stats(team_list, season)

raw_pbp_df["GAME_ID"] = "00" + raw_pbp_df["GAME_ID"].astype(str)
full_pbp_df = raw_pbp_df.merge(game_stat_df, left_on=["GAME_ID"], right_on=["Game_ID"])


# Comments for next week:
# - ~~Add in score differential~~
# - ~~Merge team stats with pbp data (may need to set a home and away merge as this isn't provided in the PBP)~~
# - Split into train and test
# - Export
