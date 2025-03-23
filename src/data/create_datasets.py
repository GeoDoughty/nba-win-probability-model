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


team_list = raw_pbp_df[["tid", "oftid"]].melt()["value"].unique()
team_stat_list = []
for team_id in team_list:
    if team_id == 0:
        continue
    team_stat_list.append(get_team_cum_stats(team_id, season))

team_stat_df = pd.concat(team_stat_list)

# Comments for next week:
# - Merge team stats with pbp data (may need to set a home and away merge as this isn't provided in the PBP)
# - Add in score differential
# - Split into train and test
# - Export
