"""Aim is to predict the win probs of a current game, ideally one that is live?"""

from nba_api.stats.endpoints import playbyplayv3, leaguegamelog
from nba_api.stats.endpoints import teamgamelogs
import polars as pl

GAME_ID = "0022500054"  # Denver at Houston, 21/10/25

# need to map to these features
FEATURES = [
    "gametime_elapsed",
    "score_diff",
    "HOME_ENTRY_W_PCT",
    "AWAY_ENTRY_W_PCT",
]

logs = leaguegamelog.LeagueGameLog(season="2025-26").get_data_frames()[0]
logs = pl.from_pandas(logs)


df = playbyplayv3.PlayByPlayV3(GAME_ID).get_data_frames()[0]
df = pl.from_pandas(df)
df.head()  # just looking at the head of the data
df.to_dicts()[3]

# get team unique codes
home_stats = {}
away_stats = {}
for teams in logs.filter(pl.col("GAME_ID") == GAME_ID).to_dicts():
    if "vs." in teams["MATCHUP"]:
        home_stats = teams
    elif "@" in teams["MATCHUP"]:
        away_stats = teams
