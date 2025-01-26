"""Loads Raw PBP data for a single season.
"""
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog, PlayByPlay
from tqdm import tqdm


# Retry Wrapper
def retry(func, retries=3):
    """Shamelessy stolen from https://github.com/swar/nba_api/blob/master/docs/examples/Home%20Team%20Win-Loss%20Modeling/Home%20Team%20Win-Loss%20Data%20Prep.ipynb"""
    def retry_wrapper(*args, **kwargs):
        attempts = 0
        while attempts < retries:
            try:
                return func(*args, **kwargs)
            except requests.exceptions.RequestException as e:
                print(e)
                time.sleep(30)
                attempts += 1

    return retry_wrapper


def get_games(season: str | None = None):
    log = LeagueGameLog(
        season=season,
    )
    games_df = log.get_data_frames()[0]
    away_mask = games_df["MATCHUP"].str.contains("@")
    return games_df[~away_mask]

@retry
def get_single_game_pbp(game_id: str) -> pd.DataFrame:
    return PlayByPlay(game_id).get_data_frames()[0]

def load_season_pbp_data(games_df: pd.DataFrame, output_path: Path) -> None:
    if not output_path.parent.exists():
        output_path.mkdir(parents=True)

    pbp_list = []
    for game_id in tqdm(games_df.head(10)["GAME_ID"], "Loading PBP Data"):
        single_game_pbp_df = get_single_game_pbp(game_id)
        pbp_list.append(single_game_pbp_df)

    full_pbp_df = pd.concat(pbp_list)
    full_pbp_df.to_parquet(output_path)


if __name__ == "__main__":
    games_df = get_games("2022")
    load_season_pbp_data(games_df, output_path=Path("data/raw/pbp/2022_reg_season.parquet"))

    games_path = Path("data/raw/games/2022_reg_season.parquet")
    if not games_path.parent.exists():
        games_path.parent.mkdir(parents=True)
    games_df.to_parquet("data/raw/games/2022_reg_season.parquet")
