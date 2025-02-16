"""Loads Raw PBP data for a single season."""

import time
from pathlib import Path

import pandas as pd
import requests
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


def load_season_pbp_data(season: str, raw_output_path: Path, proc_output_path: Path) -> None:
    raw_output_path = raw_output_path / season
    if not raw_output_path.exists():
        raw_output_path.mkdir(parents=True)

    if not proc_output_path.exists():
        proc_output_path.mkdir(parents=True)

    games_df = get_games(season)
    parsed_files = list(raw_output_path.glob("*.parquet"))

    # Filter files that have already been parsed
    games_df = games_df[~games_df["GAME_ID"].isin([file.stem for file in parsed_files])]

    for game_id in tqdm(games_df["GAME_ID"], "Loading PBP Data"):
        single_game_pbp_df = get_single_game_pbp(game_id)
        single_game_pbp_df.to_parquet(raw_output_path / f"{game_id}.parquet")

    pbp_list = [pd.read_parquet(file) for file in raw_output_path.glob("*.parquet")]
    full_pbp_df = pd.concat(pbp_list)
    full_pbp_df.to_parquet(proc_output_path / f"{season}_pbp.parquet")


if __name__ == "__main__":
    load_season_pbp_data(
        season="2022",
        raw_output_path=Path("data/raw/pbp"),
        proc_output_path=Path("data/processed/pbp"),
    )
