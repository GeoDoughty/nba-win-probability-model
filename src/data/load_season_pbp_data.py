"""
Loads Raw PBP data for a single season
"""
from nba_api.stats.endpoints import PlayByPlay, LeagueGameLog


def get_games(season: str | None = None):
    log = LeagueGameLog(
        season=season
    )
    games_df = log.get_data_frames()[0]
    away_mask = games_df['MATCHUP'].str.contains('@')
    return games_df[~away_mask]


if __name__ == '__main__':
    games_df = get_games('2022')