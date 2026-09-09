"""
Historical NFL data loading.

Source: nflverse's community-maintained games.csv (raw.githubusercontent.com,
free, no API key). It carries, for every game back to 1999: final score,
game_type (REG/WC/DIV/CON/SB), rest days for each team, divisional-game flag,
and — importantly — the historical closing moneyline for each team. That
last part matters: it means we can backtest the model against real market
prices instead of only simulated odds.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
import requests

from . import config

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "Mozilla/5.0 (nfl-edge research project)"}
_CACHE_FILE = config.DATA_RAW / "games_all.csv"


def refresh_games_cache(force: bool = False) -> Path:
    """Download the latest games.csv and cache it locally.

    Safe to call every week: it always fetches the freshest copy (which
    includes newly-posted lines for the upcoming week) and overwrites the
    cache. Historical rows for completed games don't change.
    """
    if _CACHE_FILE.exists() and not force:
        return _CACHE_FILE
    resp = requests.get(config.NFLVERSE_GAMES_URL, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.to_csv(_CACHE_FILE, index=False)
    logger.info("Cached %d games to %s", len(df), _CACHE_FILE)
    return _CACHE_FILE


def load_games(seasons: list[int] | None = None, refresh: bool = False) -> pd.DataFrame:
    """Load games as a DataFrame, optionally filtered to specific seasons.

    Columns of interest: season, week, game_type, gameday, home_team,
    away_team, home_score, away_score, home_moneyline, away_moneyline,
    home_rest, away_rest, div_game, result (home margin), roof, surface.
    """
    path = refresh_games_cache(force=refresh)
    df = pd.read_csv(path)
    df["gameday"] = pd.to_datetime(df["gameday"])
    if seasons is not None:
        df = df[df["season"].isin(seasons)].copy()
    return df.sort_values(["season", "week", "gameday"]).reset_index(drop=True)


def completed_games(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a final score (used for training/backtesting)."""
    return df[df["home_score"].notna() & df["away_score"].notna()].copy()


def upcoming_games(df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Rows for a specific not-yet-played (or in-progress) week."""
    return df[(df["season"] == season) & (df["week"] == week)].copy()


def next_unplayed_week(df: pd.DataFrame, season: int) -> int | None:
    """Convenience: find the next week in `season` with an unplayed game."""
    sub = df[df["season"] == season]
    unplayed = sub[sub["home_score"].isna()]
    if unplayed.empty:
        return None
    return int(unplayed["week"].min())
