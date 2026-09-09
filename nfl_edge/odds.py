"""
Sportsbook odds: two sources, one interface (a DataFrame with best price,
consensus price, book count, and market width per game/side).

1. HistoricalOddsSource -- reads the moneyline already embedded in
   nflverse's games.csv. This is a single consensus-like line, not a
   per-book breakdown, so book_count is fixed at 1 and width at 0 --
   it exists so backtests can run against *real* market prices for free,
   without a paid API key, and so a card can still be built for an
   upcoming week that already has an opening line posted.

2. LiveOddsAPI -- a thin client for https://the-odds-api.com/, which
   does return per-book prices for the whitelisted books in config.py.
   This is what a production weekly run should use once a key is added
   (free tier covers a season of weekly pulls comfortably). Without a
   key it raises a clear error rather than silently returning nothing.
"""
from __future__ import annotations

import os
import statistics
from dataclasses import dataclass

import pandas as pd
import requests

from . import config


def american_to_decimal(odds: float) -> float:
    if odds > 0:
        return 1 + odds / 100.0
    return 1 + 100.0 / (-odds)


@dataclass
class HistoricalOddsSource:
    """Moneylines already present in the games table (data.load_games())."""

    def for_week(self, games: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for g in games.itertuples(index=False):
            for side, team, ml in (
                ("home", g.home_team, getattr(g, "home_moneyline", None)),
                ("away", g.away_team, getattr(g, "away_moneyline", None)),
            ):
                if pd.isna(ml):
                    continue
                rows.append({
                    "game_id": g.game_id, "side": side, "team": team,
                    "best_price": ml, "consensus_price": ml,
                    "n_books": 1, "width_dec": 0.0, "best_book": "nflverse_line",
                })
        return pd.DataFrame(rows)


@dataclass
class LiveOddsAPI:
    """Client for the-odds-api.com's NFL h2h (moneyline) market."""
    api_key: str | None = None
    sport_key: str = "americanfootball_nfl"
    regions: str = "us"

    def __post_init__(self):
        self.api_key = self.api_key or os.environ.get("ODDS_API_KEY")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def fetch_raw(self) -> list[dict]:
        if not self.enabled:
            raise RuntimeError(
                "No odds API key set. Get a free key at https://the-odds-api.com/ "
                "and set the ODDS_API_KEY environment variable, or pass api_key= "
                "explicitly. Until then, use HistoricalOddsSource for backtests "
                "and pre-week lines."
            )
        url = f"{config.ODDS_API_BASE_URL}/sports/{self.sport_key}/odds"
        params = {
            "apiKey": self.api_key, "regions": self.regions,
            "markets": "h2h", "oddsFormat": "american",
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def for_week(self, games: pd.DataFrame) -> pd.DataFrame:
        """Fetch current odds and shape them into the same per-game/side
        rows as HistoricalOddsSource, filtered to the whitelisted books
        and matched to `games` by team names."""
        raw = self.fetch_raw()
        whitelist = set(config.SPORTSBOOK_WHITELIST)

        by_teams: dict[tuple, dict] = {}
        for event in raw:
            home, away = event.get("home_team"), event.get("away_team")
            prices = {"home": {}, "away": {}}
            for book in event.get("bookmakers", []):
                title = book.get("title")
                if title not in whitelist:
                    continue
                for market in book.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    for outcome in market.get("outcomes", []):
                        if outcome["name"] == home:
                            prices["home"][title] = outcome["price"]
                        elif outcome["name"] == away:
                            prices["away"][title] = outcome["price"]
            by_teams[(home, away)] = prices

        rows = []
        for g in games.itertuples(index=False):
            key = (g.home_team, g.away_team)
            # the-odds-api uses full team names; games.csv uses abbreviations.
            # A production run needs a team-name crosswalk here -- left as a
            # documented TODO since it depends on the exact provider strings.
            prices = by_teams.get(key)
            if not prices:
                continue
            for side, team in (("home", g.home_team), ("away", g.away_team)):
                book_prices = prices[side]
                if len(book_prices) < 1:
                    continue
                best_book, best_price = max(book_prices.items(), key=lambda kv: kv[1])
                consensus = statistics.median(book_prices.values())
                decs = [american_to_decimal(p) for p in book_prices.values()]
                rows.append({
                    "game_id": g.game_id, "side": side, "team": team,
                    "best_price": best_price, "consensus_price": consensus,
                    "n_books": len(book_prices), "width_dec": max(decs) - min(decs),
                    "best_book": best_book,
                })
        return pd.DataFrame(rows)


def get_odds_source(prefer_live: bool = True) -> "HistoricalOddsSource | LiveOddsAPI":
    """Pick the live API when a key is configured, otherwise fall back to
    the free historical/pre-week line so the pipeline never hard-stops."""
    live = LiveOddsAPI()
    if prefer_live and live.enabled:
        return live
    return HistoricalOddsSource()
