"""
Pass-rush pressure, approximated by sack rate -- walked forward with the
same no-lookahead discipline as elo.py/qb.py/epa.py.

Source: the same nflverse play-by-play release epa.py uses. Reduced, once
per season, to one row per team per game: sacks taken per dropback (how
often this team's OFFENSE let its QB get sacked) and sacks recorded per
opponent dropback (how often this team's DEFENSE got to the QB). Dropback
(pass attempt + sack + scramble, nflverse's `qb_dropback` flag) is the
standard denominator for sack rate -- using pass attempts alone would
undercount, since a sack ends the play before an "attempt" is logged.

Honest naming note: this is PRESSURE approximated by its most visible
outcome. True pressure (a QB hurried or hit but not sacked) isn't in any
free public data source -- charting that requires proprietary tracking
data (PFF, NFL Next Gen Stats) this project has no access to. Sack rate is
a real, meaningful signal on its own (it's what actually shows up on the
scoreboard and in EPA), but it understates a pass rush that generates
constant pressure without finishing the sack, and it can be inflated by a
QB who holds the ball too long rather than pure O-line/pass-rush quality.

Two factors come out of this, both signed so positive = an edge for the
home team, matching every other factor column in this project:

- sack_allowed_diff: away's sack rate taken (by their offense) minus
  home's -- positive means home's offensive line has protected its QB
  better than away's has.
- sack_generated_diff: home's sack rate recorded (by their defense) minus
  away's -- positive means home's pass rush has gotten there more often
  than away's has.

Same shrinkage-toward-a-prior + partial-season-carryover treatment as
epa.py, and the same two caveats apply here too: not opponent-adjusted (a
team that has faced three bad offensive lines will look like a better
pass rush than it may truly be), and no garbage-time filter.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from . import config

logger = logging.getLogger(__name__)

PBP_URL_TEMPLATE = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
_CACHE_DIR = config.DATA_RAW / "pressure"
_TEAM_GAME_CACHE = _CACHE_DIR / "team_game_sacks.csv"

LEAGUE_AVG_SACK_RATE = 0.065   # rough long-run NFL sack-rate-per-dropback, used only as the shrinkage prior
SHRINKAGE_DROPBACKS = 30.0     # pseudo-dropbacks of "prior belief = league average" blended in (~1 game's worth)
SEASON_CARRYOVER = 0.4         # fraction of a team's prior-season sample weight kept at a season boundary


def _team_game_sacks_for_season(season: int) -> pd.DataFrame:
    """Download one season's play-by-play and reduce it to one row per
    team per game: sacks taken / dropbacks (offense), sacks recorded /
    opponent dropbacks (defense)."""
    url = PBP_URL_TEMPLATE.format(season=season)
    logger.info("Fetching play-by-play for the %s season (~20MB, one-time per season)...", season)
    df = pd.read_csv(url, compression="gzip", low_memory=False,
                      usecols=["game_id", "season", "week", "posteam", "defteam", "sack", "qb_dropback"])
    plays = df[df["qb_dropback"] == 1]

    off = (plays.groupby(["game_id", "season", "week", "posteam"])["sack"]
           .agg(off_sacks="sum", off_dropbacks="count")
           .reset_index().rename(columns={"posteam": "team"}))
    de = (plays.groupby(["game_id", "season", "week", "defteam"])["sack"]
          .agg(def_sacks="sum", def_dropbacks="count")
          .reset_index().rename(columns={"defteam": "team"}))
    return off.merge(de, on=["game_id", "season", "week", "team"], how="outer")


def refresh_team_game_sacks(seasons: list[int], force_current: bool = True) -> pd.DataFrame:
    """Load (from cache) or fetch team-game sack counts for each requested
    season. Same caching/fallback behavior as epa.refresh_team_game_epa:
    a completed season is cached forever, the latest requested season is
    refetched every call (more weeks appear as it progresses) unless
    force_current=False, and a failed fetch falls back to whatever's
    cached rather than crashing."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = pd.read_csv(_TEAM_GAME_CACHE) if _TEAM_GAME_CACHE.exists() else pd.DataFrame()

    latest = max(seasons) if seasons else None
    frames = []
    for s in seasons:
        have_cached = not cached.empty and s in cached["season"].unique()
        need_fetch = (not have_cached) or (s == latest and force_current)
        if not need_fetch:
            frames.append(cached[cached["season"] == s])
            continue
        try:
            frames.append(_team_game_sacks_for_season(s))
        except Exception as e:
            logger.warning("Could not fetch play-by-play for %s season: %s", s, e)
            if have_cached:
                frames.append(cached[cached["season"] == s])
            # else: no data for this season -- PressureModel.fit() just
            # sees no rows for it and every team stays at its shrinkage
            # prior for that season's games, same fallback as epa.py.

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["game_id", "season", "week", "team", "off_sacks", "off_dropbacks", "def_sacks", "def_dropbacks"])
    if not result.empty:
        result.to_csv(_TEAM_GAME_CACHE, index=False)
    return result


@dataclass
class PressureModel:
    """Walk-forward rolling sack rate, mirroring epa.EPAModel's shape:
    fit() walks completed games chronologically, recording each game's
    PRE-game (no-lookahead) diff before folding that game's result in;
    current_diff_for() exposes the same pre-game state for a not-yet-
    played game."""
    off: dict = field(default_factory=dict)          # team -> (sacks_taken, dropbacks), offense/protection
    deff: dict = field(default_factory=dict)          # team -> (sacks_forced, opp_dropbacks), defense/pass-rush
    last_season: dict = field(default_factory=dict)   # team -> last season folded in

    def _current(self, store: dict, team: str) -> float:
        s, n = store.get(team, (0.0, 0.0))
        prior_s = LEAGUE_AVG_SACK_RATE * SHRINKAGE_DROPBACKS
        return (s + prior_s) / (n + SHRINKAGE_DROPBACKS)

    def _carry_over_if_new_season(self, team: str, season) -> None:
        seen = self.last_season.get(team)
        if seen is not None and seen != season:
            for store in (self.off, self.deff):
                if team in store:
                    s, n = store[team]
                    store[team] = (s * SEASON_CARRYOVER, n * SEASON_CARRYOVER)
        self.last_season[team] = season

    def _update(self, store: dict, team: str, sacks: float, dropbacks: float) -> None:
        s, n = store.get(team, (0.0, 0.0))
        store[team] = (s + sacks, n + dropbacks)

    def fit(self, completed_games: pd.DataFrame, team_game_sacks: pd.DataFrame) -> pd.DataFrame:
        """`completed_games` must be sorted chronologically (season, week)
        and have game_id/season/week/home_team/away_team, same as what's
        passed to elo.EloModel.fit()/epa.EPAModel.fit(). Returns one row
        per game_id with the PRE-game sack_allowed_diff/sack_generated_diff."""
        lookup = {(r.game_id, r.team): (r.off_sacks, r.off_dropbacks, r.def_sacks, r.def_dropbacks)
                  for r in team_game_sacks.itertuples(index=False)} if not team_game_sacks.empty else {}
        rows = []
        for g in completed_games.itertuples(index=False):
            self._carry_over_if_new_season(g.home_team, g.season)
            self._carry_over_if_new_season(g.away_team, g.season)

            home_off_pre = self._current(self.off, g.home_team)
            away_off_pre = self._current(self.off, g.away_team)
            home_def_pre = self._current(self.deff, g.home_team)
            away_def_pre = self._current(self.deff, g.away_team)

            rows.append({
                "game_id": g.game_id,
                "sack_allowed_diff": away_off_pre - home_off_pre,
                "sack_generated_diff": home_def_pre - away_def_pre,
            })

            home_actual = lookup.get((g.game_id, g.home_team))
            if home_actual is not None:
                off_s, off_n, def_s, def_n = home_actual
                if pd.notna(off_s) and pd.notna(off_n):
                    self._update(self.off, g.home_team, off_s, off_n)
                if pd.notna(def_s) and pd.notna(def_n):
                    self._update(self.deff, g.home_team, def_s, def_n)
            away_actual = lookup.get((g.game_id, g.away_team))
            if away_actual is not None:
                off_s, off_n, def_s, def_n = away_actual
                if pd.notna(off_s) and pd.notna(off_n):
                    self._update(self.off, g.away_team, off_s, off_n)
                if pd.notna(def_s) and pd.notna(def_n):
                    self._update(self.deff, g.away_team, def_s, def_n)

        return pd.DataFrame(rows)

    def current_diff_for(self, home_team: str, away_team: str) -> tuple[float, float]:
        """For a not-yet-played game: current (pre-game) sack_allowed_diff,
        sack_generated_diff, using state as of everything fit() has
        processed."""
        allowed_diff = self._current(self.off, away_team) - self._current(self.off, home_team)
        generated_diff = self._current(self.deff, home_team) - self._current(self.deff, away_team)
        return allowed_diff, generated_diff
