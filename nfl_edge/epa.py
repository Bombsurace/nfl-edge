"""
Offensive and defensive efficiency (EPA per play), walked forward with the
same no-lookahead discipline as elo.py/qb.py.

Source: nflverse's public play-by-play release (raw play-level data, free,
no key) -- https://github.com/nflverse/nflverse-data/releases/tag/pbp.
This reduces it, once per season, to a single number per team per game:
mean EPA (expected points added) per offensive snap, and per defensive
snap allowed, restricted to real pass/run plays (excludes kneels, spikes,
special teams, no-plays) -- the standard definition used across public NFL
analytics (nflfastR, rbsdm.com, and similar sites).

Two factors come out of this, both signed so positive = an edge for the
home team, matching every other factor column in this project:

- epa_off_diff: home's rolling offensive EPA/play minus away's.
- epa_def_diff: away's rolling defensive EPA/play ALLOWED minus home's
  (higher EPA allowed = a worse defense, so this being positive means
  away's defense has been leakier than home's -- an edge for home).

"Rolling" here is season-to-date, shrunk toward a league-average prior of
0.0 with a strength equal to SHRINKAGE_GAMES games -- so a team's week-2
number isn't wildly swung by one fluky game -- and a new season starts
from a partial carryover of where the team's rating settled the previous
season (SEASON_CARRYOVER), not a hard reset to zero (roster turnover means
a full carryover would be wrong too). Both constants are judgment calls,
not fit from data -- documented here rather than hidden, same as the rest
of this project's approximations.

Known limitation, stated plainly: this is NOT opponent-adjusted. A team's
offensive EPA reflects who it has actually played, not a neutral schedule
-- a team that has faced three bad defenses will look better here than it
may truly be. A proper fix (iterative opponent adjustment, like SRS or a
mini-Elo per side) is future work; treat this the same way as QB rating
and weather elsewhere in this project -- a real, disclosed approximation,
not a finished stat. It also doesn't filter garbage time (a blowout's
prevent-defense snaps count the same as a close game's), another common
refinement this v1 skips.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from . import config

logger = logging.getLogger(__name__)

PBP_URL_TEMPLATE = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
_CACHE_DIR = config.DATA_RAW / "epa"
_TEAM_GAME_CACHE = _CACHE_DIR / "team_game_epa.csv"

SHRINKAGE_GAMES = 4.0      # pseudo-games of "prior belief = league average (0)" blended in
SEASON_CARRYOVER = 0.4     # fraction of a team's prior-season sample weight kept at a season boundary


def _team_game_epa_for_season(season: int) -> pd.DataFrame:
    """Download one season's play-by-play and reduce it to one row per
    team per game: mean EPA/play on offense, and mean EPA/play allowed on
    defense, restricted to real pass/run plays."""
    url = PBP_URL_TEMPLATE.format(season=season)
    logger.info("Fetching play-by-play for the %s season (~20MB, one-time per season)...", season)
    df = pd.read_csv(url, compression="gzip", low_memory=False,
                      usecols=["game_id", "season", "week", "posteam", "defteam", "epa", "pass", "rush"])
    plays = df[((df["pass"] == 1) | (df["rush"] == 1)) & df["epa"].notna()]

    off = (plays.groupby(["game_id", "season", "week", "posteam"])["epa"]
           .agg(off_epa_play="mean", off_n_plays="count")
           .reset_index().rename(columns={"posteam": "team"}))
    de = (plays.groupby(["game_id", "season", "week", "defteam"])["epa"]
          .agg(def_epa_play="mean", def_n_plays="count")
          .reset_index().rename(columns={"defteam": "team"}))
    return off.merge(de, on=["game_id", "season", "week", "team"], how="outer")


def refresh_team_game_epa(seasons: list[int], force_current: bool = True) -> pd.DataFrame:
    """Load (from cache) or fetch team-game EPA for each requested season.

    A completed season's play-by-play never changes, so it's cached
    forever once fetched. The *latest* requested season is refetched every
    call unless force_current=False, since more weeks of pbp appear as
    that season progresses. Falls back to whatever's cached for a season
    if the network fetch for it fails (matches this project's other
    "never crash on a network hiccup, just note it" fallback behavior)."""
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
            frames.append(_team_game_epa_for_season(s))
        except Exception as e:
            logger.warning("Could not fetch play-by-play for %s season: %s", s, e)
            if have_cached:
                frames.append(cached[cached["season"] == s])
            # else: no data for this season at all -- EPAModel.fit() just
            # sees no rows for it and every team stays at its shrinkage
            # prior (0.0) for that season's games, same as any other
            # missing-data fallback in this project.

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["game_id", "season", "week", "team", "off_epa_play", "off_n_plays", "def_epa_play", "def_n_plays"])
    if not result.empty:
        result.to_csv(_TEAM_GAME_CACHE, index=False)
    return result


@dataclass
class EPAModel:
    """Walk-forward rolling EPA/play, mirroring elo.EloModel's shape:
    fit() walks completed games chronologically, recording each game's
    PRE-game (no-lookahead) diff before folding that game's result in;
    current_diff_for() exposes the same pre-game state for a not-yet-
    played game."""
    off: dict = field(default_factory=dict)          # team -> (sum, n)
    deff: dict = field(default_factory=dict)          # team -> (sum, n), defensive EPA allowed
    last_season: dict = field(default_factory=dict)   # team -> last season folded in

    def _current(self, store: dict, team: str) -> float:
        s, n = store.get(team, (0.0, 0.0))
        return s / (n + SHRINKAGE_GAMES)

    def _carry_over_if_new_season(self, team: str, season) -> None:
        seen = self.last_season.get(team)
        if seen is not None and seen != season:
            for store in (self.off, self.deff):
                if team in store:
                    s, n = store[team]
                    store[team] = (s * SEASON_CARRYOVER, n * SEASON_CARRYOVER)
        self.last_season[team] = season

    def _update(self, store: dict, team: str, value: float) -> None:
        s, n = store.get(team, (0.0, 0.0))
        store[team] = (s + value, n + 1.0)

    def fit(self, completed_games: pd.DataFrame, team_game_epa: pd.DataFrame) -> pd.DataFrame:
        """`completed_games` must be sorted chronologically (season, week)
        and have game_id/season/week/home_team/away_team, same as what's
        passed to elo.EloModel.fit()/qb.QBRatingModel.fit(). Returns one
        row per game_id with the PRE-game epa_off_diff/epa_def_diff."""
        lookup = {(r.game_id, r.team): (r.off_epa_play, r.def_epa_play)
                  for r in team_game_epa.itertuples(index=False)} if not team_game_epa.empty else {}
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
                "epa_off_diff": home_off_pre - away_off_pre,
                "epa_def_diff": away_def_pre - home_def_pre,
            })

            home_actual = lookup.get((g.game_id, g.home_team))
            if home_actual is not None:
                if pd.notna(home_actual[0]):
                    self._update(self.off, g.home_team, home_actual[0])
                if pd.notna(home_actual[1]):
                    self._update(self.deff, g.home_team, home_actual[1])
            away_actual = lookup.get((g.game_id, g.away_team))
            if away_actual is not None:
                if pd.notna(away_actual[0]):
                    self._update(self.off, g.away_team, away_actual[0])
                if pd.notna(away_actual[1]):
                    self._update(self.deff, g.away_team, away_actual[1])

        return pd.DataFrame(rows)

    def current_diff_for(self, home_team: str, away_team: str) -> tuple[float, float]:
        """For a not-yet-played game: current (pre-game) epa_off_diff,
        epa_def_diff, using state as of everything fit() has processed."""
        off_diff = self._current(self.off, home_team) - self._current(self.off, away_team)
        def_diff = self._current(self.deff, away_team) - self._current(self.deff, home_team)
        return off_diff, def_diff
