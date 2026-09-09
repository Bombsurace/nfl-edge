"""
Elo rating engine for NFL team strength.

Why Elo instead of jumping straight to a black-box classifier: with three
seasons of games (~800 rows) and no play-by-play features wired up yet, a
walk-forward Elo system is a much harder baseline to beat than it looks —
it's what most public NFL models (including FiveThirtyEight's, which this
follows closely) are built on. It has three properties this project needs:

1. No lookahead by construction. Every prediction uses only information
   known before kickoff, so a backtest of it is honest by default.
2. Built-in recency weighting. Each game nudges a team's rating; the
   further back a result is, the more it has been overwritten by
   subsequent games. That directly answers the brief's question of
   "should recent seasons count more" without a separate weighting scheme.
3. One number per team, fully inspectable — you can print the rating
   table and sanity-check it against your own sense of the league.

The rating engine itself is fed a longer history than the "training
window" (see config.TrainingWindow) so ratings aren't noisy at the start
of the window. The training window instead controls what's used to *fit
the calibration layer* and what's reported in backtests — see model.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

INITIAL_RATING = 1500.0
SEASON_BASELINE = 1505.0  # long-run mean rating teams regress toward


@dataclass
class EloConfig:
    k: float = 20.0
    home_field_advantage: float = 55.0
    season_regression: float = 1.0 / 3.0  # fraction reverted to baseline each new season
    use_margin_of_victory: bool = True


@dataclass
class EloModel:
    cfg: EloConfig = field(default_factory=EloConfig)
    ratings: dict = field(default_factory=dict)
    history: pd.DataFrame | None = None  # filled in by fit()
    last_season: int | None = None  # last season whose games have been folded into `ratings`

    def _get(self, team: str) -> float:
        return self.ratings.get(team, INITIAL_RATING)

    def _regress_all(self) -> None:
        r = self.cfg.season_regression
        for team in list(self.ratings.keys()):
            self.ratings[team] = self.ratings[team] * (1 - r) + SEASON_BASELINE * r

    def advance_to_season(self, season: int) -> None:
        """Apply the between-season mean-reversion for every season boundary
        crossed between the last season fit() actually processed and
        `season`. Must be called before predict() targets a season with no
        completed games yet (e.g. predicting week 1 of a brand-new season) --
        otherwise predictions would use raw end-of-prior-season ratings with
        none of the regression-to-the-mean that keeps a historically bad or
        good team from being treated as a lock to stay that way."""
        if self.last_season is None:
            self.last_season = season
            return
        while self.last_season < season:
            self._regress_all()
            self.last_season += 1

    @staticmethod
    def prob_from_diff(elo_diff: float) -> float:
        return 1.0 / (10 ** (-elo_diff / 400.0) + 1.0)

    def fit(self, games: pd.DataFrame) -> pd.DataFrame:
        """Walk chronologically through completed games, updating ratings.

        `games` must be sorted by (season, week, gameday) and contain only
        games with final scores. Returns a row per game with the pre-game
        ratings and elo-implied home win probability used at prediction
        time, plus the actual outcome — this is what model.py trains the
        calibration layer on.
        """
        rows = []
        current_season = None
        for g in games.itertuples(index=False):
            if current_season is None:
                current_season = g.season
            elif g.season != current_season:
                self._regress_all()
                current_season = g.season
            self.last_season = current_season

            home_r = self._get(g.home_team)
            away_r = self._get(g.away_team)
            elo_diff = (home_r + self.cfg.home_field_advantage) - away_r
            p_home = self.prob_from_diff(elo_diff)

            home_score, away_score = g.home_score, g.away_score
            if home_score > away_score:
                actual = 1.0
            elif home_score < away_score:
                actual = 0.0
            else:
                actual = 0.5

            rows.append({
                "game_id": g.game_id, "season": g.season, "week": g.week,
                "home_team": g.home_team, "away_team": g.away_team,
                "home_rating_pre": home_r, "away_rating_pre": away_r,
                "elo_prob_home": p_home, "home_win": actual,
                "home_score": home_score, "away_score": away_score,
            })

            if self.cfg.use_margin_of_victory:
                margin = abs(home_score - away_score)
                winner_elo_diff = elo_diff if actual >= 0.5 else -elo_diff
                mov_mult = math.log(margin + 1) * (2.2 / (0.001 * abs(winner_elo_diff) + 2.2))
            else:
                mov_mult = 1.0

            k_adj = self.cfg.k * mov_mult
            delta = k_adj * (actual - p_home)
            self.ratings[g.home_team] = home_r + delta
            self.ratings[g.away_team] = away_r - delta

        self.history = pd.DataFrame(rows)
        return self.history

    def predict(self, upcoming: pd.DataFrame) -> pd.DataFrame:
        """Predict elo win probability for games not yet played, using
        current ratings (i.e. after fit() has processed everything known
        so far). Does not update ratings."""
        rows = []
        for g in upcoming.itertuples(index=False):
            home_r = self._get(g.home_team)
            away_r = self._get(g.away_team)
            elo_diff = (home_r + self.cfg.home_field_advantage) - away_r
            p_home = self.prob_from_diff(elo_diff)
            rows.append({
                "game_id": g.game_id, "season": g.season, "week": g.week,
                "home_team": g.home_team, "away_team": g.away_team,
                "home_rating_pre": home_r, "away_rating_pre": away_r,
                "elo_prob_home": p_home,
            })
        return pd.DataFrame(rows)

    def ratings_table(self) -> pd.DataFrame:
        return (pd.DataFrame([{"team": t, "rating": r} for t, r in self.ratings.items()])
                .sort_values("rating", ascending=False).reset_index(drop=True))
