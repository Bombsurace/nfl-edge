"""
A walk-forward QB rating, so a starter change (injury, benching, rookie
debut) shows up as its own signal instead of being buried inside the
team's blended Elo number.

Honest limitation up front: this is NOT a true player-value model (that
needs play-by-play/EPA data this project doesn't pull in yet). It's a
538-style proxy: each QB gets their own Elo-like rating, seeded at their
team's rating when they make their first start (a reasonable prior --
their team's current strength), then updated off the same game results
their team Elo updates from, but with a smaller K so a QB gets partial,
not full, credit for a team result. The signal used downstream is the
*gap* between a game's starting QB rating and their team's own (whole-
roster) Elo at that time -- a backup starting for an injured QB1 shows up
as a negative gap; a QB who's outperforming the roster around him shows
up positive. Treat it as a useful proxy, not gospel -- it's exactly the
kind of feature that should earn its coefficient in the backtest, not be
assumed to matter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

QB_K = 8.0  # smaller than team K (20) -- a QB gets partial credit for a team result


@dataclass
class QBRatingModel:
    k: float = QB_K
    ratings: dict = field(default_factory=dict)  # qb_name -> rating
    history: pd.DataFrame | None = None

    def _get(self, qb: str, team_rating_fallback: float) -> float:
        if qb not in self.ratings:
            self.ratings[qb] = team_rating_fallback  # seed at team's current strength
        return self.ratings[qb]

    def fit(self, games: pd.DataFrame, team_elo_history: pd.DataFrame) -> pd.DataFrame:
        """`games` must be sorted chronologically and have home_qb_name/
        away_qb_name + final scores. `team_elo_history` is the DataFrame
        returned by EloModel.fit() (for the pre-game team ratings each QB
        gets seeded/compared against)."""
        team_pre = team_elo_history.set_index("game_id")[["home_rating_pre", "away_rating_pre"]]
        rows = []
        for g in games.itertuples(index=False):
            if g.game_id not in team_pre.index:
                continue
            home_team_elo = team_pre.loc[g.game_id, "home_rating_pre"]
            away_team_elo = team_pre.loc[g.game_id, "away_rating_pre"]

            home_qb, away_qb = g.home_qb_name, g.away_qb_name
            if pd.isna(home_qb) or pd.isna(away_qb):
                continue

            home_qb_r = self._get(home_qb, home_team_elo)
            away_qb_r = self._get(away_qb, away_team_elo)

            rows.append({
                "game_id": g.game_id, "home_qb_name": home_qb, "away_qb_name": away_qb,
                "home_qb_rating_pre": home_qb_r, "away_qb_rating_pre": away_qb_r,
                "home_qb_gap": home_qb_r - home_team_elo,   # >0: starter outperforming roster's blended rating
                "away_qb_gap": away_qb_r - away_team_elo,
            })

            home_score, away_score = g.home_score, g.away_score
            if home_score > away_score:
                actual = 1.0
            elif home_score < away_score:
                actual = 0.0
            else:
                actual = 0.5
            # Same-form update as team Elo, but referencing the *team's*
            # pre-game rating for the win-probability baseline (so a QB's
            # rating measures over/under-performance vs. their own roster,
            # not vs. a neutral 1500).
            elo_diff = home_team_elo - away_team_elo
            p_home = 1.0 / (10 ** (-elo_diff / 400.0) + 1.0)
            delta = self.k * (actual - p_home)
            self.ratings[home_qb] = home_qb_r + delta
            self.ratings[away_qb] = away_qb_r - delta

        self.history = pd.DataFrame(rows)
        return self.history

    def current_gap(self, qb_name: str, team_current_elo: float) -> float:
        """QB rating minus team rating, for a not-yet-played game."""
        r = self.ratings.get(qb_name, team_current_elo)
        return r - team_current_elo
