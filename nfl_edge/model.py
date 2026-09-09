"""
Wires the Elo engine to a calibration layer and exposes the small surface
the rest of the pipeline needs: fit on history, predict a given week.

Two things are deliberately kept separate:

- Elo *rating* history: needs several seasons of warm-up before games in
  the training window so ratings aren't starting from a flat 1500 with no
  information. Controlled by `warmup_seasons`.
- The *training/evaluation window* (config.TrainingWindow, default
  2023-2025): the seasons actually used to (a) fit the calibration curve
  and (b) report backtest performance. This is the number to change when
  testing "should the window be shorter/longer/weighted".

Calibration: isotonic regression mapping raw Elo probability -> observed
win rate. This is what section 7 of the brief calls p_home_cal. The
priority order p_adj > p_home_cal > p_home is implemented in `predict()`:
p_adj is left as a hook for a future manual/situational adjustment layer
(injuries, weather, etc. -- see section 14) that this project does not
yet populate, so today p_adj is always None and p_home_cal wins whenever
it's fitted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from . import data
from .config import TrainingWindow, DEFAULT_TRAINING_WINDOW
from .elo import EloConfig, EloModel

DEFAULT_WARMUP_SEASONS = 8  # seasons of history before the training window, for rating warm-up


def _logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """Smooth 2-parameter (slope + intercept) logistic recalibration of
    raw probabilities. Preferred default over sklearn's IsotonicRegression
    for a dataset this size (~800-900 training games): isotonic has no
    smoothness prior, so with this few games it tends to carve out wide
    flat 'plateaus' where dozens of genuinely different games all get
    mapped to the identical calibrated probability -- that's the
    calibrator overfitting noise in the bucket boundaries, not a real
    finding about the model. Platt scaling can't do that; it can only
    stretch or compress the sigmoid.
    """
    def __init__(self):
        self._lr: LogisticRegression | None = None

    def fit(self, p_raw, y, sample_weight=None) -> "PlattCalibrator":
        X = _logit(np.asarray(p_raw)).reshape(-1, 1)
        self._lr = LogisticRegression()
        self._lr.fit(X, np.asarray(y), sample_weight=sample_weight)
        return self

    def predict(self, p_raw):
        X = _logit(np.asarray(p_raw)).reshape(-1, 1)
        return self._lr.predict_proba(X)[:, 1]


@dataclass
class NFLWinProbModel:
    training_window: TrainingWindow = field(default_factory=lambda: DEFAULT_TRAINING_WINDOW)
    elo_cfg: EloConfig = field(default_factory=EloConfig)
    warmup_seasons: int = DEFAULT_WARMUP_SEASONS
    calibration_method: str = "platt"  # "platt" (default, recommended) or "isotonic"

    elo: EloModel = field(init=False, default=None)
    calibrator: object = field(init=False, default=None)
    history: pd.DataFrame = field(init=False, default=None)
    fit_seasons: list = field(init=False, default=None)

    def fit(self, refresh_data: bool = False) -> "NFLWinProbModel":
        start_season = min(self.training_window.seasons) - self.warmup_seasons
        end_season = max(self.training_window.seasons)
        self.fit_seasons = list(range(start_season, end_season + 1))

        games = data.load_games(seasons=self.fit_seasons, refresh=refresh_data)
        completed = data.completed_games(games)

        self.elo = EloModel(cfg=self.elo_cfg)
        history = self.elo.fit(completed)
        self.history = history

        # Fit calibration only on the training window itself (not the
        # warmup seasons) -- we want the calibration curve to reflect
        # *current* model behavior, not how well-converged Elo was 8
        # seasons ago while ratings were still warming up.
        # Drop ties (home_win == 0.5, exceedingly rare in the NFL) -- Platt
        # scaling needs a binary target, and a handful of ties have no
        # meaningful effect on either calibration method either way.
        window_mask = history["season"].isin(self.training_window.seasons) & history["home_win"].isin([0.0, 1.0])
        cal_df = history[window_mask]

        sample_weight = cal_df["season"].map(self.training_window.season_weights).fillna(1.0)
        if self.calibration_method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
        elif self.calibration_method == "platt":
            self.calibrator = PlattCalibrator()
        else:
            raise ValueError(f"Unknown calibration_method: {self.calibration_method!r}")
        self.calibrator.fit(cal_df["elo_prob_home"], cal_df["home_win"], sample_weight=sample_weight)
        return self

    def predict_week(self, season: int, week: int, refresh_data: bool = False) -> pd.DataFrame:
        """Predict win probabilities for one week's games (played or not).

        Returns one row per game with p_home (raw elo), p_home_cal
        (isotonic-calibrated), p_adj (reserved, currently None), and
        p_final = coalesce(p_adj, p_home_cal, p_home) per section 7 of
        the brief.
        """
        if self.elo is None:
            raise RuntimeError("Call .fit() before predict_week().")
        games = data.load_games(seasons=[season], refresh=refresh_data)
        wk = data.upcoming_games(games, season, week)
        if wk.empty:
            raise ValueError(f"No games found for season={season} week={week}")

        # Make sure any season boundary between the last season fit() saw
        # completed games for and this one has had its mean-reversion
        # applied -- see EloModel.advance_to_season for why this matters.
        self.elo.advance_to_season(season)
        preds = self.elo.predict(wk)
        preds["p_home"] = preds["elo_prob_home"]
        preds["p_home_cal"] = self.calibrator.predict(preds["p_home"])
        preds["p_adj"] = np.nan  # hook for future situational adjustments
        preds["p_final"] = preds["p_adj"].where(preds["p_adj"].notna(), preds["p_home_cal"])
        preds["p_final"] = preds["p_final"].where(preds["p_final"].notna(), preds["p_home"])

        merge_cols = ["game_id", "gameday", "home_score", "away_score",
                      "home_moneyline", "away_moneyline", "div_game",
                      "home_rest", "away_rest", "game_type"]
        wk_small = wk[merge_cols].copy() if all(c in wk.columns for c in merge_cols) else wk[["game_id"]]
        out = preds.merge(wk_small, on="game_id", how="left")
        return out

    def reliability_report(self, n_bins: int = 10) -> pd.DataFrame:
        """Predicted-vs-actual win rate by probability bucket, on the
        training window only -- the calibration sanity check called for
        in section 7 and section 18 of the brief.

        IMPORTANT: this is an IN-SAMPLE check (the calibrator was fit on
        these same games), so it will always look better than the model's
        true out-of-sample calibration. It's useful for catching a badly
        broken calibrator, not for judging whether calibration is helping.
        For an honest answer to that, use backtest.walk_forward_backtest,
        which refits the calibrator using only games known before each
        prediction."""
        if self.history is None:
            raise RuntimeError("Call .fit() first.")
        h = self.history[self.history["season"].isin(self.training_window.seasons)].copy()
        h["p_cal"] = self.calibrator.predict(h["elo_prob_home"])
        h["bucket"] = pd.cut(h["p_cal"], bins=np.linspace(0, 1, n_bins + 1))
        rep = h.groupby("bucket", observed=True).agg(
            n=("home_win", "size"),
            avg_predicted=("p_cal", "mean"),
            actual_win_rate=("home_win", "mean"),
        ).reset_index()
        return rep
