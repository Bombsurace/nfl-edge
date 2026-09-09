"""
The multi-factor model: team strength (Elo) plus named, independently-
weighted adjustments -- QB, weather, extra home-field emphasis -- combined
through a logistic regression instead of Elo's single number.

Design, and why it's built this way:

- Each factor becomes one standardized column (roughly logit-scale, so
  they combine linearly and sensibly): `elo_diff` (Elo's own home-minus-
  away edge, /400), `qb_diff` (home QB gap minus away QB gap from qb.py,
  /400), `weather_severity` (0-1, how nasty the conditions are -- sign of
  its effect on home_win is NOT assumed, it's fit from data), `home_extra`
  (a constant 1.0 for every game -- a second, separate home-field term on
  top of whatever HFA Elo already bakes in), `rest_diff` and `div_game`
  (schedule-derived), `epa_off_diff` / `epa_def_diff` (rolling,
  season-to-date offensive and defensive EPA-per-play differentials from
  real play-by-play data -- see epa.py for how they're built and their
  documented limitations, chiefly that they're not opponent-adjusted), and
  `sack_allowed_diff` / `sack_generated_diff` (rolling sack-rate
  differentials -- protection and pass rush, approximated by sack rate
  since true pressure-charting data isn't public -- see pressure.py), and
  `travel_distance_diff` / `travel_tz_diff` (how much farther, and how
  many more time zones, the away team travelled than the home team for
  this specific game -- fixed geography, not fit history, so unlike EPA/
  pressure this needs no walk-forward state at all; see travel.py).
- Base coefficients are FIT by logistic regression on real outcomes
  (`fit_base_weights`), not guessed. That's the data-driven starting
  point.
- A "weight" per factor is a multiplier on top of that fitted
  coefficient (1.0 = trust the data as-is; 0 = ignore the factor; 2.0 =
  lean on it twice as hard as the data alone suggested). This is what
  the interactive slider page exposes -- real, meaningful control, but
  anchored to a real fit rather than an arbitrary number.
- `walk_forward_multifactor_backtest` is the rigorous check: base
  coefficients are refit weekly using only prior window games (same
  no-lookahead discipline as backtest.py), for whatever slider weights
  you pass in. The interactive page's live numbers use coefficients fit
  ONCE on the whole window for speed -- clearly labeled there as an
  exploration shortcut, with this function as the honest version to
  confirm anything that looks interesting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import data, ev
from .config import GuardrailConfig, KellyConfig, DEFAULT_GUARDRAILS, DEFAULT_KELLY, TrainingWindow, DEFAULT_TRAINING_WINDOW
from .elo import EloConfig, EloModel
from .epa import EPAModel, refresh_team_game_epa
from .pressure import PressureModel, refresh_team_game_sacks
from .qb import QBRatingModel
from .travel import travel_features

FACTOR_COLS = ["elo_diff", "qb_diff", "weather_severity", "home_extra", "rest_diff", "div_game",
               "epa_off_diff", "epa_def_diff", "sack_allowed_diff", "sack_generated_diff",
               "travel_distance_diff", "travel_tz_diff"]
MIN_GAMES_TO_FIT = 100
EPA_WARMUP_SEASONS = 1        # EPA's own shrinkage+carryover need far less warmup than Elo's 8 seasons
PRESSURE_WARMUP_SEASONS = 1   # same rationale -- pressure.py uses the same shrinkage/carryover machinery


def weather_severity(temp, wind, roof) -> float:
    """0 (calm/indoor) to ~1 (brutal). Direction of its effect on
    home_win is deliberately NOT assumed here -- that's for the logistic
    fit to discover. Domes/closed roofs and missing readings are neutral."""
    if roof in ("dome", "closed") or pd.isna(temp) or pd.isna(wind):
        return 0.0
    cold = max(0.0, (40.0 - float(temp)) / 40.0)   # ramps up below 40F
    windy = min(float(wind), 25.0) / 25.0
    return float(np.clip(0.5 * cold + 0.5 * windy, 0, 1))


@dataclass
class FeatureBuild:
    history: pd.DataFrame          # per-game features + home_win, training window only
    elo: EloModel
    qb: QBRatingModel
    epa: EPAModel
    pressure: PressureModel


def build_features(training_window: TrainingWindow = DEFAULT_TRAINING_WINDOW,
                    elo_cfg: EloConfig = EloConfig(), warmup_seasons: int = 8,
                    refresh_data: bool = False) -> FeatureBuild:
    seasons = training_window.as_list()
    start_season = min(seasons) - warmup_seasons
    end_season = max(seasons)
    all_games = data.load_games(seasons=list(range(start_season, end_season + 1)), refresh=refresh_data)
    completed = data.completed_games(all_games)

    elo = EloModel(cfg=elo_cfg)
    elo_hist = elo.fit(completed)

    qb = QBRatingModel()
    qb_hist = qb.fit(completed, elo_hist)

    # EPA needs far less warm-up than Elo (shrinkage + season carryover
    # already tame early-season noise), so it's fit over a much shorter
    # slice of `completed` -- keeps the play-by-play download small.
    epa_start_season = min(seasons) - EPA_WARMUP_SEASONS
    epa_completed = completed[completed["season"] >= epa_start_season]
    team_game_epa = refresh_team_game_epa(sorted(epa_completed["season"].unique().tolist()))
    epa = EPAModel()
    epa_hist = epa.fit(epa_completed, team_game_epa)

    # Same reasoning as EPA above -- sack rate's own shrinkage/carryover
    # need far less warm-up than Elo's.
    pressure_start_season = min(seasons) - PRESSURE_WARMUP_SEASONS
    pressure_completed = completed[completed["season"] >= pressure_start_season]
    team_game_sacks = refresh_team_game_sacks(sorted(pressure_completed["season"].unique().tolist()))
    pressure = PressureModel()
    pressure_hist = pressure.fit(pressure_completed, team_game_sacks)

    merged = elo_hist.merge(qb_hist, on="game_id", how="left")
    merged = merged.merge(epa_hist, on="game_id", how="left")
    merged = merged.merge(pressure_hist, on="game_id", how="left")
    ctx_cols = ["game_id", "home_moneyline", "away_moneyline", "temp", "wind", "roof",
                "div_game", "game_type", "home_rest", "away_rest",
                "spread_line", "home_spread_odds", "away_spread_odds", "stadium_id"]
    merged = merged.merge(completed[ctx_cols], on="game_id", how="left")

    merged["elo_diff"] = (merged["home_rating_pre"] + elo_cfg.home_field_advantage - merged["away_rating_pre"]) / 400.0
    # Direct QB-vs-QB rating differential (NOT each QB's gap vs. their own
    # team's Elo -- an earlier version tried that and it produced a
    # mechanical artifact: team Elo updates faster (K=20) than QB rating
    # (K=8), so during any team hot/cold streak the "gap" moves for
    # reasons that have nothing to do with the QB, and it even predicted
    # home_win with the wrong sign on its own. The direct differential
    # below is sane on its own (positive coefficient, as expected) -- see
    # docs/DESIGN.md for what happens once it's combined with elo_diff.
    merged["qb_diff"] = (merged["home_qb_rating_pre"] - merged["away_qb_rating_pre"]) / 400.0
    merged["weather_severity"] = merged.apply(lambda r: weather_severity(r["temp"], r["wind"], r["roof"]), axis=1)
    merged["home_extra"] = 1.0
    # Rest advantage: days since each team's last game, home minus away,
    # normalized by a standard week. A short week (Thursday game off a
    # normal Sunday) shows up as strongly negative for the team facing it;
    # a bye-week return shows up positive. Both teams default to a normal
    # week (7 days, diff 0) if rest data is somehow missing.
    merged["rest_diff"] = ((merged["home_rest"].fillna(7) - merged["away_rest"].fillna(7)) / 7.0)
    # Divisional game: already a clean 0/1 flag in the source data --
    # division rivals know each other well, which some public modeling
    # (and plenty of bettor folklore) treats as a closer-than-the-rating-
    # suggests game. Fed in as-is; let the fit decide if that's real here.
    merged["div_game"] = merged["div_game"].fillna(0).astype(float)
    # Travel distance / time-zone shift: fixed geography, known for every
    # game (even future ones) -- no walk-forward state needed, see
    # travel.py. Computed row-wise from each game's actual stadium_id.
    travel_cols = merged.apply(
        lambda r: travel_features(r["home_team"], r["away_team"], r["stadium_id"]), axis=1, result_type="expand")
    merged["travel_distance_diff"] = travel_cols[0]
    merged["travel_tz_diff"] = travel_cols[1]
    # Rows before epa_start_season (Elo's deeper warmup range) never got an
    # EPA fit -- NaN here, but those rows are dropped below anyway since
    # they're outside the training window.
    merged["epa_off_diff"] = merged["epa_off_diff"].fillna(0.0)
    merged["epa_def_diff"] = merged["epa_def_diff"].fillna(0.0)
    # Same fallback as EPA: rows before pressure_start_season never got a
    # sack-rate fit -- NaN here, but those rows are outside the training
    # window and dropped below anyway.
    merged["sack_allowed_diff"] = merged["sack_allowed_diff"].fillna(0.0)
    merged["sack_generated_diff"] = merged["sack_generated_diff"].fillna(0.0)

    window_hist = merged[merged["season"].isin(seasons)].copy().sort_values(["season", "week"]).reset_index(drop=True)
    return FeatureBuild(history=window_hist, elo=elo, qb=qb, epa=epa, pressure=pressure)


def fit_base_weights(df: pd.DataFrame, feature_cols=FACTOR_COLS) -> dict:
    """One logistic regression, in-sample on `df`. Returns {feature: coef}
    plus 'intercept'. Used both for the (in-sample, fast) exploration page
    and, refit repeatedly on growing prior-only slices, for the honest
    walk-forward backtest below."""
    d = df[df["home_win"].isin([0.0, 1.0])]
    if len(d) < MIN_GAMES_TO_FIT:
        return {c: 0.0 for c in feature_cols} | {"intercept": 0.0, "n_fit": len(d)}
    X = d[feature_cols].values
    y = d["home_win"].values
    lr = LogisticRegression()
    lr.fit(X, y)
    weights = dict(zip(feature_cols, lr.coef_[0]))
    weights["intercept"] = float(lr.intercept_[0])
    weights["n_fit"] = len(d)
    return weights


def apply_weights(row, weights: dict, multipliers: dict | None = None) -> float:
    """logit(p_home) for one row, given base weights and optional
    per-factor multipliers (the 'sliders'; default 1.0 each)."""
    multipliers = multipliers or {}
    z = weights.get("intercept", 0.0)
    for f in FACTOR_COLS:
        m = multipliers.get(f, 1.0)
        z += weights.get(f, 0.0) * m * row[f]
    return z


def predict_prob(df: pd.DataFrame, weights: dict, multipliers: dict | None = None) -> pd.Series:
    z = df.apply(lambda r: apply_weights(r, weights, multipliers), axis=1)
    return 1.0 / (1.0 + np.exp(-z))


def walk_forward_multifactor_backtest(
    training_window: TrainingWindow = DEFAULT_TRAINING_WINDOW,
    multipliers: dict | None = None,
    elo_cfg: EloConfig = EloConfig(), warmup_seasons: int = 8,
    gcfg: GuardrailConfig = DEFAULT_GUARDRAILS, kcfg: KellyConfig = DEFAULT_KELLY,
    refresh_data: bool = False,
) -> pd.DataFrame:
    """The rigorous version: base weights refit each week using only prior
    window games, same no-lookahead discipline as backtest.py. `multipliers`
    (e.g. {'qb_diff': 1.5, 'weather_severity': 0.0}) are applied on top of
    whatever that week's fit produced."""
    fb = build_features(training_window, elo_cfg, warmup_seasons, refresh_data)
    hist = fb.history

    rows = []
    for (season, week), wk_games in hist.groupby(["season", "week"], sort=True):
        prior = hist[(hist["season"] < season) | ((hist["season"] == season) & (hist["week"] < week))]
        weights = fit_base_weights(prior)

        for g in wk_games.itertuples(index=False):
            if pd.isna(g.home_moneyline) or pd.isna(g.away_moneyline):
                continue
            row = {c: getattr(g, c) for c in FACTOR_COLS}
            z = weights.get("intercept", 0.0)
            for f in FACTOR_COLS:
                m = (multipliers or {}).get(f, 1.0)
                z += weights.get(f, 0.0) * m * row[f]
            p_home = 1.0 / (1.0 + np.exp(-z))
            p_away = 1 - p_home

            side_results = {}
            for side, team, p_win, odds in (
                ("home", g.home_team, p_home, g.home_moneyline),
                ("away", g.away_team, p_away, g.away_moneyline),
            ):
                res = ev.evaluate_side(p_win=p_win, odds=odds, n_books=1, consensus_gap=0.0, width_dec=0.0,
                                        gcfg=gcfg, enforce_market_quality=False)
                res.update(side=side, team=team)
                side_results[side] = res
            best = max(side_results.values(), key=lambda r: r["edge"])

            if g.home_win == 0.5:
                profit = 0.0
            else:
                won = (best["side"] == "home") == (g.home_win == 1.0)
                profit = kcfg.flat_stake * ev.profit_per_dollar(best["odds"]) if won else -kcfg.flat_stake

            rows.append({
                "season": season, "week": week, "game_id": g.game_id,
                "model_side": best["side"], "model_odds": best["odds"], "model_edge": best["edge"],
                "bet_flag": int(best["bet_flag"]), "model_profit": profit,
                "n_fit": weights.get("n_fit", 0),
            })

    return pd.DataFrame(rows)
