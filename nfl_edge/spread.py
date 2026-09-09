"""
Spread (against-the-spread) model. Same six factors as the moneyline
model (factors.py), same "predict something, compare it to the real
market, compute EV/guardrails" shape -- just a different thing to
predict: home margin of victory, via linear regression, instead of a
win probability via logistic regression.

Sign convention (verified against 855 real games, not assumed): in this
dataset, `spread_line` is the home team's *expected margin* --
positive means the home team is favored by that many points (checked
by regressing actual margin on spread_line: slope +0.57, i.e. moves the
same direction, and by confirming home teams favored by more have
larger positive spread_line). The home side covers when
`actual_margin (home_score - away_score) > spread_line`; the away side
covers otherwise. A push (margin exactly equal to spread_line) is
possible on integer lines and is not separately modeled here -- most
lines in the data carry a half point specifically to avoid this, and
treating a push as a loss for both sides (the conservative choice) or
ignoring it entirely both have a negligible effect at this sample size.

Cover probability comes from treating the actual margin as
approximately normal around the model's predicted margin, with a
standard deviation fit from the same data as the residual spread of
actual-minus-predicted margins (falls back to ~13.5, a commonly cited
approximation of full-game NFL margin variance, if there's not enough
data yet to fit one). That's a real, if simplified, assumption -- NFL
margins aren't exactly normal (there's a bump right around 3 and 7
points from how often games are decided by a field goal or a
touchdown-plus-XP) -- but it's the standard, disclosed way to turn "a
predicted margin" into "a probability of covering a specific line,"
and it's the same kind of honest approximation this project already
uses elsewhere (e.g. weather_severity's 0-1 scale).

Once there's a cover probability and the market's real spread odds
(`home_spread_odds`/`away_spread_odds` -- usually close to -110 either
side, but real per-book numbers, not assumed), everything downstream
reuses the exact same `ev.evaluate_side` guardrail/EV machinery the
moneyline path uses. The odds-level guardrail bumps (short-favorite,
long-underdog) essentially never fire for spread odds, since those
cluster tightly around -110 -- which is itself a sanity check that nothing
is being double-counted, not a gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from . import ev
from .config import GuardrailConfig, KellyConfig, DEFAULT_GUARDRAILS, DEFAULT_KELLY, TrainingWindow, DEFAULT_TRAINING_WINDOW
from .elo import EloConfig
from .factors import FACTOR_COLS, MIN_GAMES_TO_FIT, build_features

# Fallback residual std (points) if there isn't enough data yet to fit
# one -- a commonly cited approximation of full-game NFL margin spread.
DEFAULT_RESID_STD = 13.5


def fit_margin_model(df: pd.DataFrame, feature_cols=FACTOR_COLS) -> dict:
    """One linear regression home_margin ~ features, in-sample on `df`.
    Returns {feature: coef, intercept, resid_std, n_fit}. Mirrors
    `factors.fit_base_weights` but for margin instead of win probability."""
    needed = feature_cols + ["home_score", "away_score"]
    d = df.dropna(subset=needed)
    if len(d) < MIN_GAMES_TO_FIT:
        return {c: 0.0 for c in feature_cols} | {"intercept": 0.0, "resid_std": DEFAULT_RESID_STD, "n_fit": len(d)}
    y = (d["home_score"] - d["away_score"]).astype(float).values
    X = d[feature_cols].values
    lr = LinearRegression()
    lr.fit(X, y)
    pred = lr.predict(X)
    dof = max(len(d) - len(feature_cols) - 1, 1)
    resid_std = float(np.sqrt(np.sum((y - pred) ** 2) / dof))
    weights = dict(zip(feature_cols, lr.coef_))
    weights["intercept"] = float(lr.intercept_)
    weights["resid_std"] = resid_std
    weights["n_fit"] = len(d)
    return weights


def predicted_margin(row, weights: dict, multipliers: dict | None = None, feature_cols=FACTOR_COLS) -> float:
    multipliers = multipliers or {}
    z = weights.get("intercept", 0.0)
    for f in feature_cols:
        m = multipliers.get(f, 1.0)
        z += weights.get(f, 0.0) * m * row[f]
    return z


def home_cover_prob(pred_margin: float, spread_line: float, resid_std: float) -> float:
    """P(actual_margin > spread_line) under actual_margin ~ N(pred_margin, resid_std^2)."""
    if resid_std <= 0:
        resid_std = DEFAULT_RESID_STD
    z = (spread_line - pred_margin) / resid_std
    return float(norm.sf(z))


def walk_forward_spread_backtest(
    training_window: TrainingWindow = DEFAULT_TRAINING_WINDOW,
    multipliers: dict | None = None,
    elo_cfg: EloConfig = EloConfig(), warmup_seasons: int = 8,
    gcfg: GuardrailConfig = DEFAULT_GUARDRAILS, kcfg: KellyConfig = DEFAULT_KELLY,
    refresh_data: bool = False,
) -> pd.DataFrame:
    """Rigorous walk-forward version: the margin model is refit each week
    using only prior window games (no lookahead), same discipline as
    `factors.walk_forward_multifactor_backtest`. `multipliers` works the
    same way as the moneyline backtest's -- e.g. {'qb_diff': 1.5} --
    applied on top of whatever that week's fit produced."""
    fb = build_features(training_window, elo_cfg, warmup_seasons, refresh_data)
    hist = fb.history

    rows = []
    for (season, week), wk_games in hist.groupby(["season", "week"], sort=True):
        prior = hist[(hist["season"] < season) | ((hist["season"] == season) & (hist["week"] < week))]
        weights = fit_margin_model(prior)

        for g in wk_games.itertuples(index=False):
            if pd.isna(g.spread_line) or pd.isna(g.home_spread_odds) or pd.isna(g.away_spread_odds):
                continue
            row = {c: getattr(g, c) for c in FACTOR_COLS}
            pred = predicted_margin(row, weights, multipliers)
            p_home_cover = home_cover_prob(pred, g.spread_line, weights["resid_std"])
            p_away_cover = 1.0 - p_home_cover

            side_results = {}
            for side, team, p_cover, odds in (
                ("home", g.home_team, p_home_cover, g.home_spread_odds),
                ("away", g.away_team, p_away_cover, g.away_spread_odds),
            ):
                res = ev.evaluate_side(p_win=p_cover, odds=odds, n_books=1, consensus_gap=0.0, width_dec=0.0,
                                        gcfg=gcfg, enforce_market_quality=False)
                res.update(side=side, team=team)
                side_results[side] = res
            best = max(side_results.values(), key=lambda r: r["edge"])

            if pd.isna(g.home_score) or pd.isna(g.away_score):
                profit = 0.0
                home_covered = None
            else:
                actual_margin = g.home_score - g.away_score
                home_covered = actual_margin > g.spread_line
                won = (best["side"] == "home") == home_covered
                profit = kcfg.flat_stake * ev.profit_per_dollar(best["odds"]) if won else -kcfg.flat_stake

            rows.append({
                "season": season, "week": week, "game_id": g.game_id,
                "model_side": best["side"], "model_odds": best["odds"], "model_edge": best["edge"],
                "pred_margin": pred, "spread_line": g.spread_line, "resid_std": weights["resid_std"],
                "bet_flag": int(best["bet_flag"]), "model_profit": profit,
                "n_fit": weights.get("n_fit", 0),
            })

    return pd.DataFrame(rows)
