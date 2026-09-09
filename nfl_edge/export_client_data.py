"""
Builds the JSON payload embedded in the interactive slider page
(dashboard-style, but live/tweakable). Two pieces:

1. `backtest_games`: every training-window game's standardized features,
   market odds, and actual outcome -- enough for the page to recompute
   picks/EV/guardrails/ROI entirely in JavaScript for ANY slider setting,
   instantly, with no server round-trip.
2. `next_week`: the same features for the next not-yet-played week, so
   the page can also show "here's what it would actually pick this week"
   under whatever weights you choose.

Base coefficients are fit ONCE, in-sample, on the whole training window
(fast, needed for a responsive page) -- NOT the per-week walk-forward
refit that `factors.walk_forward_multifactor_backtest` does. That's a
deliberate, disclosed shortcut for a fast exploration tool; the page
says so, and points back to the CLI for the honest walk-forward number
on any configuration that looks interesting.

Next-week weather is a *live forecast* (weather_forecast.py, Open-Meteo,
no API key) for outdoor/open-roof games, not a guess -- but it only
works where outbound network access to api.open-meteo.com is allowed.
Where it isn't (this project's own cloud scheduled refresh, as of this
writing), every game silently falls back to the same neutral treatment
indoor games get, and `next_week.n_forecast_used` / `n_forecast_attempted`
in the payload tell you how many games actually got a real forecast on
this run -- 0 of N means the fallback path was taken, not that the
weather was calm everywhere.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import data
from .config import DEFAULT_TRAINING_WINDOW, TrainingWindow
from .elo import EloConfig, EloModel
from .epa import EPAModel, refresh_team_game_epa
from .factors import EPA_WARMUP_SEASONS, FACTOR_COLS, PRESSURE_WARMUP_SEASONS, build_features, fit_base_weights, weather_severity
from .odds import get_odds_source
from .pressure import PressureModel, refresh_team_game_sacks
from .qb import QBRatingModel
from .spread import DEFAULT_RESID_STD, fit_margin_model
from .tracker import update_tracker
from .travel import travel_features
from .weather_forecast import fetch_forecast


def bootstrap_cis(history: pd.DataFrame, n_boot: int = 300, seed: int = 0) -> dict:
    from sklearn.linear_model import LogisticRegression
    d = history[history["home_win"].isin([0.0, 1.0])].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    boots = {c: [] for c in FACTOR_COLS}
    for _ in range(n_boot):
        idx = rng.integers(0, len(d), len(d))
        dd = d.iloc[idx]
        lr = LogisticRegression()
        lr.fit(dd[FACTOR_COLS], dd["home_win"])
        for j, c in enumerate(FACTOR_COLS):
            boots[c].append(lr.coef_[0][j])
    out = {}
    for c in FACTOR_COLS:
        arr = np.array(boots[c])
        out[c] = {"lo90": float(np.percentile(arr, 5)), "hi90": float(np.percentile(arr, 95))}
    return out


def bootstrap_margin_cis(history: pd.DataFrame, n_boot: int = 300, seed: int = 0) -> dict:
    """Same idea as bootstrap_cis, for the spread/margin regression's
    coefficients instead of the moneyline logistic's."""
    from sklearn.linear_model import LinearRegression
    needed = FACTOR_COLS + ["home_score", "away_score"]
    d = history.dropna(subset=needed).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    boots = {c: [] for c in FACTOR_COLS}
    y_all = (d["home_score"] - d["away_score"]).astype(float).values
    for _ in range(n_boot):
        idx = rng.integers(0, len(d), len(d))
        dd = d.iloc[idx]
        lr = LinearRegression()
        lr.fit(dd[FACTOR_COLS], y_all[idx])
        for j, c in enumerate(FACTOR_COLS):
            boots[c].append(lr.coef_[j])
    out = {}
    for c in FACTOR_COLS:
        arr = np.array(boots[c])
        out[c] = {"lo90": float(np.percentile(arr, 5)), "hi90": float(np.percentile(arr, 95))}
    return out


def build_payload(training_window: TrainingWindow = DEFAULT_TRAINING_WINDOW,
                   elo_cfg: EloConfig = EloConfig(), warmup_seasons: int = 8,
                   refresh_data: bool = False) -> dict:
    fb = build_features(training_window, elo_cfg, warmup_seasons, refresh_data)
    hist = fb.history

    weights = fit_base_weights(hist)
    cis = bootstrap_cis(hist)

    margin_weights = fit_margin_model(hist)
    margin_cis = bootstrap_margin_cis(hist)

    games = []
    for g in hist.itertuples(index=False):
        if pd.isna(g.home_moneyline) or pd.isna(g.away_moneyline):
            continue
        has_spread = pd.notna(getattr(g, "spread_line", None)) and \
            pd.notna(getattr(g, "home_spread_odds", None)) and pd.notna(getattr(g, "away_spread_odds", None))
        games.append({
            "season": int(g.season), "week": int(g.week),
            "home": g.home_team, "away": g.away_team,
            "elo_diff": round(g.elo_diff, 5), "qb_diff": round(g.qb_diff, 5),
            "weather_severity": round(g.weather_severity, 5), "home_extra": 1.0,
            "rest_diff": round(g.rest_diff, 5), "div_game": float(g.div_game),
            "epa_off_diff": round(g.epa_off_diff, 5), "epa_def_diff": round(g.epa_def_diff, 5),
            "sack_allowed_diff": round(g.sack_allowed_diff, 5), "sack_generated_diff": round(g.sack_generated_diff, 5),
            "travel_distance_diff": round(g.travel_distance_diff, 5), "travel_tz_diff": round(g.travel_tz_diff, 5),
            "home_ml": float(g.home_moneyline), "away_ml": float(g.away_moneyline),
            "home_win": None if g.home_win == 0.5 else int(g.home_win),
            "home_score": None if pd.isna(g.home_score) else float(g.home_score),
            "away_score": None if pd.isna(g.away_score) else float(g.away_score),
            "spread_line": float(g.spread_line) if has_spread else None,
            "home_spread_odds": float(g.home_spread_odds) if has_spread else None,
            "away_spread_odds": float(g.away_spread_odds) if has_spread else None,
        })

    # Next unplayed week, using ratings as of "now" (all completed games
    # the training window's end season + any later completed games in the
    # free dataset -- e.g. partial 2026 data if some weeks are done).
    latest_season = max(training_window.seasons)
    all_games = data.load_games(seasons=list(range(min(training_window.seasons) - warmup_seasons, latest_season + 2)),
                                 refresh=False)
    completed = data.completed_games(all_games)
    elo_full = EloModel(cfg=elo_cfg)
    elo_hist_full = elo_full.fit(completed)
    qb_full = QBRatingModel()
    qb_full.fit(completed, elo_hist_full)
    epa_full = EPAModel()
    epa_completed_full = completed[completed["season"] >= latest_season - EPA_WARMUP_SEASONS]
    team_game_epa_full = refresh_team_game_epa(sorted(epa_completed_full["season"].unique().tolist()))
    epa_full.fit(epa_completed_full, team_game_epa_full)
    pressure_full = PressureModel()
    pressure_completed_full = completed[completed["season"] >= latest_season - PRESSURE_WARMUP_SEASONS]
    team_game_sacks_full = refresh_team_game_sacks(sorted(pressure_completed_full["season"].unique().tolist()))
    pressure_full.fit(pressure_completed_full, team_game_sacks_full)

    next_season = None
    next_week = None
    for season in range(latest_season, latest_season + 2):
        season_games = data.load_games(seasons=[season], refresh=False)
        wk = data.next_unplayed_week(season_games, season)
        if wk is not None:
            next_season, next_week = season, wk
            break

    next_week_games = []
    n_forecast_used = 0
    n_forecast_attempted = 0
    if next_season is not None:
        elo_full.advance_to_season(next_season)
        wk_games = data.upcoming_games(data.load_games(seasons=[next_season], refresh=False), next_season, next_week)
        src = get_odds_source()
        market = src.for_week(wk_games)
        for g in wk_games.itertuples(index=False):
            gm = market[market["game_id"] == g.game_id]
            if gm.empty:
                continue
            home_elo = elo_full._get(g.home_team)
            away_elo = elo_full._get(g.away_team)
            elo_diff = (home_elo + elo_cfg.home_field_advantage - away_elo) / 400.0
            home_qb_r = qb_full.ratings.get(g.home_qb_name, home_elo) if pd.notna(getattr(g, "home_qb_name", None)) else home_elo
            away_qb_r = qb_full.ratings.get(g.away_qb_name, away_elo) if pd.notna(getattr(g, "away_qb_name", None)) else away_elo
            qb_diff = (home_qb_r - away_qb_r) / 400.0
            epa_off_diff, epa_def_diff = epa_full.current_diff_for(g.home_team, g.away_team)
            sack_allowed_diff, sack_generated_diff = pressure_full.current_diff_for(g.home_team, g.away_team)
            stadium_id = getattr(g, "stadium_id", None)
            travel_distance_diff, travel_tz_diff = travel_features(g.home_team, g.away_team, stadium_id)

            roof = getattr(g, "roof", None)
            forecast_used = False
            if roof in ("dome", "closed"):
                # Decided indoors -- no forecast needed, same as the
                # historical-data path.
                wx = 0.0
            else:
                n_forecast_attempted += 1
                gameday = getattr(g, "gameday", None)
                gametime = getattr(g, "gametime", None)
                date_str = gameday.strftime("%Y-%m-%d") if pd.notna(gameday) else None
                f_temp, f_wind = (None, None)
                if date_str and gametime:
                    f_temp, f_wind = fetch_forecast(stadium_id, date_str, str(gametime))
                if f_temp is not None and f_wind is not None:
                    wx = weather_severity(f_temp, f_wind, roof)
                    forecast_used = True
                    n_forecast_used += 1
                else:
                    # No live forecast available (network restricted in
                    # this environment, game too far out, unknown
                    # stadium, etc). Neutral fallback -- never a made-up
                    # number standing in as if it were real.
                    wx = 0.0

            home_rest = getattr(g, "home_rest", None)
            away_rest = getattr(g, "away_rest", None)
            rest_diff = ((home_rest if pd.notna(home_rest) else 7) - (away_rest if pd.notna(away_rest) else 7)) / 7.0
            div_game = float(getattr(g, "div_game", 0) or 0)

            home_row = gm[gm["side"] == "home"]
            away_row = gm[gm["side"] == "away"]
            if home_row.empty or away_row.empty:
                continue

            spread_line = getattr(g, "spread_line", None)
            home_spread_odds = getattr(g, "home_spread_odds", None)
            away_spread_odds = getattr(g, "away_spread_odds", None)
            has_spread = pd.notna(spread_line) and pd.notna(home_spread_odds) and pd.notna(away_spread_odds)

            next_week_games.append({
                "game_id": g.game_id,
                "season": int(next_season), "week": int(next_week),
                "home": g.home_team, "away": g.away_team,
                "elo_diff": round(elo_diff, 5), "qb_diff": round(qb_diff, 5),
                "weather_severity": round(wx, 5), "home_extra": 1.0,
                "rest_diff": round(rest_diff, 5), "div_game": div_game,
                "epa_off_diff": round(epa_off_diff, 5), "epa_def_diff": round(epa_def_diff, 5),
                "sack_allowed_diff": round(sack_allowed_diff, 5), "sack_generated_diff": round(sack_generated_diff, 5),
                "travel_distance_diff": round(travel_distance_diff, 5), "travel_tz_diff": round(travel_tz_diff, 5),
                "home_ml": float(home_row.iloc[0]["best_price"]), "away_ml": float(away_row.iloc[0]["best_price"]),
                "home_win": None,
                "weather_forecast_used": forecast_used,
                "spread_line": float(spread_line) if has_spread else None,
                "home_spread_odds": float(home_spread_odds) if has_spread else None,
                "away_spread_odds": float(away_spread_odds) if has_spread else None,
            })

    # Locks in predictions for any newly-upcoming games (default 1x
    # weights, never overwritten once locked) and grades any previously-
    # locked games that now have a final score. See tracker.py.
    tracked = update_tracker(next_week_games, weights, margin_weights,
                              margin_weights.get("resid_std", DEFAULT_RESID_STD), completed)

    return {
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "training_seasons": list(training_window.seasons),
        "n_games": len(games),
        "base_weights": {k: round(float(v), 5) for k, v in weights.items() if k != "n_fit"},
        "n_fit": weights.get("n_fit", 0),
        "confidence_intervals": cis,
        "margin_weights": {k: round(float(v), 5) for k, v in margin_weights.items() if k not in ("n_fit", "resid_std")},
        "margin_resid_std": round(float(margin_weights.get("resid_std", DEFAULT_RESID_STD)), 4),
        "margin_n_fit": margin_weights.get("n_fit", 0),
        "margin_confidence_intervals": margin_cis,
        "backtest_games": games,
        "next_week": {
            "season": next_season, "week": next_week, "games": next_week_games,
            "n_forecast_used": n_forecast_used, "n_forecast_attempted": n_forecast_attempted,
        },
        "tracked_weeks": tracked,
    }


def write_payload(path: str = "data/processed/client_data.json", **kwargs) -> str:
    payload = build_payload(**kwargs)
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


if __name__ == "__main__":
    p = write_payload()
    print(f"Wrote {p}")
