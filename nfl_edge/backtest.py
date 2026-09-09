"""
Walk-forward backtest: the honest answer to "does this model have a real
edge," as opposed to model.reliability_report()'s in-sample check.

For every week inside the training window, the calibration curve used to
predict that week is fit ONLY on window games completed strictly before
that week (Elo ratings are already no-lookahead by construction -- see
elo.py). Nothing about a game's own outcome, or any later game, leaks
into its own prediction. Early weeks with too little prior window data
to calibrate fall back to the raw Elo probability.

The output is one row per game with: the model's pick, its EV/edge and
bet_flag, five baseline picks (section 18/19 of the brief), and the
actual result/profit for each -- so grading.py can slice performance
every way the brief asks for without re-deriving picks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data, ev
from .config import GuardrailConfig, KellyConfig, DEFAULT_GUARDRAILS, DEFAULT_KELLY, TrainingWindow, DEFAULT_TRAINING_WINDOW
from .elo import EloConfig, EloModel
from .model import PlattCalibrator
from sklearn.isotonic import IsotonicRegression

MIN_GAMES_TO_CALIBRATE = 100  # below this, use raw elo prob (no calibration yet)


def _fit_calibrator(cal_df: pd.DataFrame, method: str):
    cal_df = cal_df[cal_df["home_win"].isin([0.0, 1.0])]
    if len(cal_df) < MIN_GAMES_TO_CALIBRATE:
        return None
    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
    else:
        cal = PlattCalibrator()
    cal.fit(cal_df["elo_prob_home"], cal_df["home_win"])
    return cal


def _pick_row(side_results: dict, key_fn) -> dict:
    return max(side_results.values(), key=key_fn)


def walk_forward_backtest(
    training_window: TrainingWindow = DEFAULT_TRAINING_WINDOW,
    elo_cfg: EloConfig = EloConfig(),
    calibration_method: str = "platt",
    warmup_seasons: int = 8,
    gcfg: GuardrailConfig = DEFAULT_GUARDRAILS,
    kcfg: KellyConfig = DEFAULT_KELLY,
    refresh_data: bool = False,
) -> pd.DataFrame:
    seasons = training_window.as_list()
    start_season = min(seasons) - warmup_seasons
    end_season = max(seasons)
    all_games = data.load_games(seasons=list(range(start_season, end_season + 1)), refresh=refresh_data)
    completed = data.completed_games(all_games)

    elo = EloModel(cfg=elo_cfg)
    history = elo.fit(completed)  # no-lookahead elo_prob_home for every game, in order

    window_hist = history[history["season"].isin(seasons)].copy().sort_values(["season", "week"]).reset_index(drop=True)
    # bring in odds + context columns needed for EV/guardrails
    ctx_cols = ["game_id", "home_moneyline", "away_moneyline", "div_game", "game_type"]
    window_hist = window_hist.merge(completed[ctx_cols], on="game_id", how="left")

    rows = []
    for (season, week), wk_games in window_hist.groupby(["season", "week"], sort=True):
        prior = window_hist[(window_hist["season"] < season) | ((window_hist["season"] == season) & (window_hist["week"] < week))]
        calibrator = _fit_calibrator(prior, calibration_method)

        for g in wk_games.itertuples(index=False):
            if pd.isna(g.home_moneyline) or pd.isna(g.away_moneyline):
                continue  # no market line for this historical game, skip

            p_home_raw = g.elo_prob_home
            p_home = float(calibrator.predict([p_home_raw])[0]) if calibrator is not None else p_home_raw
            p_away = 1 - p_home

            side_results = {}
            for side, team, p_win, odds in (
                ("home", g.home_team, p_home, g.home_moneyline),
                ("away", g.away_team, p_away, g.away_moneyline),
            ):
                res = ev.evaluate_side(
                    p_win=p_win, odds=odds, n_books=1, consensus_gap=0.0, width_dec=0.0,
                    gcfg=gcfg, enforce_market_quality=False,  # single historical line, see odds.py
                )
                res.update(side=side, team=team)
                side_results[side] = res

            best_ev = _pick_row(side_results, lambda r: r["edge"])
            best_prob = _pick_row(side_results, lambda r: r["p_win"])
            fav = min(side_results.values(), key=lambda r: r["odds"])  # most negative = biggest favorite
            home_pick = side_results["home"]

            actual_home_win = g.home_win  # 1.0 / 0.0 / 0.5 (tie)

            def grade(pick):
                if actual_home_win == 0.5:
                    return 0.0  # push on a tie; no stake gained or lost
                won = (pick["side"] == "home") == (actual_home_win == 1.0)
                stake = kcfg.flat_stake
                return (stake * ev.profit_per_dollar(pick["odds"])) if won else -stake

            rows.append({
                "season": season, "week": week, "game_id": g.game_id,
                "game_type": g.game_type, "home_team": g.home_team, "away_team": g.away_team,
                "div_game": int(g.div_game) if not pd.isna(g.div_game) else 0,
                "actual_home_win": actual_home_win,
                "calibrated": calibrator is not None,
                # model (highest EV + guardrails, i.e. "the" pick)
                "model_side": best_ev["side"], "model_team": best_ev["team"],
                "model_odds": best_ev["odds"], "model_prob": best_ev["p_win"],
                "model_edge": best_ev["edge"], "bet_flag": int(best_ev["bet_flag"]),
                "model_reason": best_ev["reason"],
                "model_profit": grade(best_ev),
                "model_is_favorite": best_ev["odds"] < 0,
                "model_is_home": best_ev["side"] == "home",
                # baselines
                "bl_home_profit": grade(home_pick),
                "bl_favorite_profit": grade(fav),
                "bl_highprob_profit": grade(best_prob),
                "bl_highev_noguard_profit": grade(best_ev),  # same pick as model, but graded regardless of bet_flag
                "bl_highev_guard_profit": grade(best_ev) if best_ev["bet_flag"] else np.nan,  # None = no bet made
            })

    return pd.DataFrame(rows)
