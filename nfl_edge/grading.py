"""
Performance analysis: turns a graded set of picks (from backtest.py, or
from the live SQLite card/grade tables) into the win%/ROI/units cuts the
brief asks for in sections 12, 13, and 18 -- and answers section 19's
core instruction not to judge the model on any single week by always
reporting sample size (n) next to every number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data, ev
from .config import DEFAULT_KELLY, KellyConfig
from .db import connect, init_db, latest_cards


def _stats(profits: pd.Series) -> dict:
    profits = profits.dropna()
    n = len(profits)
    if n == 0:
        return {"n": 0, "win_pct": np.nan, "units": np.nan, "roi_pct": np.nan, "profit": np.nan}
    wins = int((profits > 0).sum())
    total_staked = n * DEFAULT_KELLY.flat_stake
    total_profit = profits.sum()
    return {
        "n": n,
        "win_pct": round(100 * wins / n, 1),
        "units": round(total_profit / DEFAULT_KELLY.flat_stake, 2),
        "roi_pct": round(100 * total_profit / total_staked, 2),
        "profit": round(total_profit, 2),
    }


def summarize(df: pd.DataFrame, profit_col: str = "model_profit", flag_col: str = "bet_flag") -> pd.DataFrame:
    """The headline table: all picks vs. flagged-only, with n/win%/units/ROI."""
    rows = {
        "all_picks": _stats(df[profit_col]),
        "flagged_only": _stats(df.loc[df[flag_col] == 1, profit_col]),
    }
    return pd.DataFrame(rows).T.reset_index().rename(columns={"index": "segment"})


def cut_by(df: pd.DataFrame, group_col: str, profit_col: str = "model_profit",
           flag_col: str = "bet_flag", flagged_only: bool = False) -> pd.DataFrame:
    """Performance broken out by an arbitrary column (home/away, div game,
    game_type, etc.). Set flagged_only=True to restrict to bet_flag==1
    before grouping."""
    sub = df[df[flag_col] == 1] if flagged_only else df
    out = sub.groupby(group_col)[profit_col].apply(_stats).apply(pd.Series).reset_index()
    return out


def edge_bucket_report(df: pd.DataFrame, profit_col: str = "model_profit",
                        edge_col: str = "model_edge", bins=(-1, 0, 0.02, 0.05, 0.10, 0.20, 1)) -> pd.DataFrame:
    labels = [f"{bins[i]:.0%} to {bins[i+1]:.0%}" for i in range(len(bins) - 1)]
    b = pd.cut(df[edge_col], bins=bins, labels=labels)
    out = df.groupby(b, observed=True)[profit_col].apply(_stats).apply(pd.Series).reset_index()
    return out.rename(columns={edge_col: "edge_bucket"})


def odds_bucket_report(df: pd.DataFrame, profit_col: str = "model_profit",
                        odds_col: str = "model_odds",
                        bins=(-100000, -300, -200, -150, -110, 110, 150, 200, 300, 100000)) -> pd.DataFrame:
    labels = ["<= -300", "-300 to -200", "-200 to -150", "-150 to -110",
              "-110 to +110", "+110 to +150", "+150 to +200", "+200 to +300", ">= +300"]
    b = pd.cut(df[odds_col], bins=bins, labels=labels)
    out = df.groupby(b, observed=True)[profit_col].apply(_stats).apply(pd.Series).reset_index()
    return out.rename(columns={odds_col: "odds_bucket"})


def favorite_underdog_split(df: pd.DataFrame, profit_col: str = "model_profit",
                             odds_col: str = "model_odds") -> pd.DataFrame:
    tag = np.where(df[odds_col] < 0, "favorite", "underdog")
    return df.assign(_tag=tag).groupby("_tag")[profit_col].apply(_stats).apply(pd.Series).reset_index().rename(columns={"_tag": "type"})


def home_away_split(df: pd.DataFrame, profit_col: str = "model_profit",
                     side_col: str = "model_is_home") -> pd.DataFrame:
    tag = np.where(df[side_col], "home_pick", "away_pick")
    return df.assign(_tag=tag).groupby("_tag")[profit_col].apply(_stats).apply(pd.Series).reset_index().rename(columns={"_tag": "type"})


def baseline_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Section 18's five baselines, side by side. `bl_highev_guard_profit`
    is NaN for ungraded (unflagged) games and _stats() drops NaNs, so its
    n naturally reflects only the flagged subset."""
    cols = {
        "1_always_home": "bl_home_profit",
        "2_always_favorite": "bl_favorite_profit",
        "3_highest_model_prob": "bl_highprob_profit",
        "4_highest_ev_no_guardrails": "bl_highev_noguard_profit",
        "5_highest_ev_plus_guardrails": "bl_highev_guard_profit",
    }
    rows = {label: _stats(df[col]) for label, col in cols.items()}
    return pd.DataFrame(rows).T.reset_index().rename(columns={"index": "baseline"})


# ---------------------------------------------------------------------
# Live weekly grading (against the SQLite card store, not the backtest)
# ---------------------------------------------------------------------

def grade_week(season: int, week: int, kcfg: KellyConfig = DEFAULT_KELLY) -> pd.DataFrame:
    """Grade the latest card snapshot for each game in a real (played)
    week using a flat stake, REGARDLESS of bet_flag -- section 17, step
    11: this is what lets you evaluate the model's picking ability
    independent of which picks the guardrails would have allowed."""
    init_db()
    cards = latest_cards(season=season, week=week)
    if cards.empty:
        raise ValueError(f"No saved cards found for season={season} week={week}. Run build_card + db.save_card first.")

    games = data.load_games(seasons=[season], refresh=False)
    results = data.completed_games(games)[["game_id", "home_score", "away_score"]]
    merged = cards.merge(results, on="game_id", how="left")

    graded_rows = []
    for r in merged.itertuples(index=False):
        if pd.isna(r.home_score) or pd.isna(r.away_score):
            continue  # not yet final
        home_won = r.home_score > r.away_score
        tie = r.home_score == r.away_score
        pick_won = None if tie else ((r.pick_side == "home") == home_won)
        stake = kcfg.flat_stake
        if tie:
            profit = 0.0
        elif pick_won:
            profit = stake * ev.profit_per_dollar(r.pick_ml)
        else:
            profit = -stake
        graded_rows.append({
            "card_id": r.id, "season": r.season, "week": r.week, "game_id": r.game_id,
            "home_score": r.home_score, "away_score": r.away_score,
            "pick_won": None if tie else int(pick_won), "stake": stake, "profit": profit,
        })

    if not graded_rows:
        return pd.DataFrame()

    gdf = pd.DataFrame(graded_rows)
    from datetime import datetime, timezone
    gdf["graded_at"] = datetime.now(timezone.utc).isoformat()
    with connect() as conn:
        gdf.to_sql("grades", conn, if_exists="append", index=False)
    return gdf
