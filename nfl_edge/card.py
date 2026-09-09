"""
Builds the weekly betting card: every game, both sides evaluated, the
better-EV side selected as the pick, guardrails applied. This is section
4 + section 11 + section 20 of the brief in one function.
"""
from __future__ import annotations

import pandas as pd

from . import ev, odds as odds_mod
from .config import GuardrailConfig, DEFAULT_GUARDRAILS
from .model import NFLWinProbModel


def build_card(model: NFLWinProbModel, season: int, week: int,
                odds_source=None, gcfg: GuardrailConfig = DEFAULT_GUARDRAILS,
                refresh_data: bool = False, enforce_market_quality: bool | None = None) -> pd.DataFrame:
    preds = model.predict_week(season, week, refresh_data=refresh_data)
    src = odds_source or odds_mod.get_odds_source()
    market = src.for_week(preds)  # needs game_id/home_team/away_team columns, which preds has

    if market.empty:
        raise RuntimeError(
            f"No odds available for season={season} week={week}. "
            "If using the live API, check ODDS_API_KEY and that this week's "
            "lines are posted; otherwise the nflverse fallback may not have "
            "a line for this game yet."
        )

    # Auto-detect a single-line source (max n_books == 1) and relax the
    # market-quality guardrails accordingly, unless the caller overrides.
    if enforce_market_quality is None:
        enforce_market_quality = bool(market["n_books"].max() > 1)

    rows = []
    for g in preds.itertuples(index=False):
        p_home = g.p_final
        p_away = 1 - p_home
        game_market = market[market["game_id"] == g.game_id]
        if game_market.empty:
            continue

        side_results = {}
        for side, p_win in (("home", p_home), ("away", p_away)):
            m = game_market[game_market["side"] == side]
            if m.empty:
                continue
            m = m.iloc[0]
            consensus_gap = abs(m["best_price"] - m["consensus_price"])
            res = ev.evaluate_side(
                p_win=p_win, odds=m["best_price"], n_books=int(m["n_books"]),
                consensus_gap=consensus_gap, width_dec=m["width_dec"], gcfg=gcfg,
                enforce_market_quality=enforce_market_quality,
            )
            res["side"] = side
            res["team"] = m["team"]
            side_results[side] = res

        if not side_results:
            continue

        # Pick whichever side has the higher EV -- section 4/6 of the brief.
        # This is NOT "whoever the model thinks wins"; it's whoever the
        # market is mispricing more in our favor.
        best_side = max(side_results.values(), key=lambda r: r["edge"])

        row = {
            "season": g.season, "week": g.week, "game_id": g.game_id,
            "game_type": getattr(g, "game_type", None),
            "gameday": str(getattr(g, "gameday", "")),
            "home_team": g.home_team, "away_team": g.away_team,
            "div_game": int(getattr(g, "div_game", 0) or 0),
            "p_home": g.p_home, "p_home_cal": g.p_home_cal, "p_final": g.p_final,
            "pick_side": best_side["side"], "pick_team": best_side["team"],
            "pick_ml": best_side["odds"], "pick_prob": best_side["p_win"],
            "fair_ml": best_side["fair_ml"], "edge": best_side["edge"],
            "required_edge": best_side["required_edge"],
            "n_books": best_side["n_books"], "consensus_gap": best_side["consensus_gap"],
            "width_dec": best_side["width_dec"], "bet_flag": int(best_side["bet_flag"]),
            "reason": best_side["reason"],
        }
        rows.append(row)

    card = pd.DataFrame(rows)
    if not card.empty:
        card = card.sort_values(["bet_flag", "edge"], ascending=[False, False]).reset_index(drop=True)
    return card


def format_display_table(card: pd.DataFrame) -> pd.DataFrame:
    """Human-readable version matching section 20's example table."""
    if card.empty:
        return card
    disp = pd.DataFrame({
        "Game": card["away_team"] + " @ " + card["home_team"],
        "Model Pick": card["pick_team"],
        "Model Prob.": (card["pick_prob"] * 100).round(1).astype(str) + "%",
        "Best ML": card["pick_ml"].map(lambda x: f"{x:+.0f}"),
        "Fair ML": card["fair_ml"].map(lambda x: f"{x:+.0f}"),
        "EV": (card["edge"] * 100).round(1).astype(str) + "%",
        "Bet?": card["bet_flag"].map({1: "YES", 0: "NO"}),
        "Reason": card["reason"],
    })
    return disp
