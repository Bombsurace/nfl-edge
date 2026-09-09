"""
A running, no-lookahead record of the tuner page's own predictions: for
every game, lock in the moneyline pick and the spread pick BEFORE it's
played, then once nflverse's data shows a final score, grade that locked
pick against what actually happened. This is what answers "how has the
model actually done," as opposed to `factors.py`'s walk-forward backtest
(which re-fits and re-evaluates historical seasons on demand) or the
tuner page's live sliders (which recompute instantly for hypothetical
weight settings you haven't necessarily bet on).

Two properties make this a fair record rather than a hindsight report:

1. A game's prediction is written once, the first time it appears as an
   upcoming game, and is never overwritten afterward -- not even by a
   later run with a model that has since seen more data. Re-running the
   export mid-week (before that game kicks off) is fine and just uses
   the freshest available ratings; once a game has been locked, later
   runs leave it alone and only ever fill in its actual result.
2. Predictions here always use the DEFAULT (1x every factor) weights --
   "trust the data as fit," the same starting point the page itself opens
   to. That's a deliberate choice: the page's sliders let you explore
   hypothetical weightings, but a track record needs one fixed,
   reproducible configuration to be meaningful across weeks. Dragging a
   slider on the live page does not change what's recorded here.

Each week also gets a confidence ranking, for filling out a real
confidence-pool or "pick 5" style office pool: moneyline picks are
ranked by the model's own win probability for its picked side, and
spread picks are ranked separately by cover probability (the two often
don't agree on which game is "safest"), most lopsided call getting the
week's game count and the closest-to-a-coin-flip call getting 1. Same
fairness logic as the picks themselves -- the ranking is assigned once,
the moment a week's games are locked, and never reshuffled afterward,
so it reflects what the model would have told you before you filled out
a pool sheet, not after. It's computed only once a week's full game
slate is known in a single run (the normal case, since a week's games
all appear together); a game added to a week's slate after that
week's ranking has already been assigned is left unranked (`null`)
rather than bumping everyone else's numbers.

Storage is a flat JSON file (`data/processed/tracked_predictions.json`),
NOT regenerated from scratch like `client_data.json` -- it's the one file
in this project that holds information that can never be recreated after
the fact (what the model actually said before kickoff), so it's meant to
be kept, not treated as a cache.

Scope, stated plainly: this starts tracking from whenever this feature
first ran, going forward. It does NOT retroactively backfill the
2023-2025 training window (that's what the "Full backtest" tab already
covers, under whatever slider settings you choose) -- backfilling would
require deciding what the model "would have said" at each past moment
using a feature set that didn't exist yet, which is a different, murkier
question than this file is trying to answer.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config
from .factors import FACTOR_COLS, apply_weights
from .spread import home_cover_prob, predicted_margin

TRACKED_PATH = config.DATA_PROCESSED / "tracked_predictions.json"


def _load() -> dict:
    if TRACKED_PATH.exists():
        with open(TRACKED_PATH) as f:
            return json.load(f)
    return {"weeks": {}}


def _save(store: dict) -> None:
    with open(TRACKED_PATH, "w") as f:
        json.dump(store, f, indent=1)


def _pick_moneyline(game: dict, weights: dict) -> tuple[str, float]:
    z = apply_weights(game, weights)  # multipliers=None -> every factor at 1x
    p_home = 1.0 / (1.0 + np.exp(-z))
    return ("home" if p_home >= 0.5 else "away"), float(p_home if p_home >= 0.5 else 1 - p_home)


def _pick_spread(game: dict, margin_weights: dict, resid_std: float) -> tuple[str, float, float] | None:
    if game.get("spread_line") is None:
        return None
    pred = predicted_margin(game, margin_weights, feature_cols=FACTOR_COLS)
    p_home_cover = home_cover_prob(pred, game["spread_line"], resid_std)
    side = "home" if p_home_cover >= 0.5 else "away"
    return side, float(pred), float(p_home_cover if p_home_cover >= 0.5 else 1 - p_home_cover)


def _lock_new_games(store: dict, next_week_games: list[dict], weights: dict,
                     margin_weights: dict, margin_resid_std: float) -> None:
    """Add any game not already in the store, under its (season, week).
    Never touches a game_id that's already present."""
    for g in next_week_games:
        game_id = g.get("game_id")
        if not game_id:
            continue
        wk_key = f"{g['season']}-{g['week']}"
        wk = store["weeks"].setdefault(wk_key, {"season": g["season"], "week": g["week"], "games": {}})
        if game_id in wk["games"]:
            continue  # already locked -- never overwrite a prediction

        ml_pick, ml_prob = _pick_moneyline(g, weights)
        spread = _pick_spread(g, margin_weights, margin_resid_std)
        wk["games"][game_id] = {
            "home": g["home"], "away": g["away"],
            "locked_at": pd.Timestamp.now("UTC").isoformat(),
            "home_ml": g.get("home_ml"), "away_ml": g.get("away_ml"),
            "ml_pick": ml_pick, "ml_pick_prob": round(ml_prob, 4),
            "confidence": None,
            "spread_line": g.get("spread_line"),
            "home_spread_odds": g.get("home_spread_odds"), "away_spread_odds": g.get("away_spread_odds"),
            "spread_pick": spread[0] if spread else None,
            "spread_pred_margin": round(spread[1], 2) if spread else None,
            "spread_pick_prob": round(spread[2], 4) if spread else None,
            "spread_confidence": None,
            "home_score": None, "away_score": None,
            "ml_actual": None, "ml_correct": None,
            "spread_actual": None, "spread_correct": None,
            "graded_at": None,
        }


def _assign_confidence(wk: dict) -> None:
    """Rank a week's moneyline picks (by `ml_pick_prob`) and spread picks
    (by `spread_pick_prob`) independently, most lopsided call getting the
    week's game count down to 1 for the closest-to-a-coin-flip call.
    Called for every week on every run (cheap, and safe to call on a week
    that's fully graded) but guarded by `confidence_locked` so it only
    ever actually ranks a week once -- a later run never reshuffles picks
    you may have already copied onto a pool sheet. This also means a week
    locked before this ranking existed gets ranked retroactively the next
    time the export runs, using whichever games were already on record --
    exactly what you want since nothing about the rank depends on when it
    was computed, only on the (already-locked, already-fixed) pick
    probabilities. A game with no spread line simply doesn't get a
    `spread_confidence`."""
    if wk.get("confidence_locked"):
        return
    items = list(wk["games"].items())  # [(game_id, game_dict), ...]

    ml_ranked = sorted(items, key=lambda kv: (-kv[1]["ml_pick_prob"], kv[0]))
    n_ml = len(ml_ranked)
    for i, (_, g) in enumerate(ml_ranked):
        g["confidence"] = n_ml - i

    spread_items = [kv for kv in items if kv[1].get("spread_pick_prob") is not None]
    spread_ranked = sorted(spread_items, key=lambda kv: (-kv[1]["spread_pick_prob"], kv[0]))
    n_sp = len(spread_ranked)
    for i, (_, g) in enumerate(spread_ranked):
        g["spread_confidence"] = n_sp - i

    wk["confidence_locked"] = True


def _grade_pending(store: dict, completed: pd.DataFrame) -> None:
    """Fill in the actual result for any locked game that now has a final
    score in `completed`, without touching its already-locked prediction."""
    by_id = {r.game_id: r for r in completed.itertuples(index=False)} if not completed.empty else {}
    for wk in store["weeks"].values():
        for game_id, g in wk["games"].items():
            if g["graded_at"] is not None:
                continue
            row = by_id.get(game_id)
            if row is None or pd.isna(row.home_score) or pd.isna(row.away_score):
                continue  # not played yet (or not in this data pull)

            home_score, away_score = float(row.home_score), float(row.away_score)
            g["home_score"], g["away_score"] = home_score, away_score

            if home_score == away_score:
                g["ml_actual"] = "tie"
                g["ml_correct"] = None
            else:
                g["ml_actual"] = "home" if home_score > away_score else "away"
                g["ml_correct"] = (g["ml_pick"] == g["ml_actual"])

            if g["spread_pick"] is not None and g["spread_line"] is not None:
                actual_margin = home_score - away_score
                if actual_margin == g["spread_line"]:
                    g["spread_actual"] = "push"
                    g["spread_correct"] = None
                else:
                    g["spread_actual"] = "home" if actual_margin > g["spread_line"] else "away"
                    g["spread_correct"] = (g["spread_pick"] == g["spread_actual"])

            g["graded_at"] = pd.Timestamp.now("UTC").isoformat()


def _summarize(store: dict) -> dict:
    weeks_out = []
    ml_w = ml_l = spread_w = spread_l = spread_p = 0
    for wk_key in sorted(store["weeks"].keys(), key=lambda k: tuple(map(int, k.split("-")))):
        wk = store["weeks"][wk_key]
        games_out = []
        for game_id, g in sorted(wk["games"].items(), key=lambda kv: kv[0]):
            games_out.append({"game_id": game_id, **g})
            if g["ml_correct"] is True:
                ml_w += 1
            elif g["ml_correct"] is False:
                ml_l += 1
            if g["spread_correct"] is True:
                spread_w += 1
            elif g["spread_correct"] is False:
                spread_l += 1
            elif g["spread_actual"] == "push":
                spread_p += 1
        weeks_out.append({"season": wk["season"], "week": wk["week"], "games": games_out})

    return {
        "weeks": weeks_out,
        "record": {
            "ml_wins": ml_w, "ml_losses": ml_l,
            "spread_wins": spread_w, "spread_losses": spread_l, "spread_pushes": spread_p,
        },
    }


def update_tracker(next_week_games: list[dict], weights: dict, margin_weights: dict,
                    margin_resid_std: float, completed: pd.DataFrame) -> dict:
    """Called once per `export_client_data` run. Locks in any new
    upcoming games, ranks any not-yet-ranked week's confidence order,
    grades any previously-locked games that now have a final score,
    persists the result, and returns a summary dict ready to embed in the
    exported payload."""
    store = _load()
    _lock_new_games(store, next_week_games, weights, margin_weights, margin_resid_std)
    for wk in store["weeks"].values():
        _assign_confidence(wk)
    _grade_pending(store, completed)
    _save(store)
    return _summarize(store)
