"""
Command-line entry points for the weekly workflow (brief section 17).
Run `python -m nfl_edge.cli <command> --help` for options on any command.

Typical weekly use:
    python -m nfl_edge.cli refresh-data
    python -m nfl_edge.cli next-week --season 2026
    python -m nfl_edge.cli card --season 2026 --week 2
    ... (after games finish) ...
    python -m nfl_edge.cli grade --season 2026 --week 2
    python -m nfl_edge.cli report --season 2026
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import data, db, grading
from .backtest import walk_forward_backtest
from .card import build_card, format_display_table
from .config import DEFAULT_TRAINING_WINDOW, TrainingWindow
from .factors import walk_forward_multifactor_backtest, FACTOR_COLS
from .model import NFLWinProbModel
from .spread import walk_forward_spread_backtest

pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 200)


def _train_model(seasons=None, calibration_method="platt") -> NFLWinProbModel:
    window = TrainingWindow(seasons=tuple(seasons)) if seasons else DEFAULT_TRAINING_WINDOW
    m = NFLWinProbModel(training_window=window, calibration_method=calibration_method)
    m.fit()
    return m


def cmd_refresh_data(args):
    path = data.refresh_games_cache(force=True)
    print(f"Refreshed game/odds cache: {path}")


def cmd_next_week(args):
    games = data.load_games(seasons=[args.season])
    wk = data.next_unplayed_week(games, args.season)
    print(f"Next unplayed week in {args.season}: {wk}")


def cmd_card(args):
    m = _train_model(args.seasons, args.calibration)
    card = build_card(m, args.season, args.week)
    print(format_display_table(card).to_string(index=False))
    if not args.no_save:
        n = db.save_card(card, training_seasons=m.training_window.seasons,
                          odds_source="live_api" if args.week else "nflverse_line")
        print(f"\nSaved {n} rows to {db.DB_PATH}")


def cmd_grade(args):
    graded = grading.grade_week(args.season, args.week)
    if graded.empty:
        print("Nothing to grade yet (no final scores found for saved cards).")
        return
    print(graded.to_string(index=False))
    print(f"\nWeek total: {graded['profit'].sum():.2f} "
          f"({int((graded['pick_won']==1).sum())}-{int((graded['pick_won']==0).sum())})")


def cmd_backtest(args):
    window = TrainingWindow(seasons=tuple(args.seasons)) if args.seasons else DEFAULT_TRAINING_WINDOW
    bt = walk_forward_backtest(training_window=window, calibration_method=args.calibration)
    out_path = args.out or "backtest_results.csv"
    bt.to_csv(out_path, index=False)
    print(f"Backtested {len(bt)} games across seasons {window.seasons}. Saved detail to {out_path}\n")
    print("=== Headline: all picks vs. guardrail-flagged only ===")
    print(grading.summarize(bt))
    print("\n=== Five baselines (section 18/19) ===")
    print(grading.baseline_comparison(bt))
    print("\n=== Favorite vs. underdog ===")
    print(grading.favorite_underdog_split(bt))
    print("\n=== Home vs. away pick ===")
    print(grading.home_away_split(bt))
    print("\n=== By edge bucket ===")
    print(grading.edge_bucket_report(bt))
    print("\n=== By odds bucket ===")
    print(grading.odds_bucket_report(bt))


def _multipliers_from_args(args) -> dict:
    multipliers = {}
    if args.qb is not None:
        multipliers["qb_diff"] = args.qb
    if args.weather is not None:
        multipliers["weather_severity"] = args.weather
    if args.home is not None:
        multipliers["home_extra"] = args.home
    if args.rest is not None:
        multipliers["rest_diff"] = args.rest
    if args.div is not None:
        multipliers["div_game"] = args.div
    if args.epa_off is not None:
        multipliers["epa_off_diff"] = args.epa_off
    if args.epa_def is not None:
        multipliers["epa_def_diff"] = args.epa_def
    if args.sack_allowed is not None:
        multipliers["sack_allowed_diff"] = args.sack_allowed
    if args.sack_generated is not None:
        multipliers["sack_generated_diff"] = args.sack_generated
    if args.travel_distance is not None:
        multipliers["travel_distance_diff"] = args.travel_distance
    if args.travel_tz is not None:
        multipliers["travel_tz_diff"] = args.travel_tz
    return multipliers


def cmd_multifactor_backtest(args):
    """The rigorous version of the interactive tuner page's moneyline tab:
    base weights refit weekly using only prior window games, for the
    multipliers you pass in (default 1.0 = trust the data as fit,
    matching slider position 3 on the page)."""
    multipliers = _multipliers_from_args(args)

    window = TrainingWindow(seasons=tuple(args.seasons)) if args.seasons else DEFAULT_TRAINING_WINDOW
    bt = walk_forward_multifactor_backtest(training_window=window, multipliers=multipliers)
    print(f"Multipliers: {multipliers or '(all 1.0x, i.e. trust the data fit)'}")
    print(f"Walk-forward, {len(bt)} games, base weights refit weekly on prior games only.\n")
    print(grading.summarize(bt, profit_col="model_profit"))
    print()
    print(grading.favorite_underdog_split(bt))


def cmd_spread_backtest(args):
    """The rigorous version of the interactive tuner page's spread tab:
    the margin-of-victory regression is refit weekly using only prior
    window games, for the multipliers you pass in. Reports win% against
    the spread (52.4% needed to break even at standard -110 odds), not
    win% of the game outright."""
    multipliers = _multipliers_from_args(args)

    window = TrainingWindow(seasons=tuple(args.seasons)) if args.seasons else DEFAULT_TRAINING_WINDOW
    bt = walk_forward_spread_backtest(training_window=window, multipliers=multipliers)
    print(f"Multipliers: {multipliers or '(all 1.0x, i.e. trust the data fit)'}")
    print(f"Walk-forward, {len(bt)} games, margin model refit weekly on prior games only.\n")
    print(grading.summarize(bt, profit_col="model_profit"))
    print()
    home_picks = bt[bt["model_side"] == "home"]
    away_picks = bt[bt["model_side"] == "away"]
    print(f"Picked home side: {len(home_picks)} games, "
          f"{100 * (home_picks['model_profit'] > 0).mean() if len(home_picks) else float('nan'):.1f}% won")
    print(f"Picked away side: {len(away_picks)} games, "
          f"{100 * (away_picks['model_profit'] > 0).mean() if len(away_picks) else float('nan'):.1f}% won")


def cmd_report(args):
    grades = db.all_grades()
    grades = grades[grades["season"] == args.season] if args.season else grades
    if grades.empty:
        print("No live-graded games yet. Run `card` then `grade` for at least one played week first.")
        return
    grades = grades.rename(columns={"profit": "model_profit"})
    print(f"=== Live tracked results{' for ' + str(args.season) if args.season else ''} ===")
    print(grading.summarize(grades, profit_col="model_profit"))


def main(argv=None):
    p = argparse.ArgumentParser(prog="nfl_edge")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("refresh-data", help="Re-download the games/odds cache")
    s.set_defaults(func=cmd_refresh_data)

    s = sub.add_parser("next-week", help="Find the next unplayed week for a season")
    s.add_argument("--season", type=int, required=True)
    s.set_defaults(func=cmd_next_week)

    s = sub.add_parser("card", help="Build (and save) the card for one week")
    s.add_argument("--season", type=int, required=True)
    s.add_argument("--week", type=int, required=True)
    s.add_argument("--seasons", type=int, nargs="+", help="training window seasons, e.g. --seasons 2023 2024 2025")
    s.add_argument("--calibration", choices=["platt", "isotonic"], default="platt")
    s.add_argument("--no-save", action="store_true")
    s.set_defaults(func=cmd_card)

    s = sub.add_parser("grade", help="Grade a played week's saved card against final scores")
    s.add_argument("--season", type=int, required=True)
    s.add_argument("--week", type=int, required=True)
    s.set_defaults(func=cmd_grade)

    s = sub.add_parser("backtest", help="Walk-forward backtest over a training window")
    s.add_argument("--seasons", type=int, nargs="+")
    s.add_argument("--calibration", choices=["platt", "isotonic"], default="platt")
    s.add_argument("--out", type=str)
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("multifactor-backtest", help="Walk-forward backtest of the moneyline multi-factor model")
    s.add_argument("--seasons", type=int, nargs="+")
    s.add_argument("--qb", type=float, help="multiplier on the QB rating differential factor (1.0 = data-fit)")
    s.add_argument("--weather", type=float, help="multiplier on the weather severity factor (1.0 = data-fit)")
    s.add_argument("--home", type=float, help="multiplier on the extra home-field factor (1.0 = data-fit)")
    s.add_argument("--rest", type=float, help="multiplier on the rest-advantage factor (1.0 = data-fit)")
    s.add_argument("--div", type=float, help="multiplier on the divisional-game factor (1.0 = data-fit)")
    s.add_argument("--epa-off", type=float, help="multiplier on the offensive EPA/play differential factor (1.0 = data-fit)")
    s.add_argument("--epa-def", type=float, help="multiplier on the defensive EPA/play differential factor (1.0 = data-fit)")
    s.add_argument("--sack-allowed", type=float, help="multiplier on the sack-rate-allowed (protection) differential factor (1.0 = data-fit)")
    s.add_argument("--sack-generated", type=float, help="multiplier on the sack-rate-generated (pass rush) differential factor (1.0 = data-fit)")
    s.add_argument("--travel-distance", type=float, help="multiplier on the travel-distance differential factor (1.0 = data-fit)")
    s.add_argument("--travel-tz", type=float, help="multiplier on the travel time-zone-shift differential factor (1.0 = data-fit)")
    s.set_defaults(func=cmd_multifactor_backtest)

    s = sub.add_parser("spread-backtest", help="Walk-forward backtest of the spread (against-the-line) multi-factor model")
    s.add_argument("--seasons", type=int, nargs="+")
    s.add_argument("--qb", type=float, help="multiplier on the QB rating differential factor (1.0 = data-fit)")
    s.add_argument("--weather", type=float, help="multiplier on the weather severity factor (1.0 = data-fit)")
    s.add_argument("--home", type=float, help="multiplier on the extra home-field factor (1.0 = data-fit; note: mathematically 0 for this linear model, see docs/DESIGN.md)")
    s.add_argument("--rest", type=float, help="multiplier on the rest-advantage factor (1.0 = data-fit)")
    s.add_argument("--div", type=float, help="multiplier on the divisional-game factor (1.0 = data-fit)")
    s.add_argument("--epa-off", type=float, help="multiplier on the offensive EPA/play differential factor (1.0 = data-fit)")
    s.add_argument("--epa-def", type=float, help="multiplier on the defensive EPA/play differential factor (1.0 = data-fit)")
    s.add_argument("--sack-allowed", type=float, help="multiplier on the sack-rate-allowed (protection) differential factor (1.0 = data-fit)")
    s.add_argument("--sack-generated", type=float, help="multiplier on the sack-rate-generated (pass rush) differential factor (1.0 = data-fit)")
    s.add_argument("--travel-distance", type=float, help="multiplier on the travel-distance differential factor (1.0 = data-fit)")
    s.add_argument("--travel-tz", type=float, help="multiplier on the travel time-zone-shift differential factor (1.0 = data-fit)")
    s.set_defaults(func=cmd_spread_backtest)

    s = sub.add_parser("report", help="Summarize live-graded results saved in the DB")
    s.add_argument("--season", type=int)
    s.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
