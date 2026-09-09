"""
The core betting-math layer: probability <-> American odds conversions,
expected value, guardrails, and Kelly sizing. Every threshold used here
comes from config.GuardrailConfig / config.KellyConfig -- nothing is
hardcoded, so testing "what if EDGE_MIN_BASE were 0.03 instead of 0.02"
is a config change, not a code change.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import GuardrailConfig, KellyConfig, DEFAULT_GUARDRAILS, DEFAULT_KELLY


def prob_to_fair_ml(p: float) -> float:
    """Model probability -> the moneyline that would make it a exactly-fair bet."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    if p >= 0.5:
        return -100.0 * p / (1 - p)
    return 100.0 * (1 - p) / p


def american_to_implied_prob(odds: float) -> float:
    """Sportsbook price -> implied probability (includes the vig, i.e.
    home+away implied probs sum to > 1)."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def profit_per_dollar(odds: float) -> float:
    """Net profit on a winning $1 bet at the given American odds."""
    if odds > 0:
        return odds / 100.0
    return 100.0 / (-odds)


def expected_value(p_win: float, odds: float, stake: float = 1.0) -> float:
    """EV in dollars for `stake` risked at `odds`, given true win prob p_win."""
    win_profit = profit_per_dollar(odds) * stake
    return p_win * win_profit - (1 - p_win) * stake


def kelly_fraction(p_win: float, odds: float) -> float:
    """Full-Kelly fraction of bankroll (can be negative -- means no bet)."""
    b = profit_per_dollar(odds)
    if b <= 0:
        return 0.0
    q = 1 - p_win
    f = (b * p_win - q) / b
    return f


def kelly_stake(p_win: float, odds: float, bankroll: float, kcfg: KellyConfig = DEFAULT_KELLY) -> float:
    f = kelly_fraction(p_win, odds)
    if f <= 0:
        return 0.0
    fractional = f * kcfg.kelly_fraction
    capped = min(fractional, kcfg.stake_cap_pct)
    return round(capped * bankroll, 2)


@dataclass
class GuardrailResult:
    passes: bool
    required_edge: float
    reasons: list  # human-readable notes, populated even when passes=True


def required_edge_for_side(odds: float, p_win: float, gcfg: GuardrailConfig = DEFAULT_GUARDRAILS) -> tuple[float, list]:
    """The minimum EV this side needs to clear, after guardrail add-ons."""
    req = gcfg.edge_min_base
    notes = [f"base minimum edge {gcfg.edge_min_base:.2%}"]

    if odds <= gcfg.fav_floor:
        req += gcfg.fav_extra_edge
        notes.append(f"short favorite ({odds:.0f} <= {gcfg.fav_floor}): +{gcfg.fav_extra_edge:.2%}")
    if odds >= gcfg.dog_cap:
        req += gcfg.dog_extra_edge
        notes.append(f"long underdog ({odds:.0f} >= +{gcfg.dog_cap}): +{gcfg.dog_extra_edge:.2%}")
    if abs(p_win - 0.5) <= gcfg.coinflip_band:
        req += gcfg.coinflip_extra_edge
        notes.append(f"coin-flip probability ({p_win:.1%} within {gcfg.coinflip_band:.0%} of 50%): +{gcfg.coinflip_extra_edge:.2%}")
    return req, notes


def market_quality_check(n_books: int, consensus_gap: float, width_dec: float,
                          gcfg: GuardrailConfig = DEFAULT_GUARDRAILS,
                          enforce: bool = True) -> tuple[bool, list]:
    if not enforce:
        # Single-line sources (the free nflverse fallback) only ever report
        # one book, so min-books/consensus-gap/width checks are meaningless
        # against them -- they'd fail every game on a technicality rather
        # than a real market-quality problem. Skip them and say so, rather
        # than silently zeroing out every bet_flag.
        return True, ["market-quality checks skipped (single-line source, no per-book spread available)"]
    ok = True
    notes = []
    if n_books < gcfg.min_books:
        ok = False
        notes.append(f"only {n_books} book(s) quoting, need >= {gcfg.min_books}")
    if consensus_gap > gcfg.max_consensus_gap:
        ok = False
        notes.append(f"consensus gap {consensus_gap:.0f} pts > max {gcfg.max_consensus_gap:.0f}")
    if width_dec > gcfg.max_width_dec:
        ok = False
        notes.append(f"market width {width_dec:.3f} (decimal) > max {gcfg.max_width_dec:.2f}")
    return ok, notes


def evaluate_side(p_win: float, odds: float, n_books: int, consensus_gap: float,
                   width_dec: float, gcfg: GuardrailConfig = DEFAULT_GUARDRAILS,
                   enforce_market_quality: bool = True) -> dict:
    """Full guardrail evaluation for one side of one game. Returns a dict
    ready to drop straight into a card row."""
    edge = expected_value(p_win, odds, stake=1.0)  # EV per $1 = edge as a fraction
    fair_ml = prob_to_fair_ml(p_win)

    required_edge, edge_notes = required_edge_for_side(odds, p_win, gcfg)
    market_ok, market_notes = market_quality_check(n_books, consensus_gap, width_dec, gcfg,
                                                     enforce=enforce_market_quality)
    edge_ok = edge >= required_edge

    bet_flag = bool(edge_ok and market_ok)
    reasons = []
    if not edge_ok:
        reasons.append(f"edge {edge:.2%} below required {required_edge:.2%} ({'; '.join(edge_notes)})")
    if not market_ok:
        reasons.extend(market_notes)
    if bet_flag:
        reasons.append(f"edge {edge:.2%} clears required {required_edge:.2%}; market quality OK")

    return {
        "p_win": p_win, "odds": odds, "fair_ml": round(fair_ml),
        "edge": edge, "required_edge": required_edge,
        "n_books": n_books, "consensus_gap": consensus_gap, "width_dec": width_dec,
        "bet_flag": bet_flag, "reason": " | ".join(reasons),
    }
