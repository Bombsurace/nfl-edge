"""
Central configuration: every tunable constant lives here, in one place,
instead of scattered through notebook cells. Change a number here and every
module (EV engine, guardrails, card builder, backtester) picks it up.
"""
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DB_PATH = PROJECT_ROOT / "data" / "nfl_edge.sqlite"

for _p in (DATA_RAW, DATA_PROCESSED, DB_PATH.parent):
    _p.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingWindow:
    """Which seasons feed the model, and how much each one counts.

    Kept as its own object (not a bare list of years) so a run can be
    reproduced later just by recording this config alongside the results.
    """
    seasons: tuple = (2023, 2024, 2025)
    # Per-season sample weight multiplier applied on top of the Elo engine's
    # own natural recency weighting. 1.0 = every season counts equally.
    # Set e.g. {2023: 0.7, 2024: 0.85, 2025: 1.0} to test recency weighting.
    season_weights: dict = field(default_factory=lambda: {2023: 1.0, 2024: 1.0, 2025: 1.0})

    def as_list(self):
        return list(self.seasons)


DEFAULT_TRAINING_WINDOW = TrainingWindow()


@dataclass
class GuardrailConfig:
    """Every threshold from the original brief, centralized and named."""
    edge_min_base: float = 0.02       # minimum EV required before considering a bet
    fav_floor: int = -250             # favorites shorter than this need extra edge
    fav_extra_edge: float = 0.02      # additional edge required beyond fav_floor
    dog_cap: int = 300                # dogs longer than this need extra edge
    dog_extra_edge: float = 0.02      # additional edge required beyond dog_cap
    coinflip_band: float = 0.05       # +/- around 50% probability
    coinflip_extra_edge: float = 0.01 # additional edge required inside the coin-flip band
    min_books: int = 3                # minimum sportsbooks quoting a price
    max_consensus_gap: float = 30.0   # max American-odds points from best price to median
    max_width_dec: float = 0.35       # max spread between best/worst price in decimal odds


DEFAULT_GUARDRAILS = GuardrailConfig()


@dataclass
class KellyConfig:
    kelly_fraction: float = 0.25
    stake_cap_pct: float = 0.02       # of bankroll
    flat_stake: float = 10.0          # used for model evaluation / grading


DEFAULT_KELLY = KellyConfig()

# Whitelisted sportsbooks (must match names as returned by the odds provider)
SPORTSBOOK_WHITELIST = [
    "DraftKings", "FanDuel", "BetMGM", "Caesars",
    "PointsBet", "Bet365", "WynnBet", "Barstool",
]

# nflverse's public games.csv (free, no key) carries a historical closing
# moneyline for every game back to the 1999 season, plus this week's
# already-posted opening lines once the schedule is released. It's the
# backbone for backtesting the model against *real* market prices, and
# a solid fallback "consensus-of-one" line before a paid multi-book feed
# is wired up.
NFLVERSE_GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

# Set this (env var ODDS_API_KEY, or edit here) to enable live multi-book
# odds pulls from https://the-odds-api.com/ (free tier: 500 req/month).
# Without a key, the odds module falls back to the nflverse single line.
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
