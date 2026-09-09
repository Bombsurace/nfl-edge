"""
Travel distance and time-zone shift for a game -- how far the away team
had to travel to reach the stadium, and how many hours their body clock
is offset from home, relative to the home team's own (usually zero)
figures for the same game.

Unlike epa.py/pressure.py, this needs no historical fitting or
walk-forward state at all. A team's home city and a stadium's location
are both fixed facts, known in advance for every game including future
ones -- so there's no lookahead concern here, same category of
"already known before kickoff" input as `div_game` or `home_extra`.
Every value is a pure function of (home_team, away_team, stadium_id).

Two factors, both signed the project's usual positive-means-home-edge
way, though (like `weather_severity`) the actual DIRECTION of the
effect on home_win is deliberately not assumed here -- that's for the
logistic/linear fit to discover, not something to bake in ahead of time:

- travel_distance_diff: away team's great-circle distance from their own
  home stadium to this game's stadium, minus the home team's own distance
  (normally 0, since they're playing in their own city -- nonzero only
  for a true neutral-site game like London or Mexico City, where BOTH
  teams travel and this correctly nets out to whichever team travelled
  farther). In thousands of miles.
- travel_tz_diff: same idea for time zones -- the away team's body-clock
  shift (this stadium's UTC offset minus their home stadium's) minus the
  home team's own shift. In hours / 3 (a "roughly one zone" unit, chosen
  only to keep this column on a similar scale to the project's other
  factors -- the fit's own coefficient absorbs whatever the real-world
  per-hour effect turns out to be).

Known limitations, stated plainly:

1. Distances are straight-line (haversine) between stadium coordinates,
   not actual flight-path or drive miles -- a reasonable travel-burden
   proxy, not a travel agent's mileage.
2. Time zones here are fixed STANDARD-time UTC offsets with no
   daylight-saving calendar. Since the U.S. mainland's DST-observing
   zones all shift together, this cancels out for most games -- Arizona
   is the one real exception (it doesn't observe DST, so an ARI game
   against a DST-observing opponent is off by an hour for about the
   first two months of every season). Treated the same way as this
   project's other coarse-but-disclosed approximations (weather severity
   ignoring exact kickoff hour, EPA not being opponent-adjusted): a real
   signal, not a finished, survey-grade stat.
3. A team's "home stadium" is whichever stadium they've hosted the most
   games at across all available seasons -- correct for every team as of
   this writing, and self-corrects if a team gets a new stadium and plays
   enough seasons there, but a mid-season stadium change (a temporary
   home during a rebuild) could misattribute a handful of games.
"""
from __future__ import annotations

import math

from .stadiums import STADIUM_COORDS

EARTH_RADIUS_MILES = 3958.8

# team -> the stadium_id they've hosted the most home games at (2015-2025
# nflverse data). Only the 32 current franchises -- not meant to cover
# relocated/defunct team codes (OAK, SD, STL) from older seasons.
TEAM_HOME_STADIUM: dict[str, str] = {
    "ARI": "PHO00", "ATL": "ATL97", "BAL": "BAL00", "BUF": "BUF00",
    "CAR": "CAR00", "CHI": "CHI98", "CIN": "CIN00", "CLE": "CLE00",
    "DAL": "DAL00", "DEN": "DEN00", "DET": "DET00", "GB": "GNB00",
    "HOU": "HOU00", "IND": "IND00", "JAX": "JAX00", "KC": "KAN00",
    "LA": "LAX01", "LAC": "LAX01", "LV": "VEG00", "MIA": "MIA00",
    "MIN": "MIN01", "NE": "BOS00", "NO": "NOR00", "NYG": "NYC01",
    "NYJ": "NYC01", "PHI": "PHI00", "PIT": "PIT00", "SEA": "SEA00",
    "SF": "SFO01", "TB": "TAM00", "TEN": "NAS00", "WAS": "WAS00",
}

# stadium_id -> standard-time UTC offset in hours (no DST calendar --
# see module docstring, limitation 2).
STADIUM_TZ_OFFSET: dict[str, float] = {
    "PHO00": -7, "ATL97": -5, "BAL00": -5, "BUF00": -5, "CAR00": -5,
    "CHI98": -6, "CIN00": -5, "CLE00": -5, "DAL00": -6, "DEN00": -7,
    "DET00": -5, "GNB00": -6, "HOU00": -6, "IND00": -5, "JAX00": -5,
    "KAN00": -6, "VEG00": -8, "LAX01": -8, "MIA00": -5, "MIN01": -6,
    "BOS00": -5, "NOR00": -6, "NYC01": -5, "PHI00": -5, "PIT00": -5,
    "SEA00": -8, "SFO01": -8, "TAM00": -5, "NAS00": -6,
    "WAS00": -5, "WAS01": -5,
    # International / occasional neutral sites
    "LON00": 0, "LON01": 0, "LON02": 0,
    "FRA00": 1, "GER00": 1, "MUN01": 1, "PAR00": 1, "MAD01": 1,
    "MEX00": -6, "MEL00": 10, "RIO00": -3, "SAO00": -3,
}


def _haversine_miles(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * math.asin(min(1.0, math.sqrt(h)))


def _team_distance_and_shift(team: str, game_stadium_id: str | None) -> tuple[float, float]:
    """This team's travel distance (miles) and time-zone shift (hours) to
    reach `game_stadium_id`, relative to their own home stadium. (0, 0) if
    any coordinate/offset is unknown -- same neutral fallback as this
    project's other data-missing cases, never a fabricated number."""
    home_stadium = TEAM_HOME_STADIUM.get(team)
    home_coords = STADIUM_COORDS.get(home_stadium) if home_stadium else None
    game_coords = STADIUM_COORDS.get(game_stadium_id) if game_stadium_id else None
    home_tz = STADIUM_TZ_OFFSET.get(home_stadium) if home_stadium else None
    game_tz = STADIUM_TZ_OFFSET.get(game_stadium_id) if game_stadium_id else None
    if home_coords is None or game_coords is None:
        return 0.0, 0.0
    dist = _haversine_miles(home_coords, game_coords)
    shift = (game_tz - home_tz) if (home_tz is not None and game_tz is not None) else 0.0
    return dist, shift


def travel_features(home_team: str, away_team: str, game_stadium_id: str | None) -> tuple[float, float]:
    """(travel_distance_diff, travel_tz_diff) for one game -- positive
    values mean the AWAY team travelled farther / shifted more time zones
    than the home team did to be at this stadium (0 for the home team on
    all but a true neutral-site game)."""
    home_dist, home_shift = _team_distance_and_shift(home_team, game_stadium_id)
    away_dist, away_shift = _team_distance_and_shift(away_team, game_stadium_id)
    travel_distance_diff = (away_dist - home_dist) / 1000.0
    travel_tz_diff = (away_shift - home_shift) / 3.0
    return travel_distance_diff, travel_tz_diff
