"""
Stadium coordinates, keyed by nflverse's `stadium_id` (not team, since a
few teams have played "home" games at a neutral site -- London, Mexico
City, Melbourne, a temporary home during a stadium rebuild, etc; keying
by the actual venue for that game is more correct than keying by team).

Only used for weather: `weather_forecast.py` looks up a game's
`stadium_id` here to know where to ask for a forecast. A stadium missing
from this table (a new venue nflverse hasn't been added to yet, or a
one-off international site) just means weather forecasting is skipped
for that game -- same neutral fallback as an indoor stadium, not a
crash and not a fabricated number. Coordinates are approximate (city-
center-of-stadium precision) -- fine for "how cold/windy will it be,"
not surveyed exact.
"""
from __future__ import annotations

# stadium_id -> (latitude, longitude)
STADIUM_COORDS: dict[str, tuple[float, float]] = {
    # Current (2024+) primary team stadiums
    "PHO00": (33.5276, -112.2626),   # State Farm Stadium (ARI) -- dome
    "ATL97": (33.7554, -84.4009),    # Mercedes-Benz Stadium (ATL) -- retractable, usually closed
    "BAL00": (39.2780, -76.6227),    # M&T Bank Stadium (BAL)
    "BUF00": (42.7738, -78.7870),    # Highmark Stadium (BUF)
    "CAR00": (35.2258, -80.8528),    # Bank of America Stadium (CAR)
    "CHI98": (41.8623, -87.6167),    # Soldier Field (CHI)
    "CIN00": (39.0955, -84.5161),    # Paycor Stadium (CIN)
    "CLE00": (41.5061, -81.6995),    # Huntington Bank Field (CLE)
    "DAL00": (32.7473, -97.0945),    # AT&T Stadium (DAL) -- retractable, usually closed
    "DEN00": (39.7439, -105.0201),   # Empower Field at Mile High (DEN)
    "DET00": (42.3400, -83.0456),    # Ford Field (DET) -- dome
    "GNB00": (44.5013, -88.0622),    # Lambeau Field (GB)
    "HOU00": (29.6847, -95.4107),    # NRG Stadium (HOU) -- retractable
    "IND00": (39.7601, -86.1639),    # Lucas Oil Stadium (IND) -- retractable
    "JAX00": (30.3239, -81.6373),    # EverBank Stadium (JAX)
    "KAN00": (39.0489, -94.4839),    # GEHA Field at Arrowhead (KC)
    "VEG00": (36.0909, -115.1833),   # Allegiant Stadium (LV) -- dome
    "LAX01": (33.9535, -118.3392),   # SoFi Stadium (LA/LAC) -- fixed translucent roof, treated as dome
    "MIA00": (25.9580, -80.2389),    # Hard Rock Stadium (MIA) -- open-air canopy
    "MIN01": (44.9735, -93.2575),    # U.S. Bank Stadium (MIN) -- dome
    "BOS00": (42.0909, -71.2643),    # Gillette Stadium (NE)
    "NOR00": (29.9511, -90.0812),    # Caesars Superdome (NO) -- dome
    "NYC01": (40.8135, -74.0745),    # MetLife Stadium (NYG/NYJ)
    "PHI00": (39.9008, -75.1675),    # Lincoln Financial Field (PHI)
    "PIT00": (40.4468, -80.0158),    # Acrisure Stadium (PIT)
    "SEA00": (47.5952, -122.3316),   # Lumen Field (SEA)
    "SFO01": (37.4030, -121.9700),   # Levi's Stadium (SF)
    "TAM00": (27.9759, -82.5033),    # Raymond James Stadium (TB)
    "NAS00": (36.1665, -86.7713),    # Nissan Stadium (TEN)
    "WAS00": (38.9078, -76.8645),    # (old) FedExField (WAS)
    "WAS01": (38.9078, -76.8645),    # Northwest Stadium / Commanders Field (WAS)
    # International / occasional neutral sites
    "LON00": (51.5560, -0.2795),     # Wembley Stadium
    "LON01": (51.4562, -0.3416),     # Twickenham Stadium
    "LON02": (51.6043, -0.0664),     # Tottenham Hotspur Stadium
    "FRA00": (50.0686, 8.6455),      # Deutsche Bank Park, Frankfurt
    "GER00": (48.2188, 11.6247),     # Allianz Arena, Munich
    "MUN01": (48.2188, 11.6247),     # FC Bayern Munich Stadium (Allianz Arena)
    "MEX00": (19.3029, -99.1505),    # Estadio Azteca / Banorte, Mexico City
    "MEL00": (-37.8199, 144.9834),   # Melbourne Cricket Ground
    "RIO00": (-22.9122, -43.2302),   # Maracana, Rio de Janeiro
    "SAO00": (-23.5449, -46.4728),   # Arena Corinthians, Sao Paulo
    "PAR00": (48.9244, 2.3601),      # Stade de France, Paris
    "MAD01": (40.4531, -3.6883),     # Santiago Bernabeu, Madrid
}


def coords_for_stadium(stadium_id: str | None) -> tuple[float, float] | None:
    if not stadium_id:
        return None
    return STADIUM_COORDS.get(stadium_id)
