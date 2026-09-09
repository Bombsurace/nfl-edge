"""
Live weather forecasts for upcoming (not-yet-played) games, via
Open-Meteo (free, no API key: https://open-meteo.com).

Why this exists: `data.py`'s `temp`/`wind` columns only ever hold the
*actual, post-game* conditions nflverse recorded -- they're NaN for any
game that hasn't been played yet, which is exactly the case the
interactive tuner's "this week" tab needs weather for. This module
fills that specific gap: given a game's stadium and local kickoff time,
ask Open-Meteo's hourly forecast for that stadium's coordinates and
pull the temperature/wind for the hour closest to kickoff.

Honest about its limits, on purpose:
- Open-Meteo's forecast horizon is about 16 days. A game further out
  than that gets no forecast -- `fetch_forecast` returns (None, None)
  rather than guessing, and callers fall back to the same neutral
  (severity 0) treatment used for indoor games today.
- No stadium coordinates (see stadiums.py) -> same neutral fallback.
- Any request error (network, rate limit, bad response shape) -> same
  neutral fallback, logged to stderr, never a crash and never a made-up
  number standing in as if it were real.
- **This code needs outbound network access to api.open-meteo.com.**
  Some hosting environments -- including, as of this writing, the cloud
  sandbox this project's own scheduled refresh runs in -- restrict
  outbound requests to an allowlist that doesn't include general
  third-party APIs, so a call from there fails immediately (a
  ProxyError, not a timeout) and the fallback kicks in every time. If
  you run this from a normal machine (your own laptop, a CI runner, a
  host without that restriction) it works as designed. See the
  README's weather section for how to tell which situation you're in.
"""
from __future__ import annotations

import sys
from datetime import datetime

import requests

from .stadiums import coords_for_stadium

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_S = 10


def fetch_forecast(stadium_id: str | None, date_str: str, time_str: str) -> tuple[float | None, float | None]:
    """Returns (temp_f, wind_mph) for the forecast hour closest to
    `time_str` (local clock time, e.g. "13:00") on `date_str`
    ("YYYY-MM-DD") at the given stadium, or (None, None) if a real
    forecast isn't available for any reason (too far out, unknown
    stadium, request failure). `timezone=auto` makes Open-Meteo return
    hourly timestamps in the stadium's own local time, matching
    nflverse's local `gameday`/`gametime` fields with no manual
    timezone math needed."""
    coords = coords_for_stadium(stadium_id)
    if coords is None:
        return None, None
    lat, lon = coords

    try:
        resp = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,wind_speed_10m",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": "auto",
                "start_date": date_str,
                "end_date": date_str,
            },
            timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        payload = resp.json()
        hourly = payload["hourly"]
        times = hourly["time"]
        temps = hourly["temperature_2m"]
        winds = hourly["wind_speed_10m"]
    except Exception as exc:  # network error, non-200, unexpected shape -- never fatal
        print(f"[weather_forecast] no forecast for stadium {stadium_id} on {date_str}: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return None, None

    target = _parse_hhmm(time_str)
    if target is None or not times:
        return None, None

    best_idx, best_diff = None, None
    for i, t in enumerate(times):
        try:
            hh, mm = t.split("T")[1].split(":")[:2]
            cand = int(hh) * 60 + int(mm)
        except (IndexError, ValueError):
            continue
        diff = abs(cand - target)
        if best_diff is None or diff < best_diff:
            best_idx, best_diff = i, diff

    if best_idx is None:
        return None, None
    temp = temps[best_idx]
    wind = winds[best_idx]
    if temp is None or wind is None:
        return None, None
    return float(temp), float(wind)


def _parse_hhmm(time_str: str) -> int | None:
    try:
        hh, mm = time_str.split(":")[:2]
        return int(hh) * 60 + int(mm)
    except (ValueError, AttributeError):
        return None
