"""
SQLite results store -- replaces the "save CSVs to Google Drive by
season/week" approach with one real, queryable database file
(data/nfl_edge.sqlite). This is what section 18 of the brief asks for:
a growing table of every game/pick/result that can be sliced by edge
bucket, odds bucket, home/away, favorite/underdog, etc.

Cards are stored append-only (one row per snapshot each time the odds
are refreshed for a game -- see section 16), so closing-line-value
analysis is possible: compare the first snapshot's price to the last.
Grading always operates on the latest snapshot per game unless told
otherwise.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_id TEXT NOT NULL,
    game_type TEXT,
    gameday TEXT,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    div_game INTEGER,
    p_home REAL,
    p_home_cal REAL,
    p_final REAL,
    pick_side TEXT NOT NULL,
    pick_team TEXT NOT NULL,
    pick_ml REAL NOT NULL,
    pick_prob REAL NOT NULL,
    fair_ml REAL,
    edge REAL,
    required_edge REAL,
    n_books INTEGER,
    consensus_gap REAL,
    width_dec REAL,
    bet_flag INTEGER NOT NULL,
    reason TEXT,
    odds_source TEXT,
    snapshot_time TEXT NOT NULL,
    training_seasons TEXT
);

CREATE TABLE IF NOT EXISTS grades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id INTEGER NOT NULL REFERENCES cards(id),
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_id TEXT NOT NULL,
    home_score REAL,
    away_score REAL,
    pick_won INTEGER,
    stake REAL,
    profit REAL,
    graded_at TEXT NOT NULL,
    UNIQUE(game_id, card_id)
);

CREATE INDEX IF NOT EXISTS idx_cards_season_week ON cards(season, week);
CREATE INDEX IF NOT EXISTS idx_cards_game ON cards(game_id);
CREATE INDEX IF NOT EXISTS idx_grades_game ON grades(game_id);
"""


@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def save_card(card_df: pd.DataFrame, training_seasons: list[int] | None = None,
              odds_source: str = "unknown") -> int:
    """Append every row of a generated card as a new snapshot. Returns the
    number of rows written."""
    init_db()
    df = card_df.copy()
    df["snapshot_time"] = datetime.now(timezone.utc).isoformat()
    df["training_seasons"] = ",".join(str(s) for s in (training_seasons or []))
    df["odds_source"] = odds_source

    cols = [c.name for c in _table_columns("cards") if c.name != "id"]
    df = df[[c for c in cols if c in df.columns]]
    with connect() as conn:
        df.to_sql("cards", conn, if_exists="append", index=False)
    return len(df)


def _table_columns(table: str):
    with connect() as conn:
        cur = conn.execute(f"PRAGMA table_info({table})")
        return [Col(*row) for row in cur.fetchall()]


class Col:
    def __init__(self, cid, name, type_, notnull, dflt, pk):
        self.name = name


def latest_cards(season: int | None = None, week: int | None = None) -> pd.DataFrame:
    """The most recent snapshot per game_id (optionally filtered)."""
    init_db()
    where = []
    params = []
    if season is not None:
        where.append("season = ?")
        params.append(season)
    if week is not None:
        where.append("week = ?")
        params.append(week)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    q = f"""
    SELECT c.* FROM cards c
    INNER JOIN (
        SELECT game_id, MAX(snapshot_time) AS max_ts FROM cards {where_sql} GROUP BY game_id
    ) latest ON c.game_id = latest.game_id AND c.snapshot_time = latest.max_ts
    """
    with connect() as conn:
        return pd.read_sql_query(q, conn, params=params)


def first_and_last_snapshots(season: int, week: int) -> pd.DataFrame:
    """One row per game with the opening (first) and closing (last)
    snapshot's pick_ml side by side, for closing-line-value analysis."""
    init_db()
    q = """
    SELECT * FROM cards WHERE season = ? AND week = ? ORDER BY game_id, snapshot_time
    """
    with connect() as conn:
        df = pd.read_sql_query(q, conn, params=[season, week])
    if df.empty:
        return df
    first = df.groupby("game_id").first().add_suffix("_open")
    last = df.groupby("game_id").last().add_suffix("_close")
    return first.join(last).reset_index()


def all_grades() -> pd.DataFrame:
    init_db()
    q = """
    SELECT g.*, c.pick_side, c.pick_team, c.pick_ml, c.pick_prob, c.edge,
           c.required_edge, c.bet_flag, c.n_books, c.consensus_gap, c.width_dec,
           c.home_team, c.away_team, c.game_type, c.div_game
    FROM grades g
    INNER JOIN cards c ON g.card_id = c.id
    """
    with connect() as conn:
        return pd.read_sql_query(q, conn)
