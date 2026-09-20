"""
SQLite persistence for the prediction history.

Design notes
------------
* **SQL injection is prevented by parameterised queries.** Every value that
  reaches SQL is passed as a ``?`` placeholder parameter - user input is never
  concatenated or f-string-formatted into a query. The only dynamically built
  fragments are the WHERE clause *structure* and the ORDER BY column, and the
  latter is validated against a fixed allow-list.
* Connections are per-call and closed by a context manager. Flask's dev server
  is threaded and a SQLite connection may not be shared across threads, so a
  short-lived connection per request is both the simplest and the safest option
  at this scale.
* The table stores a truncated copy of the input text, not the whole article,
  to keep the database small and to limit how much user data is retained.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Allow this file to be run directly as `python src/database.py` as well as via
# `python -m src.database`. When run directly, Python puts src/ on the path
# instead of the project root, so `from src import ...` would fail. Adding the
# project root here makes both invocations work identically.
# ---------------------------------------------------------------------------
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from src import config
from src.exceptions import DatabaseError

logger = logging.getLogger(__name__)

# How much of the submitted text is retained in the history table.
STORED_TEXT_LIMIT = 1000

# Columns the history list may be sorted by. Anything not in this set is
# rejected, which is what makes the dynamic ORDER BY safe.
SORTABLE_COLUMNS: frozenset[str] = frozenset(
    {"id", "created_at", "prediction", "confidence"}
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    input_text  TEXT    NOT NULL,
    prediction  TEXT    NOT NULL CHECK (prediction IN ('FAKE', 'REAL')),
    confidence  REAL    NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    model_name  TEXT    NOT NULL,
    word_count  INTEGER NOT NULL DEFAULT 0,
    source      TEXT    NOT NULL DEFAULT 'web',
    created_at  TEXT    NOT NULL
);

-- Indexes that match the two ways the history page is queried.
CREATE INDEX IF NOT EXISTS idx_predictions_created_at
    ON predictions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_prediction
    ON predictions (prediction);
"""


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """
    Yield a SQLite connection, committing on success and closing always.

    Rows are returned as ``sqlite3.Row`` so they can be accessed by column
    name and converted to dictionaries for the templates.
    """
    config.ensure_directories()
    connection: Optional[sqlite3.Connection] = None
    try:
        connection = sqlite3.connect(config.DATABASE_PATH, timeout=10)
        connection.row_factory = sqlite3.Row
        # Enforce the CHECK constraints and enable safer concurrent reads.
        connection.execute("PRAGMA foreign_keys = ON")
        yield connection
        connection.commit()
    except sqlite3.Error as exc:
        if connection is not None:
            connection.rollback()
        logger.exception("Database operation failed.")
        raise DatabaseError(f"SQLite error: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()


def init_database() -> None:
    """Create the schema if it does not already exist. Safe to call repeatedly."""
    with get_connection() as connection:
        connection.executescript(SCHEMA)
    logger.info("Database ready at %s", config.DATABASE_PATH)


def save_prediction(
    input_text: str,
    prediction: str,
    confidence: float,
    model_name: str,
    word_count: int = 0,
    source: str = "web",
) -> int:
    """
    Insert one prediction and return its new row id.

    ``input_text`` is truncated to ``STORED_TEXT_LIMIT`` characters.
    """
    stored = input_text[:STORED_TEXT_LIMIT]
    if len(input_text) > STORED_TEXT_LIMIT:
        stored += "..."

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions
                (input_text, prediction, confidence, model_name,
                 word_count, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stored,
                prediction,
                float(confidence),
                model_name,
                int(word_count),
                source,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        return int(cursor.lastrowid or 0)


def fetch_predictions(
    search: str = "",
    label_filter: str = "",
    sort_by: str = "created_at",
    descending: bool = True,
    limit: int = config.HISTORY_PAGE_SIZE,
    offset: int = 0,
) -> List[dict]:
    """
    Return history rows matching the given search and filter.

    Parameters
    ----------
    search:
        Case-insensitive substring match against the stored text.
    label_filter:
        "FAKE", "REAL", or "" for no filter.
    sort_by:
        Column to order by; validated against ``SORTABLE_COLUMNS``.
    """
    clauses: List[str] = []
    params: List[Any] = []

    if search:
        # LIKE with a bound parameter - the wildcards are part of the *value*,
        # so the user's text is never interpolated into the SQL string.
        clauses.append("LOWER(input_text) LIKE ?")
        params.append(f"%{search.lower()}%")

    if label_filter in config.CLASS_NAMES:
        clauses.append("prediction = ?")
        params.append(label_filter)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    # Allow-list check: an attacker-supplied sort column can never reach SQL.
    if sort_by not in SORTABLE_COLUMNS:
        sort_by = "created_at"
    direction = "DESC" if descending else "ASC"

    query = (
        f"SELECT * FROM predictions {where} "
        f"ORDER BY {sort_by} {direction} LIMIT ? OFFSET ?"
    )
    params.extend([max(1, int(limit)), max(0, int(offset))])

    with get_connection() as connection:
        rows = connection.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def count_predictions(search: str = "", label_filter: str = "") -> int:
    """Count history rows matching the same filters as ``fetch_predictions``."""
    clauses: List[str] = []
    params: List[Any] = []

    if search:
        clauses.append("LOWER(input_text) LIKE ?")
        params.append(f"%{search.lower()}%")
    if label_filter in config.CLASS_NAMES:
        clauses.append("prediction = ?")
        params.append(label_filter)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with get_connection() as connection:
        row = connection.execute(
            f"SELECT COUNT(*) AS total FROM predictions {where}", params
        ).fetchone()
    return int(row["total"]) if row else 0


def get_statistics() -> dict:
    """
    Aggregate the history for the analytics dashboard.

    Returns totals, the FAKE/REAL breakdown, average confidence, and a
    day-by-day series covering the last 14 days.
    """
    with get_connection() as connection:
        totals = connection.execute(
            """
            SELECT
                COUNT(*)                                            AS total,
                SUM(CASE WHEN prediction = 'FAKE' THEN 1 ELSE 0 END) AS fake_count,
                SUM(CASE WHEN prediction = 'REAL' THEN 1 ELSE 0 END) AS real_count,
                AVG(confidence)                                      AS avg_confidence
            FROM predictions
            """
        ).fetchone()

        daily = connection.execute(
            """
            SELECT
                DATE(created_at) AS day,
                SUM(CASE WHEN prediction = 'FAKE' THEN 1 ELSE 0 END) AS fake_count,
                SUM(CASE WHEN prediction = 'REAL' THEN 1 ELSE 0 END) AS real_count
            FROM predictions
            WHERE created_at >= DATE('now', '-13 days')
            GROUP BY day
            ORDER BY day ASC
            """
        ).fetchall()

        recent = connection.execute(
            "SELECT * FROM predictions ORDER BY created_at DESC LIMIT 5"
        ).fetchall()

    total = int(totals["total"] or 0)
    return {
        "total": total,
        "fake_count": int(totals["fake_count"] or 0),
        "real_count": int(totals["real_count"] or 0),
        "avg_confidence": round(float(totals["avg_confidence"] or 0.0), 4),
        "fake_percentage": round(100 * (totals["fake_count"] or 0) / total, 1)
        if total
        else 0.0,
        "real_percentage": round(100 * (totals["real_count"] or 0) / total, 1)
        if total
        else 0.0,
        "daily": [dict(row) for row in daily],
        "recent": [dict(row) for row in recent],
    }


def delete_prediction(prediction_id: int) -> bool:
    """Delete a single history row. Returns True if a row was removed."""
    with get_connection() as connection:
        cursor = connection.execute(
            "DELETE FROM predictions WHERE id = ?", (int(prediction_id),)
        )
        return cursor.rowcount > 0


def clear_history() -> int:
    """
    Delete every history row and return how many were removed.

    The confirmation step lives in the UI; by the time this is called the user
    has already confirmed the destructive action.
    """
    with get_connection() as connection:
        total = connection.execute(
            "SELECT COUNT(*) AS total FROM predictions"
        ).fetchone()["total"]
        connection.execute("DELETE FROM predictions")
        # Reset the AUTOINCREMENT counter so ids start from 1 again.
        connection.execute(
            "DELETE FROM sqlite_sequence WHERE name = 'predictions'"
        )
    logger.info("Cleared %d history row(s).", total)
    return int(total)


if __name__ == "__main__":  # pragma: no cover - manual setup helper
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    init_database()
    print(f"Database initialised at {config.DATABASE_PATH}")
    print(f"Existing rows: {count_predictions()}")
