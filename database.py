"""SQLite persistence for the NSE alpha terminal."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sqlite3
from typing import Any


DEFAULT_DATABASE_PATH = Path("nse_alpha.db")
SCAN_BATCH_SIZE = 100


SCHEMA = """
CREATE TABLE IF NOT EXISTS portfolio (
    symbol TEXT PRIMARY KEY,
    quantity REAL NOT NULL,
    avg_price REAL NOT NULL,
    entry_date TEXT NOT NULL,
    stop_loss REAL,
    target_price REAL
);

CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scan_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    scan_date TEXT NOT NULL,
    batch_number INTEGER NOT NULL DEFAULT 0,
    batch_size INTEGER NOT NULL DEFAULT 100
);
"""


def get_connection(database_path: str | Path = DEFAULT_DATABASE_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with row access by column name."""
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(database_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
    """Create the application tables if they do not already exist."""
    with get_connection(database_path) as connection:
        connection.executescript(SCHEMA)


def set_config(
    key: str,
    value: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> None:
    """Store or replace a string configuration value."""
    initialize_database(database_path)
    with get_connection(database_path) as connection:
        connection.execute(
            """
            INSERT INTO config (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )


def get_config(
    key: str,
    default: str | None = None,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> str | None:
    """Read a configuration value, returning default when it is absent."""
    initialize_database(database_path)
    with get_connection(database_path) as connection:
        row = connection.execute(
            "SELECT value FROM config WHERE key = ?", (key,)
        ).fetchone()
    return row["value"] if row else default


def upsert_portfolio_position(
    symbol: str,
    quantity: float,
    avg_price: float,
    entry_date: str,
    stop_loss: float | None = None,
    target_price: float | None = None,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> None:
    """Insert a portfolio position or replace the existing position for a symbol."""
    initialize_database(database_path)
    with get_connection(database_path) as connection:
        connection.execute(
            """
            INSERT INTO portfolio
                (symbol, quantity, avg_price, entry_date, stop_loss, target_price)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                quantity = excluded.quantity,
                avg_price = excluded.avg_price,
                entry_date = excluded.entry_date,
                stop_loss = excluded.stop_loss,
                target_price = excluded.target_price
            """,
            (symbol, quantity, avg_price, entry_date, stop_loss, target_price),
        )


def get_scan_batch(
    total_stocks: int,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    today: date | None = None,
) -> tuple[int, int]:
    """Return ``(start, end)`` indexes for today's rotating stock batch.

    The first batch is indexes 0 through 99. A batch advances once when the
    supplied date changes and wraps after all available stocks have been covered.
    """
    if total_stocks < 1:
        raise ValueError("total_stocks must be greater than zero")

    current_date = today or date.today()
    total_batches = (total_stocks + SCAN_BATCH_SIZE - 1) // SCAN_BATCH_SIZE
    initialize_database(database_path)

    with get_connection(database_path) as connection:
        row = connection.execute(
            "SELECT scan_date, batch_number FROM scan_state WHERE id = 1"
        ).fetchone()
        if row is None:
            batch_number = 0
            connection.execute(
                """
                INSERT INTO scan_state (id, scan_date, batch_number, batch_size)
                VALUES (1, ?, ?, ?)
                """,
                (current_date.isoformat(), batch_number, SCAN_BATCH_SIZE),
            )
        elif row["scan_date"] != current_date.isoformat():
            batch_number = (row["batch_number"] + 1) % total_batches
            connection.execute(
                """
                UPDATE scan_state
                SET scan_date = ?, batch_number = ?, batch_size = ?
                WHERE id = 1
                """,
                (current_date.isoformat(), batch_number, SCAN_BATCH_SIZE),
            )
        else:
            batch_number = row["batch_number"] % total_batches

    start = batch_number * SCAN_BATCH_SIZE
    end = min(start + SCAN_BATCH_SIZE, total_stocks)
    return start, end


def initialize_database_and_get_scan_batch(
    total_stocks: int,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    today: date | None = None,
) -> tuple[int, int]:
    """Initialize storage and return today's rotating scan batch."""
    return get_scan_batch(total_stocks, database_path, today)


__all__ = [
    "DEFAULT_DATABASE_PATH",
    "SCAN_BATCH_SIZE",
    "get_connection",
    "initialize_database",
    "set_config",
    "get_config",
    "upsert_portfolio_position",
    "get_scan_batch",
    "initialize_database_and_get_scan_batch",
]
