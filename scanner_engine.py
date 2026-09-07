"""Parallel NSE market scanners and the daily Morning Digest."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Final

import pandas as pd
import streamlit as st
import yfinance as yf

from database import DEFAULT_DATABASE_PATH, get_scan_batch
from fno_engine import normalize_ohlcv_columns


DEFAULT_MAX_WORKERS: Final[int] = 8
NIFTY_BENCHMARK: Final[str] = "^NSEI"


def _yfinance_symbol(symbol: str) -> str:
    """Convert a plain NSE symbol to its Yahoo Finance ticker."""
    return symbol if symbol.startswith("^") or symbol.endswith(".NS") else f"{symbol}.NS"


def _fetch_symbol_history(
    symbol: str,
    period: str,
    interval: str,
) -> tuple[str, pd.DataFrame]:
    """Fetch one symbol's history without allowing one failure to stop the scan."""
    try:
        history = yf.download(
            _yfinance_symbol(symbol),
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if history.empty:
            return symbol, pd.DataFrame()
        return symbol, normalize_ohlcv_columns(history)
    except Exception:
        return symbol, pd.DataFrame()


def fetch_historical_data(
    symbols: Sequence[str],
    period: str = "2y",
    interval: str = "1d",
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, pd.DataFrame]:
    """Fetch historical data for symbols in parallel with a thread pool.

    Failed or empty downloads are retained as empty dataframes so a single
    unavailable ticker does not abort the remaining batch.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be greater than zero")

    unique_symbols = list(dict.fromkeys(symbols))
    histories: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_symbol_history, symbol, period, interval): symbol
            for symbol in unique_symbols
        }
        for future in as_completed(futures):
            symbol, history = future.result()
            histories[symbol] = history
    return histories


def _close_series(history: pd.DataFrame) -> pd.Series:
    """Extract a clean close-price series from standard or multi-index data."""
    if history.empty:
        return pd.Series(dtype="float64")

    history = normalize_ohlcv_columns(history)
    if "Close" not in history:
        return pd.Series(dtype="float64")

    close = history["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna().astype(float)


def _scan_result(
    rows: list[dict[str, float | str]],
    score_column: str,
) -> pd.DataFrame:
    """Build a consistently ranked scan result."""
    if not rows:
        return pd.DataFrame(columns=["symbol", score_column])
    return (
        pd.DataFrame(rows)
        .sort_values(score_column, ascending=False)
        .reset_index(drop=True)
    )


def scan_15_day_momentum(
    historical_data: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Rank symbols by their trailing 15-trading-day price return."""
    rows: list[dict[str, float | str]] = []
    for symbol, history in historical_data.items():
        close = _close_series(history)
        if len(close) <= 15:
            continue
        rows.append(
            {
                "symbol": symbol,
                "momentum_15d": float(close.pct_change(15).iloc[-1] * 100),
            }
        )
    return _scan_result(rows, "momentum_15d")


def scan_90_day_alpha(
    historical_data: Mapping[str, pd.DataFrame],
    benchmark_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rank symbols by 90-day return relative to the Nifty benchmark."""
    benchmark_return = 0.0
    benchmark_close = _close_series(benchmark_data) if benchmark_data is not None else pd.Series(dtype="float64")
    if len(benchmark_close) > 90:
        benchmark_return = float(benchmark_close.pct_change(90).iloc[-1])

    rows: list[dict[str, float | str]] = []
    for symbol, history in historical_data.items():
        close = _close_series(history)
        if len(close) <= 90:
            continue
        stock_return = float(close.pct_change(90).iloc[-1])
        rows.append(
            {
                "symbol": symbol,
                "return_90d": stock_return * 100,
                "alpha_90d": (stock_return - benchmark_return) * 100,
            }
        )
    return _scan_result(rows, "alpha_90d")


def scan_long_term_compounders(
    historical_data: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Rank symbols by annualized return and consistency over available history."""
    rows: list[dict[str, float | str]] = []
    for symbol, history in historical_data.items():
        close = _close_series(history)
        if len(close) < 126 or close.iloc[0] <= 0:
            continue
        years = max(len(close) / 252, 1 / 252)
        cagr = (close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1
        positive_days = float(close.pct_change().dropna().gt(0).mean() * 100)
        rows.append(
            {
                "symbol": symbol,
                "cagr": float(cagr * 100),
                "positive_days_pct": positive_days,
            }
        )
    return _scan_result(rows, "cagr")


@st.cache_data(ttl=86400, show_spinner=False)
def morning_digest(
    nse_symbols: Sequence[str],
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, pd.DataFrame]:
    """Build and cache the daily Morning Digest for the next NSE batch.

    ``get_scan_batch`` reads and updates ``scan_state`` so repeated calls on
    the same date use the same 100-stock slice, while the next date advances.
    """
    if not nse_symbols:
        raise ValueError("nse_symbols must contain at least one ticker")

    start, end = get_scan_batch(len(nse_symbols), database_path)
    batch_symbols = list(nse_symbols[start:end])
    histories = fetch_historical_data(batch_symbols + [NIFTY_BENCHMARK])
    benchmark_data = histories.pop(NIFTY_BENCHMARK, pd.DataFrame())

    return {
        "15-Day Momentum": scan_15_day_momentum(histories),
        "90-Day Alpha": scan_90_day_alpha(histories, benchmark_data),
        "Long-Term Compounders": scan_long_term_compounders(histories),
    }


__all__ = [
    "DEFAULT_MAX_WORKERS",
    "NIFTY_BENCHMARK",
    "fetch_historical_data",
    "scan_15_day_momentum",
    "scan_90_day_alpha",
    "scan_long_term_compounders",
    "morning_digest",
]
