"""Deterministic NSE options analytics with a synthetic fallback."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
from typing import Any, Final
from urllib.parse import quote

import pandas as pd
import requests
import yfinance as yf


NSE_HOME_URL: Final[str] = "https://www.nseindia.com"
NSE_INDEX_CHAIN_URL: Final[str] = (
    "https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
)
NSE_EQUITY_CHAIN_URL: Final[str] = (
    "https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
)
NSE_USER_AGENT: Final[str] = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "Chrome/124.0.0.0 Safari/537.36"
)

OPTION_COLUMNS: Final[tuple[str, ...]] = (
    "strikePrice",
    "call_open_interest",
    "call_change_in_oi",
    "call_ltp",
    "put_open_interest",
    "put_change_in_oi",
    "put_ltp",
)

_OHLCV_COLUMNS: Final[tuple[str, ...]] = ("Open", "High", "Low", "Close", "Volume")


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": NSE_USER_AGENT,
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": NSE_HOME_URL + "/",
            "Connection": "keep-alive",
        }
    )
    return session


def _is_index_symbol(symbol: str) -> bool:
    return symbol.upper() in {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}


def _as_number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def normalize_ohlcv_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance columns and normalize OHLCV names case-insensitively.

    yfinance may return columns such as ``('Close', '^NSEI')`` even for a
    single ticker. The first matching OHLCV level is promoted to its canonical
    name, while unrelated columns are flattened to readable strings.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    normalized = data.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = [
            column[0] if isinstance(column, tuple) else column
            for column in normalized.columns
        ]
    used_names: set[str] = set()
    output_names: list[str] = []
    for column in normalized.columns:
        levels = column if isinstance(column, tuple) else (column,)
        labels = [str(level).strip() for level in levels if str(level).strip()]
        folded = {label.casefold() for label in labels}
        canonical = next(
            (name for name in _OHLCV_COLUMNS if name.casefold() in folded),
            "_".join(labels) or "column",
        )
        if canonical in used_names:
            suffix = 2
            candidate = f"{canonical}_{suffix}"
            while candidate in used_names:
                suffix += 1
                candidate = f"{canonical}_{suffix}"
            canonical = candidate
        used_names.add(canonical)
        output_names.append(canonical)

    normalized.columns = output_names
    for name in _OHLCV_COLUMNS:
        if name not in normalized.columns:
            normalized[name] = pd.NA
    return normalized


def _normalize_chain_payload(payload: Mapping[str, Any]) -> pd.DataFrame:
    records: list[dict[str, float]] = []
    for item in payload.get("records", {}).get("data", []):
        strike = _as_number(item.get("strikePrice"))
        if not strike:
            continue
        call = item.get("CE") or {}
        put = item.get("PE") or {}
        records.append(
            {
                "strikePrice": strike,
                "call_open_interest": _as_number(call.get("openInterest")),
                "call_change_in_oi": _as_number(call.get("changeinOpenInterest")),
                "call_ltp": _as_number(call.get("lastPrice")),
                "put_open_interest": _as_number(put.get("openInterest")),
                "put_change_in_oi": _as_number(put.get("changeinOpenInterest")),
                "put_ltp": _as_number(put.get("lastPrice")),
            }
        )
    return pd.DataFrame(records, columns=OPTION_COLUMNS).sort_values("strikePrice").reset_index(drop=True)


def _live_spot(symbol: str) -> float:
    """Get a live spot used only to place a dynamic fallback strike grid."""
    ticker = "^NSEI" if symbol.upper() == "NIFTY" else f"{symbol.upper()}.NS"
    history = yf.download(ticker, period="5d", interval="1d", auto_adjust=True, progress=False, threads=False)
    if isinstance(history, pd.DataFrame) and not history.empty:
        df = history.copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        close = pd.to_numeric(df.get("Close"), errors="coerce").dropna()
        if not close.empty:
            return float(close.iloc[-1])
    raise ValueError(f"Live spot unavailable for {symbol}")


def _synthetic_chain(symbol: str, spot: float) -> pd.DataFrame:
    """Create a strike grid centered on the latest live spot when NSE is unavailable."""
    digest = hashlib.sha256(symbol.upper().encode("utf-8")).digest()
    step = 50 if spot < 30_000 else 100
    center = round(spot / step) * step
    strikes = [center + (offset * step) for offset in range(-10, 11)]
    call_peak = 5 + digest[1] % 8
    put_peak = 4 + digest[2] % 8
    rows = []
    for index, strike in enumerate(strikes):
        distance = abs(index - 10)
        rows.append(
            {
                "strikePrice": float(strike),
                "call_open_interest": float((distance + 1) * 1_000 + ((index + call_peak) % 5) * 250),
                "call_change_in_oi": float(((index + digest[3]) % 7 - 3) * 100),
                "call_ltp": float(max(1, center - strike) / 10),
                "put_open_interest": float((distance + 1) * 1_000 + ((index + put_peak) % 5) * 250),
                "put_change_in_oi": float(((index + digest[4]) % 7 - 3) * 100),
                "put_ltp": float(max(1, strike - center) / 10),
            }
        )
    return pd.DataFrame(rows, columns=OPTION_COLUMNS)


def calculate_option_metrics(chain: pd.DataFrame) -> dict[str, float]:
    """Calculate PCR, standard max pain, and the largest OI walls."""
    required = {"strikePrice", "call_open_interest", "put_open_interest"}
    missing = required - set(chain.columns)
    if missing or chain.empty:
        raise ValueError("Option chain requires non-empty strikePrice and call/put open interest columns")

    calls = pd.to_numeric(chain["call_open_interest"], errors="coerce").fillna(0)
    puts = pd.to_numeric(chain["put_open_interest"], errors="coerce").fillna(0)
    strikes = pd.to_numeric(chain["strikePrice"], errors="coerce")
    valid = strikes.notna()
    calls, puts, strikes = calls[valid], puts[valid], strikes[valid]
    total_calls = float(calls.sum())
    total_puts = float(puts.sum())
    pcr = total_puts / total_calls if total_calls else math.nan

    pain_rows = []
    for settlement in strikes:
        call_payout = ((settlement - strikes).clip(lower=0) * calls).sum()
        put_payout = ((strikes - settlement).clip(lower=0) * puts).sum()
        pain_rows.append(float(call_payout + put_payout))
    minimum_pain_index = min(range(len(pain_rows)), key=pain_rows.__getitem__)

    return {
        "put_call_ratio": pcr,
        "max_pain_strike": float(strikes.iloc[minimum_pain_index]),
        "heavy_call_wall": float(strikes.iloc[calls.argmax()]),
        "heavy_put_wall": float(strikes.iloc[puts.argmax()]),
    }


def fetch_nse_option_chain(symbol: str = "NIFTY") -> dict[str, Any]:
    """Fetch and process an NSE option chain, falling back deterministically.

    The returned dictionary contains a normalized ``data`` dataframe, source,
    PCR, standard max-pain strike, heavy call wall, and heavy put wall.
    """
    clean_symbol = symbol.strip().upper()
    if not clean_symbol:
        raise ValueError("symbol must not be empty")

    endpoint_template = NSE_INDEX_CHAIN_URL if _is_index_symbol(clean_symbol) else NSE_EQUITY_CHAIN_URL
    source = "nse"
    live_spot: float | None = None
    try:
        with _session() as session:
            session.get(NSE_HOME_URL, timeout=10)
            response = session.get(
                endpoint_template.format(symbol=quote(clean_symbol)),
                timeout=15,
            )
            response.raise_for_status()
            chain = _normalize_chain_payload(response.json())
            if chain.empty:
                raise ValueError("NSE returned an empty option chain")
    except (requests.RequestException, ValueError, TypeError, KeyError):
        live_spot = _live_spot(clean_symbol)
        step = 50 if live_spot < 30_000 else 100
        atm_strike = round(live_spot / step) * step
        chain = _synthetic_chain(clean_symbol, live_spot)
        source = "synthetic"

    result: dict[str, Any] = {
        "symbol": clean_symbol,
        "source": source,
        "data": chain,
        **calculate_option_metrics(chain),
    }
    if live_spot is not None:
        result.update(
            {
                "spot": live_spot,
                "atm_strike": float(atm_strike),
                "dynamic_support": float(atm_strike - (2 * step)),
                "dynamic_resistance": float(atm_strike + (2 * step)),
            }
        )
    return result


def _prior_session_values(history: pd.DataFrame) -> tuple[float, float, float]:
    history = normalize_ohlcv_columns(history)
    required = {"High", "Low", "Close"}
    missing = required - set(history.columns)
    if missing or history.empty:
        raise ValueError("Historical data requires High, Low, and Close columns")
    row = history.iloc[-1]
    return _as_number(row["High"]), _as_number(row["Low"]), _as_number(row["Close"])


def calculate_camarilla_levels(history: pd.DataFrame) -> dict[str, float]:
    """Calculate H1-H4 and L1-L4 from the prior session's OHLC values."""
    high, low, close = _prior_session_values(history)
    session_range = high - low
    if session_range < 0:
        raise ValueError("High must be greater than or equal to Low")
    multiplier = 1.1
    return {
        "H1": close + session_range * multiplier / 12,
        "H2": close + session_range * multiplier / 6,
        "H3": close + session_range * multiplier / 4,
        "H4": close + session_range * multiplier / 2,
        "L1": close - session_range * multiplier / 12,
        "L2": close - session_range * multiplier / 6,
        "L3": close - session_range * multiplier / 4,
        "L4": close - session_range * multiplier / 2,
    }


__all__ = [
    "NSE_INDEX_CHAIN_URL",
    "NSE_EQUITY_CHAIN_URL",
    "OPTION_COLUMNS",
    "normalize_ohlcv_columns",
    "calculate_option_metrics",
    "fetch_nse_option_chain",
    "calculate_camarilla_levels",
]
