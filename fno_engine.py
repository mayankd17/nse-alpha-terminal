"""Deterministic NSE options analytics with a synthetic fallback."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
from typing import Any, Final
from urllib.parse import quote

import pandas as pd
import requests


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


def _synthetic_chain(symbol: str) -> pd.DataFrame:
    """Create a repeatable fallback chain when the public endpoint is unavailable."""
    digest = hashlib.sha256(symbol.upper().encode("utf-8")).digest()
    center = 18_000 if _is_index_symbol(symbol) and symbol.upper() == "NIFTY" else 45_000
    center += (digest[0] - 128) * 5
    step = 50 if center < 30_000 else 100
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
        chain = _synthetic_chain(clean_symbol)
        source = "synthetic"

    return {
        "symbol": clean_symbol,
        "source": source,
        "data": chain,
        **calculate_option_metrics(chain),
    }


def _prior_session_values(history: pd.DataFrame) -> tuple[float, float, float]:
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
    "calculate_option_metrics",
    "fetch_nse_option_chain",
    "calculate_camarilla_levels",
]
