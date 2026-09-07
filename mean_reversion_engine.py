"""Statistical mean-reversion scans and defined-risk option spread blueprints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import pandas as pd

from fno_engine import calculate_option_metrics


DEFAULT_BAND_STD: Final[float] = 2.5
DEFAULT_Z_SCORE_THRESHOLD: Final[float] = 2.2
DEFAULT_RSI_PERIOD: Final[int] = 14
DEFAULT_WALL_TOLERANCE: Final[float] = 0.02
DEFAULT_LOT_SIZE: Final[float] = 1.0

SETUP_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp",
    "direction",
    "spread_type",
    "signal_price",
    "z_score",
    "rsi",
    "wall_strike",
    "entry_strike",
    "hedge_strike",
    "sell_premium",
    "buy_premium",
    "net_credit",
    "max_risk",
    "break_even",
    "break_even_buffer",
    "lot_size",
    "hedge_direction",
)


def _empty_setups() -> pd.DataFrame:
    return pd.DataFrame(columns=SETUP_COLUMNS)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    average_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    relative_strength = average_gain / average_loss.replace(0, float("nan"))
    result = 100 - (100 / (1 + relative_strength))
    return result.where(average_loss.ne(0), 100.0)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _option_chain_and_metadata(fno_data: Any) -> tuple[pd.DataFrame, dict[str, float]]:
    if isinstance(fno_data, pd.DataFrame):
        chain = fno_data.copy()
        metadata: dict[str, float] = {}
    elif isinstance(fno_data, Mapping):
        chain_value = fno_data.get("data")
        if not isinstance(chain_value, pd.DataFrame):
            raise ValueError("fno_data must contain a pandas DataFrame under 'data'")
        chain = chain_value.copy()
        metadata = {
            key: _number(fno_data[key])
            for key in ("heavy_call_wall", "heavy_put_wall", "lot_size")
            if key in fno_data and fno_data[key] is not None
        }
    else:
        raise TypeError("fno_data must be an option-chain DataFrame or result mapping")

    required = {
        "strikePrice",
        "call_ltp",
        "put_ltp",
        "call_open_interest",
        "put_open_interest",
    }
    missing = required - set(chain.columns)
    if missing or chain.empty:
        raise ValueError(f"F&O data is missing required columns: {', '.join(sorted(missing))}")

    if "heavy_call_wall" not in metadata or "heavy_put_wall" not in metadata:
        metadata.update(calculate_option_metrics(chain))
    metadata["lot_size"] = metadata.get("lot_size", DEFAULT_LOT_SIZE)
    if metadata["lot_size"] <= 0:
        raise ValueError("lot_size must be greater than zero")
    return chain.sort_values("strikePrice").reset_index(drop=True), metadata


def _nearest_strike(chain: pd.DataFrame, target: float, direction: int = 0) -> int | None:
    strikes = pd.to_numeric(chain["strikePrice"], errors="coerce")
    candidates = pd.Series(True, index=chain.index)
    if direction < 0:
        candidates = strikes < target
    elif direction > 0:
        candidates = strikes > target
    candidates &= strikes.notna()
    if not candidates.any():
        return None
    distances = (strikes[candidates] - target).abs()
    return int(distances.idxmin())


def _spread_blueprint(
    chain: pd.DataFrame,
    metadata: Mapping[str, float],
    direction: str,
    signal_price: float,
    z_score: float,
    rsi: float,
    timestamp: Any,
) -> dict[str, Any] | None:
    if direction == "LONG":
        wall = metadata["heavy_put_wall"]
        sell_index = _nearest_strike(chain, wall)
        if sell_index is None or wall >= signal_price:
            return None
        hedge_index = _nearest_strike(chain, wall, direction=-1)
        sell_premium_column, buy_premium_column = "put_ltp", "put_ltp"
        hedge_direction = -1
    else:
        wall = metadata["heavy_call_wall"]
        sell_index = _nearest_strike(chain, wall)
        if sell_index is None or wall <= signal_price:
            return None
        hedge_index = _nearest_strike(chain, wall, direction=1)
        sell_premium_column, buy_premium_column = "call_ltp", "call_ltp"
        hedge_direction = 1

    if hedge_index is None:
        return None
    entry_strike = _number(chain.loc[sell_index, "strikePrice"])
    hedge_strike = _number(chain.loc[hedge_index, "strikePrice"])
    sell_premium = _number(chain.loc[sell_index, sell_premium_column])
    buy_premium = _number(chain.loc[hedge_index, buy_premium_column])
    net_credit = sell_premium - buy_premium
    width = abs(entry_strike - hedge_strike)
    if net_credit <= 0 or width <= 0:
        return None

    lot_size = metadata["lot_size"]
    if direction == "LONG":
        break_even = entry_strike - net_credit
        break_even_buffer = signal_price - break_even
        spread_type = "Bull Put Credit Spread"
    else:
        break_even = entry_strike + net_credit
        break_even_buffer = break_even - signal_price
        spread_type = "Bear Call Credit Spread"

    return {
        "timestamp": timestamp,
        "direction": direction,
        "spread_type": spread_type,
        "signal_price": signal_price,
        "z_score": z_score,
        "rsi": rsi,
        "wall_strike": wall,
        "entry_strike": entry_strike,
        "hedge_strike": hedge_strike,
        "sell_premium": sell_premium,
        "buy_premium": buy_premium,
        "net_credit": net_credit * lot_size,
        "max_risk": (width - net_credit) * lot_size,
        "break_even": break_even,
        "break_even_buffer": break_even_buffer,
        "lot_size": lot_size,
        "hedge_direction": hedge_direction,
    }


def scan_reversion_setups(
    ohlcv_df: pd.DataFrame,
    fno_data: Any,
    wall_tolerance: float = DEFAULT_WALL_TOLERANCE,
) -> pd.DataFrame:
    """Return multi-condition exhaustion setups with defined-risk spreads.

    A long setup requires price at or below the lower 2.5-sigma band, a
    z-score at or below -2.2, RSI below 25, and proximity to the put wall.
    A short setup mirrors those conditions at the upper band and call wall.
    ``max_risk`` and ``net_credit`` are multiplied by the supplied lot size.
    """
    if wall_tolerance < 0:
        raise ValueError("wall_tolerance must be non-negative")
    required = {"Close"}
    missing = required - set(ohlcv_df.columns)
    if missing:
        raise ValueError(f"OHLCV data is missing required columns: {', '.join(sorted(missing))}")
    if ohlcv_df.empty:
        return _empty_setups()

    chain, metadata = _option_chain_and_metadata(fno_data)
    close = pd.to_numeric(ohlcv_df["Close"], errors="coerce")
    sma = close.rolling(20, min_periods=20).mean()
    standard_deviation = close.rolling(20, min_periods=20).std(ddof=0)
    upper_band = sma + DEFAULT_BAND_STD * standard_deviation
    lower_band = sma - DEFAULT_BAND_STD * standard_deviation
    z_score = (close - sma) / standard_deviation.replace(0, float("nan"))
    rsi = _rsi(close, DEFAULT_RSI_PERIOD)

    results: list[dict[str, Any]] = []
    for index in ohlcv_df.index:
        price = _number(close.loc[index], default=float("nan"))
        if pd.isna(price) or pd.isna(z_score.loc[index]) or pd.isna(rsi.loc[index]):
            continue
        put_wall = metadata["heavy_put_wall"]
        call_wall = metadata["heavy_call_wall"]
        put_collision = abs(price - put_wall) / max(abs(price), 1) <= wall_tolerance
        call_collision = abs(price - call_wall) / max(abs(price), 1) <= wall_tolerance
        timestamp = index

        if (
            price <= lower_band.loc[index]
            and z_score.loc[index] <= -DEFAULT_Z_SCORE_THRESHOLD
            and rsi.loc[index] < 25
            and put_collision
        ):
            setup = _spread_blueprint(
                chain, metadata, "LONG", price, float(z_score.loc[index]), float(rsi.loc[index]), timestamp
            )
            if setup:
                results.append(setup)
        elif (
            price >= upper_band.loc[index]
            and z_score.loc[index] >= DEFAULT_Z_SCORE_THRESHOLD
            and rsi.loc[index] > 75
            and call_collision
        ):
            setup = _spread_blueprint(
                chain, metadata, "SHORT", price, float(z_score.loc[index]), float(rsi.loc[index]), timestamp
            )
            if setup:
                results.append(setup)

    if not results:
        return _empty_setups()
    return pd.DataFrame(results).sort_values("timestamp").reset_index(drop=True)


__all__ = [
    "DEFAULT_BAND_STD",
    "DEFAULT_Z_SCORE_THRESHOLD",
    "DEFAULT_RSI_PERIOD",
    "scan_reversion_setups",
]
