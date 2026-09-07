"""Market mood scoring, regime overrides, and projection gauges."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import plotly.graph_objects as go


FACTOR_WEIGHTS: Final[dict[str, float]] = {
    "global_scenarios": 20.0,
    "micro_local_policy": 15.0,
    "internal_fundamentals": 25.0,
    "peer_competition": 10.0,
    "external_news_support": 30.0,
}

TAIL_RISK_WEIGHTS: Final[dict[str, float]] = {
    "global_scenarios": 30.0,
    "micro_local_policy": 5.0,
    "internal_fundamentals": 15.0,
    "peer_competition": 5.0,
    "external_news_support": 45.0,
}

TAIL_RISK_VIX_THRESHOLD: Final[float] = 22.0
TAIL_RISK_SENTIMENT_THRESHOLD: Final[float] = -6.0

PREDICTION_HORIZONS: Final[tuple[str, ...]] = (
    "Current",
    "7D",
    "15D",
    "30D",
    "90D",
)


def get_dynamic_weights(
    india_vix: float,
    news_sentiment_score: float,
) -> dict[str, float]:
    """Return baseline or tail-risk weights for the current market regime."""
    if india_vix < 0:
        raise ValueError("india_vix must be non-negative")
    if not -10 <= news_sentiment_score <= 10:
        raise ValueError("news_sentiment_score must be between -10 and 10")
    is_tail_risk = (
        india_vix > TAIL_RISK_VIX_THRESHOLD
        or news_sentiment_score < TAIL_RISK_SENTIMENT_THRESHOLD
    )
    return dict(TAIL_RISK_WEIGHTS if is_tail_risk else FACTOR_WEIGHTS)


def calculate_market_mood_score(
    variables: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
    india_vix: float | None = None,
    news_sentiment_score: float | None = None,
) -> float:
    """Calculate a weighted Market Mood Score from 0 to 100.

    Each variable is expected to be scored from 0 (very negative) to 100
    (very positive). Custom weights may be supplied at runtime; they are
    normalized so their total contribution remains 100 points.
    """
    if (india_vix is None) != (news_sentiment_score is None):
        raise ValueError("india_vix and news_sentiment_score must be provided together")
    active_weights = dict(weights or FACTOR_WEIGHTS)
    if weights is None and india_vix is not None and news_sentiment_score is not None:
        active_weights = get_dynamic_weights(india_vix, news_sentiment_score)
    missing_variables = set(active_weights) - set(variables)
    if missing_variables:
        missing = ", ".join(sorted(missing_variables))
        raise ValueError(f"Missing market mood variables: {missing}")

    if not active_weights or any(weight < 0 for weight in active_weights.values()):
        raise ValueError("Weights must be non-negative and include at least one factor")

    weight_total = sum(active_weights.values())
    if weight_total <= 0:
        raise ValueError("At least one weight must be greater than zero")

    invalid_variables = {
        name for name in active_weights if not 0 <= variables[name] <= 100
    }
    if invalid_variables:
        invalid = ", ".join(sorted(invalid_variables))
        raise ValueError(f"Market mood variables must be between 0 and 100: {invalid}")

    weighted_score = sum(
        variables[name] * weight for name, weight in active_weights.items()
    )
    return weighted_score / weight_total


def calculate_mood_trajectories(
    variables: Mapping[str, float],
    india_vix: float,
    news_sentiment_score: float,
    weekly_momentum: float = 0.0,
    expiry_cycle: float = 0.0,
    macro_drift: float = 0.0,
    monthly_rollover: float = 0.0,
    quarterly_earnings: float = 0.0,
) -> dict[str, float]:
    """Project bounded mood scores across current, weekly, and quarterly horizons.

    All trajectory inputs are directional tilts in the range -100 to 100. The
    function is deterministic so the dashboard remains reproducible; callers
    can supply probabilistic expected tilts from their forecasting models.
    """
    drivers = {
        "weekly_momentum": weekly_momentum,
        "expiry_cycle": expiry_cycle,
        "macro_drift": macro_drift,
        "monthly_rollover": monthly_rollover,
        "quarterly_earnings": quarterly_earnings,
    }
    invalid_drivers = [name for name, value in drivers.items() if not -100 <= value <= 100]
    if invalid_drivers:
        raise ValueError(f"Trajectory drivers must be between -100 and 100: {', '.join(invalid_drivers)}")

    current = calculate_market_mood_score(
        variables,
        india_vix=india_vix,
        news_sentiment_score=news_sentiment_score,
    )
    return {
        "Current": current,
        "7D": _bounded_score(current + weekly_momentum * 0.35 + expiry_cycle * 0.15),
        "15D": _bounded_score(current + macro_drift * 0.30 + weekly_momentum * 0.10),
        "30D": _bounded_score(current + macro_drift * 0.35 + monthly_rollover * 0.25),
        "90D": _bounded_score(current + quarterly_earnings * 0.40 + macro_drift * 0.20),
    }


def _bounded_score(score: float) -> float:
    return max(0.0, min(100.0, float(score)))


def generate_mood_gauges(
    scores_dict: Mapping[str, float] | None = None,
) -> go.Figure:
    """Create five responsive semi-circular mood gauges.

    Scores are in the range 0-100. Missing horizons default to the current
    score, or 50 when no current score is supplied.
    """
    values = dict(scores_dict or {})
    legacy_labels = {"Today": "Current", "3 Months": "90D"}
    values = {legacy_labels.get(key, key): value for key, value in values.items()}
    unknown_horizons = set(values) - set(PREDICTION_HORIZONS)
    if unknown_horizons:
        unknown = ", ".join(sorted(unknown_horizons))
        raise ValueError(f"Unknown prediction horizons: {unknown}")

    invalid_horizons = {horizon for horizon in values if not 0 <= values[horizon] <= 100}
    if invalid_horizons:
        invalid = ", ".join(sorted(invalid_horizons))
        raise ValueError(f"Prediction scores must be between 0 and 100: {invalid}")

    current_score = values.get("Current", 50.0)
    figure = go.Figure()
    for index, horizon in enumerate(PREDICTION_HORIZONS):
        score = values.get(horizon, current_score)
        figure.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=score,
                title={"text": horizon},
                domain={"row": 0, "column": index},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#123c2f"},
                    "steps": [
                        {"range": [0, 30], "color": "#d9534f"},
                        {"range": [30, 50], "color": "#f2c94c"},
                        {"range": [50, 70], "color": "#b7dfb0"},
                        {"range": [70, 100], "color": "#159957"},
                    ],
                    "threshold": {
                        "line": {"color": "#17324d", "width": 3},
                        "thickness": 0.75,
                        "value": score,
                    },
                },
            )
        )

    figure.update_layout(
        grid={"rows": 1, "columns": len(PREDICTION_HORIZONS), "pattern": "independent"},
        height=300,
        margin={"l": 10, "r": 10, "t": 45, "b": 15},
    )
    return figure


def create_market_prediction_gauges(
    predictions: Mapping[str, float] | None = None,
) -> go.Figure:
    """Backward-compatible alias for :func:`generate_mood_gauges`."""
    return generate_mood_gauges(predictions)


__all__ = [
    "FACTOR_WEIGHTS",
    "TAIL_RISK_WEIGHTS",
    "TAIL_RISK_VIX_THRESHOLD",
    "TAIL_RISK_SENTIMENT_THRESHOLD",
    "PREDICTION_HORIZONS",
    "get_dynamic_weights",
    "calculate_market_mood_score",
    "calculate_mood_trajectories",
    "generate_mood_gauges",
    "create_market_prediction_gauges",
]
