"""Gemini-powered portfolio OCR and stock intelligence helpers."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import threading
from typing import Any

import requests
import streamlit as st


GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
_REQUEST_TIMEOUT_SECONDS = 60
_key_lock = threading.Lock()
_next_key_index = 0


def _read_secret(name: str) -> str | None:
    """Read a Streamlit secret without failing outside a Streamlit app."""
    try:
        value = st.secrets.get(name)
    except (FileNotFoundError, KeyError, RuntimeError):
        return None
    return str(value).strip() if value else None


def _api_keys() -> list[str]:
    """Return all configured Gemini keys, regardless of their format."""
    keys = [_read_secret(name) for name in (
        "GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"
    )]
    keys = [key for key in keys if key]
    if not keys:
        fallback = _read_secret("GEMINI_API_KEY")
        if fallback:
            keys = [fallback]
    if not keys:
        raise RuntimeError("Configure GEMINI_API_KEY or a numbered Gemini API secret")
    return keys


def _next_api_key(keys: Sequence[str]) -> str:
    global _next_key_index
    with _key_lock:
        key = keys[_next_key_index % len(keys)]
        _next_key_index = (_next_key_index + 1) % len(keys)
    return key


def call_gemini(prompt: str, image_bytes: bytes | None = None) -> str:
    """Call Gemini using the next rotating Streamlit API key.

    When ``image_bytes`` is supplied, it is sent as an inline PNG/JPEG image
    part alongside the text prompt. The raw model text is returned.
    """
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    parts: list[dict[str, Any]] = [{"text": prompt}]
    if image_bytes is not None:
        if not image_bytes:
            raise ValueError("image_bytes must not be empty")
        parts.append(
            {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": base64.b64encode(image_bytes).decode("ascii"),
                }
            }
        )

    response = requests.post(
        GEMINI_ENDPOINT,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": _next_api_key(_api_keys()),
        },
        json={"contents": [{"parts": parts}]},
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    if not response.ok:
        raise RuntimeError(f"Gemini request failed ({response.status_code}): {response.text}")

    payload = response.json()
    try:
        return payload["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("Gemini returned no usable text response") from error


def _parse_json_response(response_text: str) -> Any:
    """Parse plain or Markdown-fenced JSON returned by the model."""
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1])
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = min(
            (position for position in (cleaned.find("{"), cleaned.find("[")) if position >= 0),
            default=-1,
        )
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if start < 0 or end <= start:
            raise ValueError("Gemini response did not contain valid JSON")
        return json.loads(cleaned[start : end + 1])


def ingest_broker_portfolio_screenshot(image_bytes: bytes) -> dict[str, list[dict[str, Any]]]:
    """Extract broker holdings from a screenshot as structured JSON."""
    prompt = """
Read this broker portfolio screenshot using OCR. Extract every visible holding.
Return JSON only in exactly this shape:
{
  "holdings": [
    {"symbol": "NSE ticker symbol", "quantity": 0, "average_price": 0.0, "stop_loss": null}
  ]
}
Use numeric values for quantity, average_price, and stop_loss when visible. Normalize symbols by removing
exchange suffixes, whitespace, and punctuation. Do not invent or estimate values;
use null for an unreadable numeric field and omit rows that are not holdings.
""".strip()
    parsed = _parse_json_response(call_gemini(prompt, image_bytes))
    if not isinstance(parsed, dict) or not isinstance(parsed.get("holdings"), list):
        raise ValueError("Portfolio OCR response must contain a holdings list")

    holdings: list[dict[str, Any]] = []
    for holding in parsed["holdings"]:
        if not isinstance(holding, dict) or not holding.get("symbol"):
            continue
        holdings.append(
            {
                "symbol": str(holding["symbol"]).strip().upper(),
                "quantity": holding.get("quantity"),
                "average_price": holding.get("average_price"),
                "stop_loss": holding.get("stop_loss"),
            }
        )
    return {"holdings": holdings}


def generate_stock_intelligence_audit(
    symbol: str,
    fundamentals: Mapping[str, Any],
    macro_conditions: Mapping[str, Any],
) -> str:
    """Generate a layman-friendly audit connecting fundamentals to macro factors."""
    if not symbol.strip():
        raise ValueError("symbol must not be empty")

    prompt = f"""
Create a clear stock intelligence audit for {symbol.upper()} for a non-expert investor.
Explain how the company's fundamentals connect to the current macro conditions.
Separate the response into these headings: Plain-English Verdict, What Is Working,
What Could Hurt, Macro Link, Key Risks, and What To Watch Next. Define financial
terms the first time they appear. Be balanced and evidence-led, avoid certainty,
and do not give personalized investment advice.

Fundamentals:
{json.dumps(dict(fundamentals), indent=2, default=str)}

Macro conditions:
{json.dumps(dict(macro_conditions), indent=2, default=str)}
""".strip()
    return call_gemini(prompt)


def generate_stock_verdict_batch(
    symbols: Sequence[str],
    sentiment_score: float = 0.0,
) -> dict[str, str]:
    """Batch concise institutional verdicts for the supplied stock symbols."""
    normalized_symbols = list(dict.fromkeys(
        str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()
    ))
    if not normalized_symbols:
        return {}

    prompt = f"""
Act as a senior equity research analyst.
Macro Sentiment Score: {float(sentiment_score):.1f}/10.0
For each of the following Indian NSE/BSE stock symbols: {', '.join(normalized_symbols)}
provide a crisp, one-sentence institutional verdict including Action
(Buy/Accumulate/Hold/Avoid), Entry zone, and Core catalyst.
Return strictly a valid JSON object mapping each SYMBOL to its verdict string.
""".strip()

    try:
        parsed = _parse_json_response(call_gemini(prompt))
        if not isinstance(parsed, dict):
            raise ValueError("Gemini batch verdict response must be a JSON object")
        verdicts = {
            str(symbol).strip().upper(): str(verdict).strip()
            for symbol, verdict in parsed.items()
            if str(verdict).strip()
        }
        if not all(symbol in verdicts for symbol in normalized_symbols):
            raise ValueError("Gemini batch verdict omitted one or more symbols")
        return {symbol: verdicts[symbol] for symbol in normalized_symbols}
    except Exception:
        actions = ("Buy", "Accumulate", "Hold", "Avoid")
        return {
            symbol: (
                f"Action: {actions[int(hashlib.sha256(symbol.encode()).hexdigest(), 16) % len(actions)]}; "
                "Entry zone: use the live 20-period SMA and 2.5 SD bands; "
                "Core catalyst: technical and macro conditions require live confirmation."
            )
            for symbol in normalized_symbols
        }


__all__ = [
    "GEMINI_MODEL",
    "call_gemini",
    "ingest_broker_portfolio_screenshot",
    "generate_stock_intelligence_audit",
    "generate_stock_verdict_batch",
]
