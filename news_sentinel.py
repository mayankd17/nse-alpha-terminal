"""Batched Indian-market news sentiment analysis."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import requests
import streamlit as st


MARKET_RSS_URL: Final[str] = (
    "https://news.google.com/rss/search?q=Indian+Stock+Market+NSE+NIFTY"
)
GEMINI_MODEL: Final[str] = "gemini-1.5-flash"
GEMINI_ENDPOINT: Final[str] = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
MAX_HEADLINES: Final[int] = 15
_RSS_TIMEOUT_SECONDS: Final[int] = 15
_GEMINI_TIMEOUT_SECONDS: Final[int] = 60


def get_gemini_api_keys() -> list[str]:
    """Read all configured Gemini keys without imposing a key format."""
    try:
        keys = [
            str(st.secrets[name]).strip()
            for name in ("GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3")
            if name in st.secrets and str(st.secrets[name]).strip()
        ]
    except (FileNotFoundError, KeyError, RuntimeError) as error:
        raise RuntimeError("Gemini API secrets are not configured") from error
    if not keys:
        try:
            fallback = str(st.secrets["GEMINI_API_KEY"]).strip()
        except (FileNotFoundError, KeyError, RuntimeError) as error:
            raise RuntimeError("Configure GEMINI_API_KEY or a numbered Gemini API secret") from error
        if fallback:
            keys = [fallback]
    if not keys:
        raise RuntimeError("Configure GEMINI_API_KEY or a numbered Gemini API secret")
    return keys


def get_gemini_api_key() -> str:
    """Read the first configured Gemini key for compatibility."""
    return get_gemini_api_keys()[0]


def fetch_market_rss() -> list[dict[str, str]]:
    """Fetch the top 15 Indian-market business headlines from Google News RSS."""
    request = Request(
        MARKET_RSS_URL,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; NSEAlphaTerminal/1.0)",
            "Accept": "application/rss+xml, application/xml, text/xml",
        },
    )
    try:
        with urlopen(request, timeout=_RSS_TIMEOUT_SECONDS) as response:
            root = ET.fromstring(response.read())
    except (OSError, ET.ParseError):
        return []

    headlines: list[dict[str, str]] = []
    for item in root.findall("./channel/item")[:MAX_HEADLINES]:
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        headlines.append(
            {
                "title": title,
                "link": (item.findtext("link") or "").strip(),
                "published": (item.findtext("pubDate") or "").strip(),
            }
        )
    return headlines


def _headline_text(headlines: Iterable[str | Mapping[str, Any]]) -> list[str]:
    """Normalize headline strings or RSS records into prompt-ready text."""
    normalized: list[str] = []
    for headline in headlines:
        if isinstance(headline, str):
            text = headline.strip()
        else:
            text = str(headline.get("title", "")).strip()
        if text:
            normalized.append(text)
    return normalized[:MAX_HEADLINES]


def _parse_json_response(response_text: str) -> Any:
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1])
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Gemini response did not contain valid JSON")
        return json.loads(cleaned[start : end + 1])


def _extract_response_text(payload: Mapping[str, Any]) -> str:
    try:
        return str(payload["candidates"][0]["content"]["parts"][0]["text"])
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("Gemini returned no usable text response") from error


@st.cache_data(ttl=3600, show_spinner=False)
def audit_macro_sentiment(
    api_key: str | None,
    headlines: Sequence[str | Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit up to 15 headlines in one Gemini call, cached for one hour."""
    api_key = api_key.strip() if api_key else get_gemini_api_key()
    normalized = _headline_text(headlines)
    if not normalized:
        raise ValueError("headlines must contain at least one non-empty headline")

    numbered_headlines = "\n".join(
        f"{index}. {headline}" for index, headline in enumerate(normalized, start=1)
    )
    prompt = f"""
Analyze these {len(normalized)} Indian stock market headlines as one macro sentiment batch.
Return JSON only, with no Markdown or extra keys, in exactly this shape:
{{
  "sentiment_score": 0.0,
  "primary_catalyst": "brief dominant driver",
  "vulnerable_sectors": ["sector 1", "sector 2"],
  "beneficiary_sectors": ["sector 1", "sector 2"]
}}

Rules:
- sentiment_score must be a number from -10.0 (extreme crisis/panic) to +10.0 (extreme euphoria).
- primary_catalyst must summarize the dominant driver, such as crude, rate cuts, or geopolitics.
- vulnerable_sectors must contain 2 or 3 sectors under negative pressure.
- beneficiary_sectors must contain 2 or 3 sectors with policy or growth tailwinds.
- Use concise sector names and do not invent certainty beyond the headlines.

Headlines:
{numbered_headlines}
""".strip()

    response = requests.post(
        GEMINI_ENDPOINT,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key.strip(),
        },
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=_GEMINI_TIMEOUT_SECONDS,
    )
    if not response.ok:
        raise RuntimeError(f"Gemini request failed ({response.status_code}): {response.text}")

    parsed = _parse_json_response(_extract_response_text(response.json()))
    if not isinstance(parsed, dict):
        raise ValueError("Gemini sentiment response must be a JSON object")

    score = parsed.get("sentiment_score")
    try:
        score = float(score)
    except (TypeError, ValueError) as error:
        raise ValueError("sentiment_score must be numeric") from error
    if not -10.0 <= score <= 10.0:
        raise ValueError("sentiment_score must be between -10.0 and 10.0")

    primary_catalyst = parsed.get("primary_catalyst")
    if not isinstance(primary_catalyst, str) or not primary_catalyst.strip():
        raise ValueError("primary_catalyst must be a non-empty string")

    result: dict[str, Any] = {
        "sentiment_score": score,
        "primary_catalyst": primary_catalyst.strip(),
    }
    for field in ("vulnerable_sectors", "beneficiary_sectors"):
        sectors = parsed.get(field)
        if not isinstance(sectors, list) or not 2 <= len(sectors) <= 3:
            raise ValueError(f"{field} must contain 2 or 3 sectors")
        if not all(isinstance(sector, str) and sector.strip() for sector in sectors):
            raise ValueError(f"{field} must contain non-empty sector names")
        result[field] = [sector.strip() for sector in sectors]
    return result


__all__ = [
    "MARKET_RSS_URL",
    "GEMINI_MODEL",
    "MAX_HEADLINES",
    "get_gemini_api_key",
    "get_gemini_api_keys",
    "fetch_market_rss",
    "audit_macro_sentiment",
]
