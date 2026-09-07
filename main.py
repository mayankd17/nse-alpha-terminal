"""Institutional-style Streamlit dashboard for the NSE Alpha Terminal."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

import pandas as pd
import streamlit as st
import yfinance as yf

from ai_vision import generate_stock_intelligence_audit, generate_stock_verdict_batch, ingest_broker_portfolio_screenshot
from database import get_connection, initialize_database, upsert_portfolio_position
from fno_engine import calculate_camarilla_levels, fetch_nse_option_chain
from macro_engine import (
    FACTOR_WEIGHTS,
    PREDICTION_HORIZONS,
    calculate_market_mood_score,
    calculate_mood_trajectories,
    generate_mood_gauges,
    get_live_macro_sentiment,
    get_dynamic_weights,
)
from mean_reversion_engine import scan_reversion_setups
from news_sentinel import fetch_market_rss
from scanner_engine import fetch_historical_data, morning_digest


DATABASE_PATH = Path("terminal_vault.db")
NSE_UNIVERSE = tuple(
    """
    ADANIENT ADANIPORTS APOLLOHOSP APOLLOTYRE ASIANPAINT AUROPHARMA AXISBANK
    BAJAJ-AUTO BAJAJFINSV BAJFINANCE BALKRISIND BANKBARODA BEL BHEL BPCL
    BHARTIARTL BIOCON BOSCHLTD BRITANNIA CANBK CHOLAFIN CIPLA COALINDIA COFORGE
    COLPAL CONCOR COROMANDEL CUMMINSIND DABUR DALBHARAT DEEPAKNTR DIVISLAB
    DLF DRREDDY EICHERMOT EXIDEIND FEDERALBNK GAIL GLANDFARM GLENMARK GMRINFRA
    GODREJCP GODREJPROP GRASIM HAL HAVELLS HCLTECH HDFCAMC HDFCBANK HDFCLIFE
    HEROMOTOCO HINDALCO HINDCOPPER HINDPETRO HINDUNILVR ICICIBANK ICICIGI
    ICICIPRULI IDEA IDFCFIRSTB IGL INDHOTEL INDIACEM INDIAMART INDIANB INDIGO
    INDUSINDBK INFY IOC IRCTC ITC JINDALSTEL JSWSTEEL JUBLFOOD KOTAKBANK LICHSGFIN
    LT LTI LTIM LUPIN M&M M&MFIN MANAPPURAM MARICO MARUTI MCX MOTHERSON MPHASIS
    MRF MUTHOOTFIN NAUKRI NAVINFLUOR NESTLEIND NMDC NTPC OBEROIRLTY OFSS OIL
    ONGC PAGEIND PEL PERSISTENT PETRONET PFC PIDILITIND PIIND PNB POLYCAB POWERGRID
    PVRINOX RAMCOCEM RBLBANK RECLTD RELIANCE SAIL SBICARD SBILIFE SBIN SHREECEM
    SIEMENS SRF SRTRANSFIN SUNPHARMA SUNTV SUPREMEIND SYNGENE TATACHEM TATACONSUM
    TATAMOTORS TATAPOWER TATASTEEL TCS TECHM TITAN TORNTPHARM TORNTPOWER TRENT
    TVSMOTOR UBL ULTRACEMCO UNIONBANK UPL VBL VEDL VOLTAS WIPRO YESBANK ZEEL
    """.split()
)


DARK_THEME_CSS = """
<style>
:root { color-scheme: dark; }
.stApp { background: #0b1117; color: #d8e1e8; }
[data-testid="stSidebar"] { background: #101820; border-right: 1px solid #24333d; }
[data-testid="stMetric"] { background: #111d26; border: 1px solid #263944; padding: 12px; }
[data-testid="stMetricValue"] { color: #e8f3ee; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
[data-testid="stDataFrame"] { border: 1px solid #263944; }
section[data-testid="stExpander"] { border: 1px solid #263944; background: #0f1921; }
.status-tag { display: inline-block; padding: 4px 9px; margin: 2px 5px 8px 0; border-radius: 3px; font: 700 0.72rem ui-monospace, monospace; letter-spacing: .04em; }
.status-green { color: #b9f6d0; background: #123a2a; border: 1px solid #228653; }
.status-red { color: #ffd0cc; background: #421d20; border: 1px solid #b84a50; }
.status-yellow { color: #ffe4a3; background: #3b3016; border: 1px solid #a9872e; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
</style>
"""


def is_sunday() -> bool:
    """Return whether today is Sunday using the local application clock."""
    return datetime.now().weekday() == 6


def _tag(label: str, value: str, tone: str = "green") -> None:
    st.markdown(
        f'<span class="status-tag status-{tone}">{label}: {value}</span>',
        unsafe_allow_html=True,
    )


def _secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except (FileNotFoundError, KeyError, RuntimeError):
        return None
    return str(value).strip() if value else None


@st.cache_data(ttl=900, show_spinner=False)
def _live_macro_context() -> dict[str, Any]:
    headlines = fetch_market_rss()
    sentiment: dict[str, Any] = {
        "sentiment_score": 0.0,
        "primary_catalyst": "Live sentiment unavailable: configure GEMINI_API_KEY_1",
        "vulnerable_sectors": [],
        "beneficiary_sectors": [],
    }
    if headlines:
        try:
            sentiment = get_live_macro_sentiment(headlines)
        except Exception:
            pass

    history = fetch_historical_data(["^INDIAVIX"], period="5d")
    vix_history = history.get("^INDIAVIX", pd.DataFrame())
    vix = 18.0
    if not vix_history.empty and "Close" in vix_history:
        close = vix_history["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        if not close.dropna().empty:
            vix = float(close.dropna().iloc[-1])
    return {"headlines": headlines, "sentiment": sentiment, "india_vix": vix}


@st.cache_data(ttl=900, show_spinner=False)
def _live_fno_chain() -> dict[str, Any]:
    return fetch_nse_option_chain("NIFTY")


@st.cache_data(ttl=900, show_spinner=False)
def _nifty_history() -> pd.DataFrame:
    return fetch_historical_data(["^NSEI"], period="1y").get("^NSEI", pd.DataFrame())


def _portfolio_rows() -> list[dict[str, Any]]:
    initialize_database(DATABASE_PATH)
    with get_connection(DATABASE_PATH) as connection:
        return [dict(row) for row in connection.execute("SELECT * FROM portfolio ORDER BY symbol")]


def _mood_context() -> tuple[dict[str, float], dict[str, Any], float, bool]:
    context = _live_macro_context()
    sentiment = context["sentiment"]
    sentiment_score = float(sentiment["sentiment_score"])
    vix = float(context["india_vix"])
    variables = {
        "global_scenarios": max(0.0, min(100.0, 70.0 - vix * 1.5)),
        "micro_local_policy": 55.0,
        "internal_fundamentals": 55.0,
        "peer_competition": 55.0,
        "external_news_support": (sentiment_score + 10.0) * 5.0,
    }
    weights = get_dynamic_weights(vix, sentiment_score)
    return variables, context, calculate_market_mood_score(variables, weights), weights == get_dynamic_weights(25.0, -7.0)


def _single_gauge(figure: Any, index: int) -> Any:
    import plotly.graph_objects as go

    single = go.Figure(figure.data[index])
    single.data[0].domain = {"x": [0, 1], "y": [0, 1]}
    single.update_layout(height=250, margin={"l": 20, "r": 20, "t": 45, "b": 10})
    return single


def render_mood_command_center() -> None:
    st.header("Market Mood Command Center")
    try:
        variables, context, current_score, tail_risk = _mood_context()
    except Exception as error:
        st.error(f"Macro context unavailable: {error}")
        variables = {factor: 50.0 for factor in FACTOR_WEIGHTS}
        context = {"india_vix": 18.0, "sentiment": {"sentiment_score": 0.0, "primary_catalyst": "Fallback"}}
        current_score = 50.0
        tail_risk = False

    sentiment = context["sentiment"]
    vix = float(context["india_vix"])
    sentiment_score = float(sentiment["sentiment_score"])
    _tag("GEMINI MACRO", f"{sentiment_score:+.1f}", "green" if sentiment_score >= 0 else "red")
    _tag("INDIA VIX", f"{vix:.2f}", "red" if vix > 22 else "green")
    _tag("REGIME", "TAIL-RISK OVERRIDE" if tail_risk else "NORMAL", "red" if tail_risk else "green")
    st.metric("Current Mood Score", f"{current_score:.1f} / 100", sentiment.get("primary_catalyst", ""))

    scores = calculate_mood_trajectories(
        variables,
        india_vix=vix,
        news_sentiment_score=sentiment_score,
        weekly_momentum=sentiment_score * 2,
        expiry_cycle=0,
        macro_drift=sentiment_score,
        monthly_rollover=0,
        quarterly_earnings=0,
    )
    gauges = generate_mood_gauges(scores)
    for index, horizon in enumerate(PREDICTION_HORIZONS):
        with st.expander(f"{horizon} projection  |  {scores[horizon]:.1f}", expanded=index == 0):
            st.plotly_chart(_single_gauge(gauges, index), use_container_width=True, key=f"mood-{index}")

    with st.expander("Dynamic weight state", expanded=False):
        st.write("Tail-risk mode reallocates attention toward volatility and news when India VIX exceeds 22 or sentiment drops below -6.")
        weight_frame = pd.DataFrame(
            {
                "factor": list(variables),
                "active_weight_pct": [get_dynamic_weights(vix, sentiment_score)[factor] for factor in variables],
                "baseline_weight_pct": [FACTOR_WEIGHTS[factor] for factor in variables],
            }
        )
        st.dataframe(weight_frame, use_container_width=True, hide_index=True)


def _tradingview_widget(symbol: str) -> None:
    encoded = quote(f"NSE:{symbol}")
    html = f"""
    <iframe src="https://www.tradingview.com/widgetembed/?symbol={encoded}&interval=15&hidesidetoolbar=1&symboledit=0&saveimage=0&toolbar_bg=%230f1921&theme=dark&style=1&timezone=Asia%2FKolkata"
      style="width:100%;height:420px;border:0" allowtransparency="true" scrolling="no"></iframe>
    """
    st.components.v1.html(html, height=420, scrolling=False)


def _research_technicals(history: pd.DataFrame) -> dict[str, float]:
    """Calculate the technical values shown in each stock research blueprint."""
    if history.empty:
        return {}
    close = history["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    if close.empty:
        return {}
    sma = close.rolling(20).mean().iloc[-1]
    deviation = close.rolling(20).std(ddof=0).iloc[-1]
    delta = close.diff()
    gains = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    losses = -delta.clip(upper=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    relative_strength = gains / losses.replace(0, float("nan"))
    rsi = (100 - 100 / (1 + relative_strength)).iloc[-1]
    if pd.isna(rsi) and losses.iloc[-1] == 0:
        rsi = 100.0
    latest = float(close.iloc[-1])
    return {
        "cmp": latest,
        "sma_20": float(sma),
        "distance_sma_20_pct": float((latest - sma) / sma * 100),
        "z_score": float((latest - sma) / deviation) if deviation else 0.0,
        "bb_upper_2_5sd": float(sma + 2.5 * deviation),
        "bb_lower_2_5sd": float(sma - 2.5 * deviation),
        "rsi_14": float(rsi),
    }


def _fundamental_metrics(symbol: str) -> dict[str, Any]:
    """Read public Yahoo Finance fundamentals, leaving unavailable fields explicit."""
    try:
        info = yf.Ticker(f"{symbol}.NS").get_info()
    except Exception:
        info = {}
    return {
        "ROE": info.get("returnOnEquity"),
        "Profit margin": info.get("profitMargins"),
        "Sector": info.get("sector", "Unavailable"),
        "Sector peer rank": "Live peer comparison pending",
        "Promoter pledge": "Check exchange filings",
        "ASM/GSM audit": "Check NSE surveillance list",
    }


def _render_research_expander(
    symbol: str,
    history: pd.DataFrame,
    fno_data: Mapping[str, Any] | None = None,
    verdict: str | None = None,
) -> None:
    """Render a complete, clickable research blueprint for one ticker."""
    with st.expander(f"🔍 {symbol} | Comprehensive Research & Computation Blueprint"):
        technicals = _research_technicals(history)
        if not technicals:
            st.warning("Live OHLC data is unavailable for this symbol.")
        else:
            try:
                pivots = calculate_camarilla_levels(history.tail(1))
            except ValueError:
                pivots = {}
            st.markdown("**Exact Math Breakdown**")
            st.dataframe(
                pd.DataFrame([{
                    **technicals,
                    "Camarilla H3": pivots.get("H3"),
                    "Camarilla L3": pivots.get("L3"),
                    "Camarilla H4": pivots.get("H4"),
                    "Camarilla L4": pivots.get("L4"),
                }]),
                use_container_width=True,
                hide_index=True,
            )

        if fno_data is None:
            try:
                fno_data = fetch_nse_option_chain(symbol)
            except Exception as error:
                fno_data = {}
                st.warning(f"Derivatives footprint unavailable: {error}")
        st.markdown("**Institutional Derivatives Footprint**")
        st.json({
            "Call OI concentration / wall": fno_data.get("heavy_call_wall"),
            "Put OI concentration / wall": fno_data.get("heavy_put_wall"),
            "Put-Call Ratio (PCR)": fno_data.get("put_call_ratio"),
            "Max Pain strike magnet": fno_data.get("max_pain_strike"),
            "data source": fno_data.get("source", "unavailable"),
        })

        st.markdown("**Fundamental & Risk Scorecard**")
        st.json(_fundamental_metrics(symbol))
        _tradingview_widget(symbol)
        st.markdown("**AI One-Sentence Verdict**")
        if verdict:
            st.write(verdict)
        else:
            try:
                audit = _audit_for_stock(symbol, {**technicals, **_fundamental_metrics(symbol)})
                st.write(audit.get("verdict", audit) if isinstance(audit, dict) else audit)
            except Exception as error:
                st.warning(f"AI one-sentence verdict unavailable: {error}")


def render_fno_desk() -> None:
    st.header("High-Probability Intraday & F&O Desk")
    try:
        fno_data = _live_fno_chain()
        metrics = fno_data
        _tag("DATA", str(fno_data.get("source", "unknown")).upper(), "green" if fno_data.get("source") == "nse" else "yellow")
        metric_columns = st.columns(4)
        metric_columns[0].metric("PCR", f"{metrics['put_call_ratio']:.2f}")
        metric_columns[1].metric("Max Pain", f"{metrics['max_pain_strike']:.0f}")
        metric_columns[2].metric("Call Wall / Resistance", f"{metrics['heavy_call_wall']:.0f}")
        metric_columns[3].metric("Put Wall / Support", f"{metrics['heavy_put_wall']:.0f}")

        history = _nifty_history()
        _render_research_expander("NIFTY", history, fno_data)
        setups = scan_reversion_setups(history, fno_data) if not history.empty else pd.DataFrame()
        st.subheader("Statistical Reversion Alerts")
        if setups.empty:
            st.info("No multi-condition exhaustion setup is active at the current wall collision.")
        else:
            st.dataframe(setups, use_container_width=True, hide_index=True)
            for index, setup in setups.iterrows():
                title = f"{setup['direction']} | {setup['spread_type']} | {setup['timestamp']}"
                with st.expander(title):
                    left, right = st.columns([1, 2])
                    with left:
                        st.json({key: setup[key] for key in ("entry_strike", "hedge_strike", "net_credit", "max_risk", "break_even", "break_even_buffer")})
                        try:
                            levels = calculate_camarilla_levels(history.tail(1))
                            st.dataframe(pd.DataFrame([{"tripwire": key, "level": levels[key]} for key in ("H3", "L3", "H4", "L4")]), hide_index=True, use_container_width=True)
                        except ValueError as error:
                            st.warning(str(error))
                    with right:
                        _tradingview_widget("NIFTY")
    except Exception as error:
        st.error(f"F&O desk unavailable: {error}")


def _audit_for_stock(symbol: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    cache = st.session_state.setdefault("audit_cache", {})
    if symbol not in cache:
        cache[symbol] = generate_stock_intelligence_audit(
            symbol,
            pe=metrics.get("P/E", metrics.get("pe")),
            roe=metrics.get("ROE", metrics.get("roe")),
        )
    return cache[symbol]


def _render_scorecard(symbol: str, metrics: Mapping[str, Any]) -> None:
    columns = st.columns(4)
    columns[0].metric("Profitability", f"{float(metrics.get('cagr', metrics.get('momentum_15d', 0))):.1f}")
    columns[1].metric("Peers", f"{float(metrics.get('alpha_90d', 0)):+.1f}")
    columns[2].metric("Red Flags", "Review")
    columns[3].metric("Analyst Rating", "AI audit")
    try:
        audit = _audit_for_stock(symbol, metrics)
        st.markdown(audit.get("verdict", str(audit)) if isinstance(audit, dict) else audit)
    except Exception as error:
        st.warning(f"AI one-sentence verdict unavailable: {error}")


def render_alpha_scanners() -> None:
    st.header("Morning Digest Alpha Scanners")
    st.caption(f"One rotating batch of up to 100 names from the {len(NSE_UNIVERSE)}-stock NSE universe.")
    names = ("15-Day Momentum", "90-Day Alpha", "Long-Term Compounders")
    progress_bars = {name: st.progress(0, text=f"Queued: {name}") for name in names}
    try:
        digest = morning_digest(NSE_UNIVERSE, DATABASE_PATH)
        for index, name in enumerate(names, start=1):
            progress_bars[name].progress(index / len(names), text=f"Complete: {name}")
    except Exception as error:
        st.error(f"Morning Digest unavailable: {error}")
        return

    selected_stocks: list[dict[str, Any]] = []
    for name in names:
        result = digest.get(name, pd.DataFrame())
        st.subheader(name)
        if result.empty:
            st.info("No qualifying names returned for this batch.")
            continue
        st.dataframe(result, use_container_width=True, hide_index=True)
        selected_stocks.extend(result.to_dict("records"))

    verdicts: dict[str, str] = {}
    if selected_stocks:
        try:
            verdicts = generate_stock_verdict_batch(
                [str(stock["symbol"]) for stock in selected_stocks],
                float(_live_macro_context().get("sentiment", {}).get("sentiment_score", 0.0)),
            )
        except Exception as error:
            st.warning(f"AI batch verdict unavailable: {error}")

    for name in names:
        result = digest.get(name, pd.DataFrame())
        for row in result.to_dict("records"):
            symbol = str(row.pop("symbol"))
            history = fetch_historical_data([symbol], period="1y").get(symbol, pd.DataFrame())
            _render_research_expander(symbol, history, verdict=verdicts.get(symbol))


def render_weekly_performance_report() -> None:
    st.subheader("Automated Weekly Prediction Performance Audit")
    rows = _portfolio_rows()
    if not rows:
        st.info("No vault positions are available for the Sunday audit.")
        return
    histories = fetch_historical_data([row["symbol"] for row in rows], period="10d")
    report = []
    for row in rows:
        history = histories.get(row["symbol"], pd.DataFrame())
        close = history.get("Close", pd.Series(dtype=float))
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        weekly_return = close.pct_change(5).iloc[-1] * 100 if len(close) > 5 else None
        report.append({"symbol": row["symbol"], "weekly_return_pct": weekly_return})
    st.dataframe(pd.DataFrame(report), use_container_width=True, hide_index=True)


def render_vault() -> None:
    st.header("Persistent Vault & Sunday Performance")
    if is_sunday():
        with st.container(border=True):
            render_weekly_performance_report()

    uploaded_file = st.file_uploader("Upload Broker Holdings Screenshot", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None and st.button("Extract and sync to terminal_vault.db", type="primary"):
        try:
            extracted = ingest_broker_portfolio_screenshot(uploaded_file.getvalue())
            saved = 0
            for holding in extracted["holdings"]:
                if holding["quantity"] is None or holding["average_price"] is None:
                    continue
                upsert_portfolio_position(
                    symbol=holding["symbol"],
                    quantity=float(holding["quantity"]),
                    avg_price=float(holding["average_price"]),
                    entry_date=datetime.now().date().isoformat(),
                    stop_loss=float(holding["stop_loss"]) if holding.get("stop_loss") is not None else None,
                    database_path=DATABASE_PATH,
                )
                saved += 1
            st.success(f"Synced {saved} holdings to terminal_vault.db.")
        except Exception as error:
            st.error(f"Portfolio OCR sync failed: {error}")

    with st.form("manual_vault_position"):
        st.subheader("Add Position Manually")
        symbol = st.text_input("Symbol").strip().upper()
        quantity = st.number_input("Quantity", min_value=0.0, step=1.0)
        buy_price = st.number_input("Buy Price", min_value=0.0, step=0.05)
        stop_loss = st.number_input("Stop Loss", min_value=0.0, step=0.05)
        save_position = st.form_submit_button("Save Position", type="primary")
    if save_position:
        if not symbol or quantity <= 0 or buy_price <= 0:
            st.error("Symbol, quantity, and buy price are required.")
        else:
            upsert_portfolio_position(
                symbol=symbol,
                quantity=quantity,
                avg_price=buy_price,
                entry_date=datetime.now().date().isoformat(),
                stop_loss=stop_loss or None,
                database_path=DATABASE_PATH,
            )
            st.success(f"Saved {symbol} to the persistent vault.")
            st.rerun()

    rows = _portfolio_rows()
    if not rows:
        st.info("No active holdings found in your vault. Upload a broker screenshot above or add positions manually.")
        return
    histories = fetch_historical_data([row["symbol"] for row in rows], period="5d")
    display_rows = []
    for row in rows:
        history = histories.get(row["symbol"], pd.DataFrame())
        technicals = _research_technicals(history)
        cmp = technicals.get("cmp")
        pnl = (cmp - row["avg_price"]) * row["quantity"] if cmp is not None else None
        if cmp is None:
            action = "Unavailable"
        elif row["stop_loss"] and cmp <= row["stop_loss"]:
            action = "Cut Loss"
        elif cmp >= row["avg_price"] * 1.15:
            action = "Take Profit"
        else:
            action = "Hold"
        display_rows.append({**row, "CMP": cmp, "P&L": pnl, "Action": action})
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## NSE ALPHA / DESK")
        st.link_button("✈️ Open Telegram Alerts Desk", "https://t.me/", use_container_width=True)
        with st.expander("Telegram credentials configuration"):
            st.text_input("Bot token", type="password", key="telegram_bot_token")
            st.text_input("Chat ID", key="telegram_chat_id")
            st.caption("Credentials are kept in this session only. Use Streamlit secrets for deployment.")
        st.divider()
        st.caption("Institutional research terminal")
        st.caption(f"Universe rotation: {len(NSE_UNIVERSE)} symbols")


def main() -> None:
    st.set_page_config(page_title="NSE Alpha Terminal", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
    render_sidebar()
    st.title("NSE Alpha Terminal")
    st.caption("Macro regime, derivatives structure, statistical reversion, and fundamental research in one morning workflow.")

    tabs = st.tabs([
        "01  Market Mood Command Center",
        "02  Intraday & F&O Desk",
        "03  Morning Digest Alpha",
        "04  Persistent Vault",
    ])
    with tabs[0]:
        render_mood_command_center()
    with tabs[1]:
        render_fno_desk()
    with tabs[2]:
        render_alpha_scanners()
    with tabs[3]:
        render_vault()


if __name__ == "__main__":
    main()
