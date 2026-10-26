"""Keyless Yahoo Finance access for NSE research. No credentials or billing.

Provider timestamps determine bar availability. Daily data and intraday data
are kept separate, and a running five-minute candle never becomes model input.
"""
from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.preprocessing import REQUIRED_COLUMNS, validate_ohlcv

EXCHANGE_TZ = "Asia/Kolkata"
BAR_MINUTES = 5


def normalize_nse_symbol(value: str) -> str:
    symbol = value.strip().upper()
    if not symbol.endswith(".NS"):
        symbol += ".NS"
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9&-]{0,25}\.NS", symbol):
        raise ValueError("Enter an NSE equity symbol such as RELIANCE, TCS, or M&M (optional .NS suffix).")
    return symbol


def utc_now(now=None) -> pd.Timestamp:
    value = pd.Timestamp(now if now is not None else datetime.now(timezone.utc))
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def _local_index(raw: pd.DataFrame) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(raw.index)
    return index.tz_localize(EXCHANGE_TZ) if index.tz is None else index.tz_convert(EXCHANGE_TZ)


def normalize_daily(raw: pd.DataFrame, now=None, session_end=None) -> tuple[pd.DataFrame, dict]:
    if raw.empty:
        raise ValueError("No daily data returned for this NSE symbol.")
    index = _local_index(raw)
    current = utc_now(now).tz_convert(EXCHANGE_TZ)
    # Keep local exchange dates, not their previous UTC calendar dates.
    frame = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    frame.insert(0, "Date", index.tz_localize(None).normalize())
    frame = frame.reset_index(drop=True)
    eligible = index.date < current.date()
    # Today is eligible only when the provider identifies the same session's
    # actual end and a 15-minute settlement buffer has elapsed. No hardcoded
    # weekday/holiday calendar or assumed special-session close is used.
    if session_end is not None:
        end = utc_now(session_end).tz_convert(EXCHANGE_TZ)
        if end.date() == current.date() and current >= end + pd.Timedelta(minutes=15):
            eligible |= index.date == current.date()
    excluded = int((~eligible).sum())
    clean, audit = validate_ohlcv(frame.loc[eligible])
    audit["unfinished_or_future_daily_rows_excluded"] = excluded
    return clean, audit


def normalize_intraday(raw: pd.DataFrame, now=None) -> tuple[pd.DataFrame, dict]:
    if raw.empty:
        raise ValueError("No intraday bars returned. Check the symbol or retry when the provider is available.")
    index = _local_index(raw)
    frame = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    frame.insert(0, "Date", index)
    frame = frame.reset_index(drop=True)
    for column in REQUIRED_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    completed = frame.Date + pd.Timedelta(minutes=BAR_MINUTES) <= utc_now(now)
    valid = (frame[REQUIRED_COLUMNS].notna().all(axis=1)
             & (frame[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
             & (frame.Volume >= 0)
             & (frame.High >= frame[["Open", "Low", "Close"]].max(axis=1))
             & (frame.Low <= frame[["Open", "High", "Close"]].min(axis=1)))
    clean = frame.loc[valid & completed].copy()
    clean = clean.drop_duplicates()
    if clean.Date.duplicated().any():
        raise ValueError("The provider returned conflicting bars at the same timestamp.")
    clean = clean.sort_values("Date").reset_index(drop=True)
    if clean.empty:
        raise ValueError("No completed five-minute bars are available yet.")
    return clean, {"input_rows": len(raw), "completed_valid_rows": len(clean),
                   "running_or_future_bars_excluded": int((~completed).sum()),
                   "invalid_rows_removed": int((~valid).sum())}


def _ticker(symbol):
    import yfinance as yf

    # Yahoo cookie/timezone cache remains local and ignored by Git.
    yf.set_tz_cache_location(str(Path(__file__).resolve().parents[1] / ".cache/yfinance"))
    return yf.Ticker(normalize_nse_symbol(symbol))


def _session_end(metadata):
    try:
        value = metadata["currentTradingPeriod"]["regular"]["end"]
        return pd.Timestamp(value, unit="s", tz="UTC") if isinstance(value, (int, float)) else utc_now(value)
    except (KeyError, TypeError, ValueError):
        return None


def fetch_daily(symbol: str) -> dict:
    symbol = normalize_nse_symbol(symbol)
    try:
        ticker = _ticker(symbol)
        raw = ticker.history(period="5y", interval="1d", auto_adjust=True,
                             actions=False, prepost=False, timeout=15)
        try:
            metadata = ticker.get_history_metadata() or {}
        except Exception:
            metadata = {}
        captured = utc_now()
        bars, audit = normalize_daily(raw, captured, _session_end(metadata))
    except Exception as exc:
        raise ValueError(f"Daily feed unavailable for {symbol}. Retry or use a local CSV; no sample data was substituted.") from exc
    return {"content": bars.to_csv(index=False, date_format="%Y-%m-%d").encode(),
            "audit": audit, "symbol": symbol, "fetched_at": captured.isoformat(),
            "currency": metadata.get("currency", "INR"),
            "source": "Yahoo Finance via yfinance", "price_basis": "auto-adjusted OHLC"}


def fetch_intraday(symbol: str) -> dict:
    symbol = normalize_nse_symbol(symbol)
    try:
        ticker = _ticker(symbol)
        raw = ticker.history(period="1mo", interval="5m", auto_adjust=True,
                             actions=False, prepost=False, timeout=15)
        captured = utc_now()
        bars, audit = normalize_intraday(raw, captured)
        try:
            metadata = ticker.get_history_metadata() or {}
        except Exception:
            metadata = {}
    except Exception as exc:
        raise ValueError(f"Intraday feed unavailable for {symbol}. Retry later; no prices or predictions were fabricated.") from exc
    end = _session_end(metadata)
    return {"bars": bars, "audit": audit, "symbol": symbol, "currency": metadata.get("currency", "INR"),
            "fetched_at": captured.isoformat(), "session_end": end.isoformat() if end is not None else None,
            "source": "Yahoo Finance via yfinance", "bar_minutes": BAR_MINUTES}
