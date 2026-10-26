"""Descriptive historical risk; no forecast or trading recommendation."""
import numpy as np
import pandas as pd

from config.settings import Settings


def risk_label(annualized_volatility: float, settings: Settings) -> str:
    if not np.isfinite(annualized_volatility):
        return "Insufficient history"
    if annualized_volatility < settings.low_risk_limit:
        return "Low Risk"
    if annualized_volatility < settings.high_risk_limit:
        return "Moderate Risk"
    return "High Risk"


def analyze_risk(bars: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, dict]:
    frame = bars[["Date", "Close"]].copy()
    returns = frame["Close"].pct_change(fill_method=None)
    sample = returns.dropna()
    annualizer = np.sqrt(settings.trading_days)
    rolling = returns.rolling(settings.risk_window)
    frame["daily_return"] = returns
    frame["annualized_volatility"] = rolling.std(ddof=1) * annualizer
    frame["drawdown"] = frame["Close"] / frame["Close"].cummax() - 1
    # Positive loss convention; floor at zero if all quantile returns are positive.
    frame["var_loss"] = (-rolling.quantile(1 - settings.var_confidence)).clip(lower=0)
    # Downside deviation about a zero target, denominator includes all returns.
    frame["downside_volatility"] = returns.clip(upper=0).pow(2).rolling(
        settings.risk_window).mean().pow(0.5) * annualizer
    frame["risk_class"] = frame["annualized_volatility"].map(lambda v: risk_label(v, settings))
    volatility = float(sample.std(ddof=1))
    summary = {
        "historical_daily_volatility": volatility,
        "annualized_volatility": volatility * annualizer,
        "maximum_drawdown": float(frame["drawdown"].min()),
        "historical_var_loss": max(0.0, float(-sample.quantile(1 - settings.var_confidence))),
        "downside_volatility": float(np.sqrt(np.mean(np.minimum(sample, 0) ** 2)) * annualizer),
        "latest_rolling_volatility": float(frame["annualized_volatility"].iloc[-1]),
        "risk_class": frame["risk_class"].iloc[-1],
        "return_observations": len(sample),
    }
    return frame, summary
