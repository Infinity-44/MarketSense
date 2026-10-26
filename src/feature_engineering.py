"""Indicators use only bars at or before the current close."""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "daily_return", "mean_return_5", "mean_return_20", "sma_10_ratio",
    "sma_20_ratio", "sma_50_ratio", "ema_12_ratio", "ema_26_ratio",
    "volatility_20", "rsi_14", "macd_ratio", "macd_signal_ratio",
    "volume_change", "volume_relative_20", "momentum_10", "intraday_range",
]
ANOMALY_COLUMNS = ["daily_return", "volatility_20", "volume_change",
                   "volume_relative_20", "momentum_10", "intraday_range"]


def engineer_features(bars: pd.DataFrame) -> pd.DataFrame:
    frame = bars.copy()
    close = frame["Close"]
    frame["daily_return"] = close.pct_change(fill_method=None)
    for window in (5, 20):
        frame[f"mean_return_{window}"] = frame["daily_return"].rolling(window).mean()
    for window in (10, 20, 50):
        frame[f"sma_{window}"] = close.rolling(window).mean()
        frame[f"sma_{window}_ratio"] = close / frame[f"sma_{window}"] - 1
    for span in (12, 26):
        frame[f"ema_{span}"] = close.ewm(span=span, adjust=False, min_periods=span).mean()
        frame[f"ema_{span}_ratio"] = close / frame[f"ema_{span}"] - 1
    frame["volatility_20"] = frame["daily_return"].rolling(20).std(ddof=1)
    change = close.diff()
    # Wilder-style recursive smoothing, seeded with the first change.
    gain = change.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    frame["rsi_14"] = rsi.mask((loss == 0) & (gain > 0), 100).mask((loss == 0) & (gain == 0), 50)
    frame["macd"] = frame["ema_12"] - frame["ema_26"]
    frame["macd_signal"] = frame["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    frame["macd_ratio"] = frame["macd"] / close
    frame["macd_signal_ratio"] = frame["macd_signal"] / close
    # A zero previous volume gives an undefined percentage change: use neutral 0
    # and retain relative volume so the observation is not discarded indefinitely.
    previous_volume = frame["Volume"].shift(1)
    frame["volume_change"] = (frame["Volume"] / previous_volume.replace(0, np.nan) - 1)
    frame.loc[previous_volume == 0, "volume_change"] = 0.0
    volume_mean = frame["Volume"].rolling(20).mean()
    frame["volume_relative_20"] = frame["Volume"] / volume_mean.replace(0, np.nan)
    frame.loc[volume_mean == 0, "volume_relative_20"] = 0.0
    frame["momentum_10"] = close.pct_change(10, fill_method=None)
    frame["intraday_range"] = (frame["High"] - frame["Low"]) / close
    return frame.replace([np.inf, -np.inf], np.nan)


def create_labels(features: pd.DataFrame) -> pd.DataFrame:
    """Keep the last target unknown. LabelDate enables boundary purging."""
    frame = features.copy()
    frame["next_return"] = frame["Close"].shift(-1) / frame["Close"] - 1
    frame["LabelDate"] = frame["Date"].shift(-1)
    frame["target"] = (frame["next_return"] > 0).astype(float)
    frame.loc[frame["next_return"].isna(), "target"] = np.nan
    return frame
