"""Synthetic bars are unit-test fixtures only, never app or benchmark data."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ohlcv():
    rng = np.random.default_rng(16)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, 440)))
    opening = close * np.exp(rng.normal(0, 0.003, len(close)))
    return pd.DataFrame({
        "Date": pd.bdate_range("2020-01-01", periods=len(close)), "Open": opening,
        "High": np.maximum(opening, close) * 1.01,
        "Low": np.minimum(opening, close) * 0.99, "Close": close,
        "Volume": rng.integers(10000, 100000, len(close)),
    })
