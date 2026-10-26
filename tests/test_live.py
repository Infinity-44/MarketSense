import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from src.live_data import normalize_daily, normalize_intraday, normalize_nse_symbol


def test_nse_symbols_and_invalid_input():
    assert normalize_nse_symbol(" reliance ") == "RELIANCE.NS"
    assert normalize_nse_symbol("m&m.ns") == "M&M.NS"
    with pytest.raises(ValueError):
        normalize_nse_symbol("../../.env")


def test_daily_exchange_dates_and_partial_session_exclusion():
    raw = pd.DataFrame({"Open": [99, 100, 101], "High": [102, 102, 103], "Low": [98, 99, 100],
                        "Close": [100, 101, 102], "Volume": [900, 1000, 200]},
                       index=pd.DatetimeIndex(["2026-10-22", "2026-10-23", "2026-10-26"], tz="Asia/Kolkata", name="Date"))
    bars, audit = normalize_daily(raw, "2026-10-26T10:00:00+05:30", "2026-10-26T15:30:00+05:30")
    assert bars.Date.iloc[-1] == pd.Timestamp("2026-10-23")
    assert audit["unfinished_or_future_daily_rows_excluded"] == 1
    after, _ = normalize_daily(raw, "2026-10-26T16:00:00+05:30", "2026-10-26T15:30:00+05:30")
    assert after.Date.iloc[-1] == pd.Timestamp("2026-10-26")
    unknown, _ = normalize_daily(raw, "2026-10-26T16:00:00+05:30")
    assert len(unknown) == 2


def test_running_intraday_bar_is_never_model_input(intraday):
    raw = intraday.iloc[:2].set_index("Date")
    now = raw.index[0] + pd.Timedelta(minutes=7)
    clean, audit = normalize_intraday(raw, now)
    assert len(clean) == 1
    assert audit["running_or_future_bars_excluded"] == 1
    assert str(clean.Date.dt.tz) == "Asia/Kolkata"
