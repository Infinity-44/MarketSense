import io

import pandas as pd
import pytest

from src.data_loader import load_csv
from src.preprocessing import validate_ohlcv


def bars():
    return pd.DataFrame({"Date": ["2020-01-02", "2020-01-01"], "Open": [10, 11],
                         "High": [12, 12], "Low": [9, 9], "Close": [11, 10], "Volume": [20, 0]})


def test_clean_sort_duplicate_and_missing_audit():
    raw = pd.concat([bars(), bars().iloc[[0]], bars().iloc[[0]].assign(Close=None)])
    clean, audit = validate_ohlcv(raw)
    assert clean.Date.is_monotonic_increasing
    assert len(clean) == 2
    assert audit["exact_duplicates_removed"] == 1
    assert audit["missing_or_nonfinite_rows_removed"] == 1


def test_conflicting_dates_rejected():
    with pytest.raises(ValueError, match="Conflicting"):
        validate_ohlcv(pd.concat([bars(), bars().iloc[[0]].assign(Close=10)]))


def test_schema_and_multistock_rejected():
    with pytest.raises(ValueError, match="Missing"):
        load_csv(io.StringIO("Date,Close\n2020-01-01,2"))
    with pytest.raises(ValueError, match="one stock"):
        validate_ohlcv(bars().assign(Symbol=["A", "B"]))


def test_invalid_ohlc_and_negative_volume_removed():
    raw = pd.concat([bars(), bars().assign(Volume=-1), bars().assign(High=1)])
    clean, audit = validate_ohlcv(raw)
    assert len(clean) == 2
    assert audit["invalid_price_or_volume_rows_removed"] == 4
