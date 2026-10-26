import io
from urllib.error import URLError

import pandas as pd
import pytest

import src.sample_data as sample
from src.data_loader import load_csv
from src.feature_engineering import engineer_features
from src.visualization import price_chart


def test_example_preserves_raw_ohlc_instead_of_mixing_adjusted_close(ohlcv):
    source = ohlcv.rename(columns={c: f"AAPL.{c}" for c in ["Open", "High", "Low", "Close", "Volume"]})
    source["AAPL.Adjusted"] = source["AAPL.Close"] * .5
    normalized = sample.normalize_example(source.to_csv(index=False).encode())
    actual, _ = load_csv(io.BytesIO(normalized))
    pd.testing.assert_series_equal(actual.Close, ohlcv.Close)


def test_download_cached_locally_and_network_failure_is_actionable(ohlcv, tmp_path, monkeypatch):
    payload = ohlcv.to_csv(index=False).encode()
    calls = []

    def fetch(*args, **kwargs):
        calls.append(1)
        return io.BytesIO(payload)

    monkeypatch.setattr(sample, "urlopen", fetch)
    first = sample.load_example(tmp_path)
    assert first == sample.load_example(tmp_path)
    assert len(calls) == 1
    assert (tmp_path / "aapl_example.source.json").exists()

    def fail(*args, **kwargs):
        raise URLError("offline")

    monkeypatch.setattr(sample, "urlopen", fail)
    with pytest.raises(ValueError, match="upload a CSV"):
        sample.load_example(tmp_path / "fresh")


def test_chart_window_keeps_anomaly_markers_within_selected_dates(ohlcv):
    features = engineer_features(ohlcv)
    visible = features.iloc[-60:]
    anomalies = features.assign(anomaly=True)
    figure = price_chart(visible, anomalies=anomalies, kind="Candlestick")
    assert figure.data[0].type == "candlestick"
    assert figure.data[1].type == "bar"
    assert len(figure.data[2].x) == len(visible)
    assert min(figure.data[2].x) == visible.Date.min()
