import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from src.anomaly_detection import fit_anomaly_detector, score_anomalies
from src.feature_engineering import ANOMALY_COLUMNS, create_labels, engineer_features
from src.market_prediction import chronological_split
from src.risk_analysis import analyze_risk, risk_label


def test_known_drawdown_and_var():
    bars = pd.DataFrame({"Date": pd.bdate_range("2020-01-01", periods=4), "Close": [100, 120, 90, 108]})
    _, summary = analyze_risk(bars, Settings(risk_window=2))
    assert summary["maximum_drawdown"] == pytest.approx(-0.25)
    assert summary["historical_var_loss"] == pytest.approx(-np.quantile([0.2, -0.25, 0.2], 0.05))
    assert summary["downside_volatility"] == pytest.approx(np.sqrt(0.25 ** 2 / 3) * np.sqrt(252))


def test_risk_threshold_boundaries_and_configuration():
    settings = Settings()
    assert risk_label(0.149, settings) == "Low Risk"
    assert risk_label(0.15, settings) == "Moderate Risk"
    assert risk_label(0.30, settings) == "High Risk"
    assert risk_label(float("nan"), settings) == "Insufficient history"
    with pytest.raises(ValueError):
        Settings(low_risk_limit=0.5, high_risk_limit=0.2)


def test_training_only_anomalies_and_future_shock(ohlcv):
    features = engineer_features(ohlcv)
    split = chronological_split(create_labels(features), Settings())
    detector = fit_anomaly_detector(split.train, Settings())
    assert detector["fit_end"] == split.train.Date.max()
    pd.testing.assert_series_equal(detector["mean"], split.train[ANOMALY_COLUMNS].mean())
    shocked = features.copy()
    shocked.loc[shocked.index[-1], ANOMALY_COLUMNS] = [1, 0.8, 50, 50, 5, 0.8]
    scored = score_anomalies(shocked, detector)
    assert scored.iloc[-1].anomaly
    assert scored.iloc[-1].anomaly_score > 0
    assert scored.iloc[-1].period == "Later history (out-of-sample)"
    assert "training SD" in scored.iloc[-1].indicator_context
