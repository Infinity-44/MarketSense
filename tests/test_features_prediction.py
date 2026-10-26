import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from src.feature_engineering import FEATURE_COLUMNS, create_labels, engineer_features
from src.market_prediction import chronological_split, train_and_evaluate


def test_future_changes_cannot_change_past_features(ohlcv):
    original = engineer_features(ohlcv)
    changed = ohlcv.copy()
    changed.loc[300:, ["Open", "High", "Low", "Close", "Volume"]] *= 3
    pd.testing.assert_frame_equal(original.iloc[:300], engineer_features(changed).iloc[:300])


def test_latest_label_is_unknown_and_boundary_labels_are_purged(ohlcv):
    labelled = create_labels(engineer_features(ohlcv))
    assert pd.isna(labelled.iloc[-1].target)
    assert labelled.iloc[-2].target == int(ohlcv.iloc[-1].Close > ohlcv.iloc[-2].Close)
    split = chronological_split(labelled, Settings())
    assert split.train.LabelDate.max() < split.validation.Date.min()
    assert split.validation.LabelDate.max() < split.test.Date.min()
    assert split.purged_rows == 2


def test_flat_prices_and_zero_volume_are_finite(ohlcv):
    features = engineer_features(ohlcv.assign(Open=100, High=100, Low=100, Close=100, Volume=0))
    assert features.rsi_14.iloc[-1] == 50
    assert np.isfinite(features[FEATURE_COLUMNS].iloc[50:]).all().all()


def test_scaler_uses_training_only_and_selection_uses_validation(ohlcv):
    split = chronological_split(create_labels(engineer_features(ohlcv)), Settings())
    run = train_and_evaluate(split, Settings())
    scaler = run["models"]["Logistic Regression"].named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, split.train[FEATURE_COLUMNS].mean())
    expected = max(run["validation_metrics"], key=lambda n: run["validation_metrics"][n]["f1_macro"])
    assert run["selected"] == expected
    for name, metrics in run["test_metrics"].items():
        assert metrics["n"] == len(split.test)
        assert np.asarray(metrics["confusion_matrix"]).sum() == len(split.test)
        assert run["test_predictions"][name].Date.min() == split.test.Date.min()


def test_short_history_rejected(ohlcv):
    with pytest.raises(ValueError, match="at least"):
        chronological_split(create_labels(engineer_features(ohlcv.iloc[:100])), Settings())
