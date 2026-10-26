import numpy as np
import pandas as pd

from config.settings import Settings
from src.analysis import run_analysis
from src.feature_engineering import FEATURE_COLUMNS, engineer_features
from src.price_prediction import estimate_next_price


def test_selected_months_filter_model_rows_but_keep_past_indicator_context(ohlcv):
    result = run_analysis(ohlcv, Settings(min_observations=50), lookback_months=3)
    cutoff = ohlcv.Date.max() - pd.DateOffset(months=3)
    assert result["bars"].Date.min() >= cutoff
    assert result["split"].train.Date.min() >= cutoff
    assert result["metadata"]["source_rows_before_lookback"] == len(ohlcv)
    assert result["metadata"]["audit"]["clean_rows"] == len(ohlcv)
    assert result["features"][FEATURE_COLUMNS].notna().all().all()
    expected = engineer_features(ohlcv)
    pd.testing.assert_frame_equal(result["features"], expected.loc[expected.Date >= cutoff])


def test_price_model_selection_and_training_only_scaling(ohlcv, tmp_path):
    result = run_analysis(ohlcv, Settings(min_observations=50), lookback_months=5)
    forecast = estimate_next_price(result)
    assert forecast["selected_model"] == min(forecast["validation_return_mae"], key=forecast["validation_return_mae"].get)
    scaler = forecast["historical_models"]["Ridge regression"].named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, result["split"].train[FEATURE_COLUMNS].mean())
    assert forecast["estimated_price"] == forecast["base_close"] * (1 + forecast["estimated_return"])
    baseline = forecast["test_metrics"]["No price change"]
    expected_mae = float(np.mean(abs(result["split"].test.Close * result["split"].test.next_return)))
    np.testing.assert_allclose(baseline["price_mae"], expected_mae)
    assert forecast["test_metrics"][forecast["selected_model"]]["observations"] == len(result["split"].test)
    import joblib
    import json
    from src.analysis import save_analysis

    result["price_forecast"] = forecast
    folder = save_analysis(result, tmp_path / "price_run")
    saved = joblib.load(folder / "models.joblib")
    assert saved["deployment_price_model"].predict(result["features"].iloc[[-1]][FEATURE_COLUMNS])[0] == forecast["estimated_return"]
    report = json.loads((folder / "report.json").read_text())
    assert report["price_forecast"]["estimated_price"] == forecast["estimated_price"]
