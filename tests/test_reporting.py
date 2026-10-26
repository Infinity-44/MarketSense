import io
import json
import zipfile

import pandas as pd

from src.analysis import run_analysis
from src.reporting import analysis_bundle, model_assessment, price_unit, research_report


def test_research_bundle_has_real_holdout_rows_and_serializable_summary(ohlcv):
    result = run_analysis(ohlcv, symbol="TEST FIXTURE")
    result["metadata"]["dataset_source"] = {"kind": "test-only fixture", "currency": "INR"}
    report = research_report(result)
    assert json.loads(json.dumps(report, allow_nan=False))["metadata"]["data_rows"] == len(ohlcv)
    with zipfile.ZipFile(io.BytesIO(analysis_bundle(result))) as archive:
        assert set(archive.namelist()) == {"report.json", "summary.md", "daily_observations.csv", "risk.csv",
                                           "direction_test_predictions.csv", "price_test_predictions.csv", "anomalies.csv"}
        predictions = pd.read_csv(archive.open("price_test_predictions.csv"))
        assert len(predictions) == len(result["split"].test)
        actual = pd.to_datetime(predictions.LabelDate)
        assert actual.min() == result["split"].test.LabelDate.min()
        assert result["metadata"]["cleaned_data_sha256"] in archive.read("summary.md").decode()
    prefix, unit = price_unit(result)
    assert (prefix, unit) == ("₹", "INR")


def test_baseline_quality_is_not_misrepresented_as_improvement(ohlcv):
    result = run_analysis(ohlcv)
    result["prediction"]["selected"] = "Majority baseline"
    result["price_forecast"]["selected_model"] = "No price change"
    quality = model_assessment(result)
    assert quality["direction_status"] == quality["price_status"] == "Baseline selected"
    assert quality["direction_f1_delta"] == quality["price_error_reduction"] == 0
    assert price_unit(result) == ("", "source units")
