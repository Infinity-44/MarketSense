import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.analysis import run_analysis, save_analysis
from src.feature_engineering import FEATURE_COLUMNS

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_full_pipeline_artifact_roundtrip_and_no_overwrite(ohlcv, tmp_path):
    result = run_analysis(ohlcv, symbol="UNIT TEST FIXTURE")
    output = save_analysis(result, tmp_path / "run")
    report = json.loads((output / "report.json").read_text())
    assert report["metadata"]["data_rows"] == len(ohlcv)
    assert report["metadata"]["selected_model"] == result["prediction"]["selected"]
    restored = joblib.load(output / "models.joblib")  # only load our own trusted test artifact
    name = result["prediction"]["selected"]
    x = result["split"].test[FEATURE_COLUMNS]
    np.testing.assert_array_equal(restored["models"][name].predict(x),
                                  result["prediction"]["models"][name].predict(x))
    with pytest.raises(FileExistsError):
        save_analysis(result, output)


def test_dashboard_empty_state():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not app.exception
    # With local data this exercises the overview; otherwise the honest empty state.
    assert app.title[0].value == "Overview"


def test_dashboard_every_section_with_explicit_test_fixture(ohlcv, monkeypatch):
    import app as dashboard
    from streamlit.testing.v1 import AppTest

    # Inject fixture only in test process. Never save synthetic bars to data/raw.
    monkeypatch.setattr(dashboard.Path, "glob", lambda self, pattern: [self / "fixture.csv"] if pattern == "*.csv" else [])
    original_read_bytes = dashboard.Path.read_bytes

    def read_bytes(path):
        if path.name == "fixture.csv":
            return ohlcv.to_csv(index=False).encode()
        return original_read_bytes(path)

    monkeypatch.setattr(dashboard.Path, "read_bytes", read_bytes)
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not app.exception
    for section in ["Market Behaviour", "Risk Analysis", "Anomaly Detection", "Model Evaluation"]:
        app.sidebar.radio[0].set_value(section).run(timeout=30)
        assert not app.exception, section
