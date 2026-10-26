import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.analysis import run_analysis, save_analysis
from src.feature_engineering import FEATURE_COLUMNS

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(autouse=True)
def offline_live_providers(ohlcv, intraday, monkeypatch):
    import src.live_dashboard as dashboard

    dashboard.cached_daily.clear()
    dashboard.cached_intraday.clear()
    monkeypatch.setattr(dashboard, "fetch_daily", lambda symbol: {
        "content": ohlcv.to_csv(index=False).encode(), "fetched_at": "2026-10-26T12:00:00+00:00",
        "audit": {}, "currency": "INR"})
    monkeypatch.setattr(dashboard, "fetch_intraday", lambda symbol: {
        "bars": intraday, "fetched_at": "2026-10-26T12:00:00+00:00", "currency": "INR",
        "session_end": "2026-10-26T15:30:00+05:30", "audit": {"running_or_future_bars_excluded": 0}})
    yield
    dashboard.cached_daily.clear()
    dashboard.cached_intraday.clear()


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


def test_dashboard_empty_state(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(Path, "glob", lambda self, pattern: [])
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    app.sidebar.selectbox[0].set_value("Upload / local CSV").run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Overview"
    assert not app.get("plotly_chart")


def test_dashboard_every_section_with_explicit_test_fixture(ohlcv, monkeypatch):
    import src.sample_data as sample
    from streamlit.testing.v1 import AppTest

    # Replace the data provider inside this test process only; never publish fixture scores.
    monkeypatch.setattr(sample, "load_example", lambda directory: ohlcv.to_csv(index=False).encode())
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    app.sidebar.selectbox[0].set_value("Apple · AAPL (historical example)").run(timeout=30)
    assert not app.exception
    assert any(m.label == "Next-session price estimate" for m in app.metric)
    next(w for w in app.radio if w.label == "Historical chart").set_value("Candlestick").run(timeout=30)
    assert not app.exception
    app.sidebar.radio[0].set_value("Market Behaviour").run(timeout=30)
    next(w for w in app.radio if w.label == "Chart window").set_value("3M").run(timeout=30)
    assert not app.exception
    for section in ["Market Behaviour", "Risk Analysis", "Anomaly Detection", "Model Evaluation"]:
        app.sidebar.radio[0].set_value(section).run(timeout=30)
        assert not app.exception, section


def test_dashboard_reports_download_failure(monkeypatch):
    import src.sample_data as sample
    from streamlit.testing.v1 import AppTest

    def fail(directory):
        raise ValueError("Example unavailable; upload a CSV.")

    monkeypatch.setattr(sample, "load_example", fail)
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    app.sidebar.selectbox[0].set_value("Apple · AAPL (historical example)").run(timeout=30)
    assert not app.exception
    assert "Example unavailable" in app.error[0].value


def test_nse_live_workspace_and_daily_analysis_are_available():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not app.exception
    assert app.sidebar.selectbox[0].value == "NSE · live + historical"
    assert any("Next-session direction estimate" == metric.label for metric in app.metric)
    assert any("Next-session price estimate" == metric.label for metric in app.metric)
    app.sidebar.radio[0].set_value("Price & Direction").run(timeout=30)
    assert not app.exception
    app.sidebar.radio[0].set_value("Risk Analysis").run(timeout=30)
    assert not app.exception


def test_live_feed_failure_does_not_fall_back_to_example(monkeypatch):
    import src.live_dashboard as dashboard
    from streamlit.testing.v1 import AppTest

    def fail(symbol):
        raise ValueError("Provider unavailable; retry.")

    monkeypatch.setattr(dashboard, "fetch_daily", fail)
    monkeypatch.setattr(dashboard, "fetch_intraday", fail)
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not app.exception
    assert len(app.error) == 1
    assert app.warning
    assert not app.metric


def test_forecast_consistency_across_pages_and_refresh_control(monkeypatch, intraday):
    import src.live_dashboard as dashboard
    from streamlit.testing.v1 import AppTest

    calls = []

    def quote(symbol):
        calls.append(symbol)
        return {"bars": intraday, "fetched_at": "2026-10-26T12:00:00Z"}

    monkeypatch.setattr(dashboard, "fetch_intraday", quote)
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    overview_direction = next(m.value for m in app.metric if m.label == "Next-session direction estimate")
    assert len(calls) == 1
    next(w for w in app.toggle if w.label.startswith("Auto-refresh")).set_value(False).run(timeout=30)
    assert len(calls) == 1
    next(w for w in app.button if w.label == "Refresh price").click().run(timeout=30)
    assert len(calls) == 2
    app.sidebar.radio[0].set_value("Market Behaviour").run(timeout=30)
    assert not app.exception
    assert app.metric[0].value == overview_direction


def test_quote_failure_keeps_daily_forecast_explicitly_labelled(monkeypatch):
    import src.live_dashboard as dashboard
    from streamlit.testing.v1 import AppTest

    def fail(symbol):
        raise ValueError("Quote request unavailable")

    monkeypatch.setattr(dashboard, "fetch_intraday", fail)
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not app.exception
    assert app.warning
    assert any(m.label == "Last completed daily close" for m in app.metric)
    assert any(m.label == "Next-session price estimate" for m in app.metric)


def test_invalid_risk_settings_produce_actionable_error():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(APP_PATH).run(timeout=30)
    next(w for w in app.number_input if w.label.startswith("Low / moderate")).set_value(40.0).run(timeout=30)
    assert not app.exception
    assert "ordered" in app.error[0].value
