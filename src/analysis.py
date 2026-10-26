"""Shared orchestration for CLI and dashboard. No Streamlit dependencies."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn

from config.settings import Settings
from src.anomaly_detection import fit_anomaly_detector, score_anomalies
from src.evaluation import CLASS_NAMES
from src.feature_engineering import FEATURE_COLUMNS, create_labels, engineer_features
from src.market_prediction import chronological_split, train_and_evaluate, refit_direction
from src.price_prediction import estimate_next_price, price_summary
from src.preprocessing import validate_ohlcv
from src.risk_analysis import analyze_risk


def run_analysis(bars: pd.DataFrame, settings: Settings | None = None,
                 symbol: str = "Uploaded stock", audit: dict | None = None,
                 lookback_months: int | None = None) -> dict:
    settings = settings or Settings()
    bars, verification = validate_ohlcv(bars)
    audit = audit or verification
    features = engineer_features(bars)
    source_rows = len(bars)
    if lookback_months is not None:
        if lookback_months not in (3, 5, 6, 12):
            raise ValueError("Choose a 3, 5, 6 or 12 month history window.")
        cutoff = bars.Date.max() - pd.DateOffset(months=lookback_months)
        # Compute causal indicator context before selecting the modelling window,
        # so a 50-bar moving average does not consume the entire 3-month sample.
        features = features.loc[features.Date >= cutoff].copy()
        bars = bars.loc[bars.Date >= cutoff].copy()
    if settings.risk_window > len(bars) - 1:
        raise ValueError(f"The {settings.risk_window}-observation risk window exceeds the available daily returns. "
                         "Choose a shorter risk window or a longer historical lookback.")
    split = chronological_split(create_labels(features), settings)
    prediction = train_and_evaluate(split, settings)
    risk_frame, risk_summary = analyze_risk(bars, settings)
    detector = fit_anomaly_detector(split.train, settings)
    anomalies = score_anomalies(features, detector)
    latest = features.iloc[[-1]]
    if latest[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Latest observation has incomplete indicators; inspect the uploaded data.")
    selected_model = prediction["models"][prediction["selected"]]
    direction = int(selected_model.predict(latest[FEATURE_COLUMNS])[0])
    latest_inference = {
        "as_of": latest.Date.iloc[0].isoformat(), "direction": CLASS_NAMES[direction],
        "bullish_probability": float(selected_model.predict_proba(latest[FEATURE_COLUMNS])[0, 1]),
        "note": "Next supplied trading observation; outcome unknown. Training-period model, no test refit.",
    }
    periods = {}
    for name in ("train", "validation", "test"):
        part = getattr(split, name)
        periods[name] = {"start": part.Date.min().isoformat(), "end": part.Date.max().isoformat(),
                         "last_label_date": part.LabelDate.max().isoformat(), "rows": len(part)}
    csv = bars.to_csv(index=False, date_format="%Y-%m-%d").encode()
    metadata = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol, "cleaned_data_sha256": hashlib.sha256(csv).hexdigest(),
        "data_start": bars.Date.min().isoformat(), "data_end": bars.Date.max().isoformat(),
        "data_rows": len(bars), "audit": audit, "settings": settings.to_dict(),
        "lookback_months": lookback_months, "source_rows_before_lookback": source_rows,
        "feature_columns": FEATURE_COLUMNS, "class_names": CLASS_NAMES,
        "target": "1 if next observed trading close / current close - 1 > 0, else 0",
        "split_periods": periods, "purged_rows": split.purged_rows,
        "selection_metric": "validation f1_macro; baseline eligible; insertion order breaks ties",
        "selected_model": prediction["selected"],
        "anomaly_fit_end": detector["fit_end"].isoformat(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                     "joblib": joblib.__version__},
    }
    result = {"bars": bars, "features": features, "split": split, "prediction": prediction,
            "risk_frame": risk_frame, "risk_summary": risk_summary, "detector": detector,
            "anomalies": anomalies, "latest_inference": latest_inference, "metadata": metadata}
    result["daily_deployment"] = refit_direction(result)
    result["price_forecast"] = estimate_next_price(result)
    return result


def save_analysis(result: dict, output: str | Path) -> Path:
    """Save an auditable run to a NEW directory; existing runs are never overwritten."""
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=False)
    prediction = result["prediction"]
    artifact = {"metadata": result["metadata"], "models": prediction["models"],
                "detector": result["detector"]}
    if "daily_deployment_model" in result:
        artifact["deployment_direction_model"] = result["daily_deployment_model"]
    if "price_forecast" in result:
        artifact["deployment_price_model"] = result["price_forecast"]["deployment_model"]
        artifact["historical_price_models"] = result["price_forecast"]["historical_models"]
    joblib.dump(artifact, folder / "models.joblib")
    report = {"metadata": result["metadata"], "validation": prediction["validation_metrics"],
              "test": prediction["test_metrics"], "risk": result["risk_summary"],
              "latest_inference": result["latest_inference"]}
    if "price_forecast" in result:
        report["price_forecast"] = price_summary(result["price_forecast"])
    if "daily_deployment" in result:
        report["deployment_direction_forecast"] = result["daily_deployment"]
    (folder / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    for name, table in prediction["test_predictions"].items():
        table.to_csv(folder / f"test_{name.lower().replace(' ', '_')}.csv", index=False)
    if "price_forecast" in result:
        for name, table in result["price_forecast"]["test_predictions"].items():
            table.to_csv(folder / f"price_test_{name.lower().replace(' ', '_')}.csv", index=False)
    result["features"].to_csv(folder / "features.csv", index=False)
    result["risk_frame"].to_csv(folder / "risk.csv", index=False)
    columns = ["Date", "Close", "anomaly", "anomaly_score", "period", "indicator_context"]
    result["anomalies"][columns].to_csv(folder / "anomalies.csv", index=False)
    if "price_forecast" in result:
        from src.reporting import analysis_bundle, markdown_summary

        (folder / "summary.md").write_text(markdown_summary(result), encoding="utf-8")
        (folder / "research_bundle.zip").write_bytes(analysis_bundle(result))
    return folder
