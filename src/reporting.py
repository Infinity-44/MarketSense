"""Consistent forecast explanations and portable, data-backed research reports."""
import io
import json
import zipfile

import pandas as pd

from src.price_prediction import price_summary


def price_unit(result: dict) -> tuple[str, str]:
    currency = result["metadata"].get("dataset_source", {}).get("currency")
    return {"INR": ("₹", "INR"), "USD": ("$", "USD")}.get(currency, ("", "source units"))


def model_assessment(result: dict) -> dict:
    prediction = result["prediction"]
    selected = prediction["selected"]
    direction = prediction["test_metrics"][selected]
    baseline = prediction["test_metrics"]["Majority baseline"]
    forecast = result["price_forecast"]
    price = forecast["test_metrics"][forecast["selected_model"]]
    price_baseline = forecast["test_metrics"]["No price change"]
    f1_delta = direction["f1_macro"] - baseline["f1_macro"]
    error_reduction = price_baseline["price_mae"] - price["price_mae"]
    return {
        "direction_status": "Baseline selected" if selected == "Majority baseline" else
                            ("Higher test F1" if f1_delta > 1e-12 else "No test F1 improvement"),
        "price_status": "Baseline selected" if forecast["selected_model"] == "No price change" else
                        ("Lower test error" if error_reduction > 1e-12 else "No test error improvement"),
        "direction_f1_delta": float(f1_delta), "price_error_reduction": float(error_reduction),
        "test_observations": int(direction["n"]), "small_holdout": direction["n"] < 30,
        "note": "Descriptive held-out comparison, not proof of statistical significance or future accuracy.",
    }


def research_report(result: dict) -> dict:
    return {"metadata": result["metadata"], "forecast_horizon": "Next trading session after the daily input date",
            "direction_forecast": result["daily_deployment"],
            "price_forecast": price_summary(result["price_forecast"]),
            "historical_direction_validation": result["prediction"]["validation_metrics"],
            "historical_direction_test": result["prediction"]["test_metrics"],
            "risk": result["risk_summary"], "model_assessment": model_assessment(result),
            "interpretation": "Frozen holdout models provide the historical scores; separate deployment clones provide the unknown next-session estimates."}


def markdown_summary(result: dict) -> str:
    prefix, unit = price_unit(result)
    metadata, forecast = result["metadata"], result["price_forecast"]
    prediction = result["prediction"]
    test = prediction["test_metrics"][prediction["selected"]]
    price_test = forecast["test_metrics"][forecast["selected_model"]]
    quality = model_assessment(result)
    return f"""# MarketSense research report — {metadata['symbol']}

Generated: {metadata['created_utc']}

## Data and forecast

- Source: {metadata.get('dataset_source', {}).get('kind', 'local OHLCV data')}
- Daily observations: {metadata['data_start'][:10]} to {metadata['data_end'][:10]} ({metadata['data_rows']} sessions)
- Modelling window: {str(metadata['lookback_months']) + ' months' if metadata['lookback_months'] else 'full supplied history'}
- Next-session closing-price estimate: {prefix}{forecast['estimated_price']:,.2f} ({unit})
- Forecast input close: {prefix}{forecast['base_close']:,.2f}, dated {forecast['input_date'][:10]}
- Direction estimate: {result['daily_deployment']['direction']}
- Price model: {forecast['selected_model']}; direction model: {prediction['selected']}

## Historical evaluation

- Held-out direction accuracy: {test['accuracy']:.1%}; macro F1: {test['f1_macro']:.3f}
- Price mean absolute error: {prefix}{price_test['price_mae']:.2f} ({unit})
- Held-out observations: {test['n']}
- Price comparison: {quality['price_status']}; direction comparison: {quality['direction_status']}
- Current historical risk category: {result['risk_summary']['risk_class']}

Selection uses chronological validation; test data does not choose the models. Earlier observations initialize indicators only when a lookback is selected. Boundary labels are purged and preprocessing fits training only. Separate deployment clones refit resolved labels after evaluation. Their next-session outcomes are unknown.

Historical error is not a future prediction interval. Model probabilities are uncalibrated. Short, dependent holdouts limit conclusions. Quotes may be delayed and are separate from completed daily forecast inputs. This is an educational analysis, with no trade execution or demonstrated trading profitability.

Reproducibility hash: `{metadata['cleaned_data_sha256']}`
"""


def analysis_bundle(result: dict) -> bytes:
    """A shareable local report bundle without executable model/pickle files."""
    stream = io.BytesIO()
    selected = result["prediction"]["selected"]
    price_selected = result["price_forecast"]["selected_model"]
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("report.json", json.dumps(research_report(result), indent=2, allow_nan=False))
        archive.writestr("summary.md", markdown_summary(result))
        for name, table in {
            "daily_observations.csv": result["bars"], "risk.csv": result["risk_frame"],
            "direction_test_predictions.csv": result["prediction"]["test_predictions"][selected],
            "price_test_predictions.csv": result["price_forecast"]["test_predictions"][price_selected],
            "anomalies.csv": result["anomalies"][["Date", "Close", "anomaly", "anomaly_score", "period", "indicator_context"]],
        }.items():
            archive.writestr(name, table.to_csv(index=False))
    return stream.getvalue()
