"""Historical next-session return regression, including a no-change baseline."""
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import FEATURE_COLUMNS, create_labels


def estimate_next_price(result: dict) -> dict:
    split = result["split"]
    models = {
        "No price change": DummyRegressor(strategy="constant", constant=0.0),
        "Ridge regression": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10))]),
        "Random Forest regression": RandomForestRegressor(n_estimators=200, max_depth=3,
            min_samples_leaf=5, random_state=result["metadata"]["settings"]["seed"], n_jobs=1),
    }
    validation = {}
    for name, model in models.items():
        model.fit(split.train[FEATURE_COLUMNS], split.train.next_return)
        validation[name] = float(mean_absolute_error(split.validation.next_return,
                                                     model.predict(split.validation[FEATURE_COLUMNS])))
    selected = min(validation, key=validation.get)
    test, predictions = {}, {}
    for name, model in models.items():
        predicted_return = model.predict(split.test[FEATURE_COLUMNS])
        table = split.test[["Date", "LabelDate", "Close", "next_return"]].copy()
        table["actual_close"] = table.Close * (1 + table.next_return)
        table["predicted_close"] = table.Close * (1 + predicted_return)
        table["predicted_return"] = predicted_return
        table["absolute_error"] = abs(table.actual_close - table.predicted_close)
        predictions[name] = table
        test[name] = {
            "return_mae": float(mean_absolute_error(split.test.next_return, predicted_return)),
            "price_mae": float(mean_absolute_error(split.test.Close * (1 + split.test.next_return),
                                                    split.test.Close * (1 + predicted_return))),
            "observations": len(split.test),
        }
    # Historical scores remain attached to the train-only models. Deployment is
    # a distinct clone refitted only after the comparison has been finalized.
    historical = create_labels(result["features"]).dropna(subset=FEATURE_COLUMNS + ["next_return"])
    deployed = clone(models[selected]).fit(historical[FEATURE_COLUMNS], historical.next_return)
    latest = result["features"].iloc[[-1]]
    estimated_return = float(deployed.predict(latest[FEATURE_COLUMNS])[0])
    return {"selected_model": selected, "validation_return_mae": validation, "test_metrics": test,
            "test_predictions": predictions,
            "deployment_model": deployed, "historical_models": models,
            "estimated_return": estimated_return, "estimated_price": float(latest.Close.iloc[0] * (1 + estimated_return)),
            "base_close": float(latest.Close.iloc[0]), "input_date": latest.Date.iloc[0].isoformat(),
            "horizon": "next trading session after input_date", "fit_label_end": historical.LabelDate.max().isoformat()}


def price_summary(forecast: dict) -> dict:
    """JSON-safe forecast fields. Model objects and tables are exported separately."""
    return {key: value for key, value in forecast.items()
            if key not in ("deployment_model", "historical_models", "test_predictions")}
