"""Historical training-only Isolation Forest and indicator context."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config.settings import Settings
from src.feature_engineering import ANOMALY_COLUMNS


def fit_anomaly_detector(training: pd.DataFrame, settings: Settings) -> dict:
    x = training[ANOMALY_COLUMNS]
    model = IsolationForest(n_estimators=200, contamination=settings.contamination,
                            max_samples="auto", random_state=settings.seed, n_jobs=1)
    model.fit(x)
    # These statistics only describe deviation from the historical training regime.
    return {"model": model, "mean": x.mean(), "std": x.std(ddof=1).replace(0, np.nan),
            "fit_start": training.Date.min(), "fit_end": training.Date.max()}


def score_anomalies(features: pd.DataFrame, detector: dict) -> pd.DataFrame:
    frame = features.dropna(subset=ANOMALY_COLUMNS).copy()
    x = frame[ANOMALY_COLUMNS]
    frame["anomaly"] = detector["model"].predict(x) == -1
    # sklearn's decision_function is negative for outliers; invert for readable charts.
    frame["anomaly_score"] = -detector["model"].decision_function(x)
    frame["period"] = np.where(frame.Date <= detector["fit_end"],
                               "Training (in-sample)", "Later history (out-of-sample)")
    frame.loc[frame.Date < detector["fit_start"], "period"] = "Earlier history (not fitted)"
    z = (x - detector["mean"]) / detector["std"]
    explanations = []
    for idx in frame.index:
        largest = z.loc[idx].abs().dropna().nlargest(2).index
        explanations.append("; ".join(f"{col}: {z.at[idx, col]:+.1f} training SD" for col in largest)
                            or "No varying training indicators")
    frame["indicator_context"] = explanations
    return frame
