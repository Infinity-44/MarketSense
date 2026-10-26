"""Fixed model comparison, purged holdouts, and training-only pipelines."""
from dataclasses import dataclass
import logging

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from config.settings import Settings
from src.evaluation import classification_metrics
from src.feature_engineering import FEATURE_COLUMNS, create_labels

logger = logging.getLogger(__name__)


def refit_direction(result: dict) -> dict:
    """Refit a deployment clone after evaluation without replacing holdout models."""
    labelled = create_labels(result["features"]).dropna(subset=FEATURE_COLUMNS + ["target", "LabelDate"])
    model = clone(result["prediction"]["models"][result["prediction"]["selected"]])
    model.fit(labelled[FEATURE_COLUMNS], labelled.target.astype(int))
    result["daily_deployment_model"] = model
    latest = result["features"].iloc[[-1]]
    return {"direction": "Bullish" if model.predict(latest[FEATURE_COLUMNS])[0] else "Bearish",
            "bullish_probability": float(model.predict_proba(latest[FEATURE_COLUMNS])[0, 1]),
            "as_of": latest.Date.iloc[0].isoformat(), "training_label_end": labelled.LabelDate.max().isoformat(),
            "note": "Deployment clone refitted on resolved historical labels after evaluation; next-session outcome unknown."}


@dataclass
class ChronologicalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    purged_rows: int


def chronological_split(labelled: pd.DataFrame, settings: Settings,
                        feature_columns: list | None = None) -> ChronologicalSplit:
    feature_columns = FEATURE_COLUMNS if feature_columns is None else feature_columns
    frame = labelled.dropna(subset=feature_columns + ["target", "LabelDate"]).copy()
    if not frame["Date"].is_monotonic_increasing or frame["Date"].duplicated().any():
        raise ValueError("Features must have unique, chronologically sorted dates.")
    n = len(frame)
    if n < settings.min_observations:
        raise ValueError(f"Need at least {settings.min_observations} labelled rows after indicator warm-up; "
                         f"found {n}. Choose a longer lookback or supply more daily history.")
    a = int(n * settings.train_fraction)
    b = int(n * (settings.train_fraction + settings.validation_fraction))
    if min(a, b - a, n - b) < 10:
        raise ValueError("Each chronological partition must contain at least 10 rows.")
    train, validation, test = frame.iloc[:a], frame.iloc[a:b], frame.iloc[b:]
    train = train.loc[train["LabelDate"] < validation["Date"].iloc[0]].copy()
    validation = validation.loc[validation["LabelDate"] < test["Date"].iloc[0]].copy()
    if train["target"].nunique() < 2:
        raise ValueError("Training period needs both Bullish and Bearish labels.")
    return ChronologicalSplit(train, validation, test, n - len(train) - len(validation) - len(test))


def model_candidates(seed: int) -> dict:
    return {
        "Majority baseline": Pipeline([("model", DummyClassifier(strategy="most_frequent"))]),
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=seed)),
        ]),
        "Random Forest": Pipeline([("model", RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=10, random_state=seed, n_jobs=1))]),
        "Gradient Boosting": Pipeline([("model", GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=2,
            min_samples_leaf=10, random_state=seed))]),
    }


def train_and_evaluate(split: ChronologicalSplit, settings: Settings,
                       feature_columns: list | None = None) -> dict:
    feature_columns = FEATURE_COLUMNS if feature_columns is None else feature_columns
    models = model_candidates(settings.seed)
    validation_metrics = {}
    for name, model in models.items():
        logger.info("Training %s on %s bars", name, len(split.train))
        model.fit(split.train[feature_columns], split.train["target"].astype(int))
        validation_metrics[name] = classification_metrics(
            split.validation["target"], model.predict(split.validation[feature_columns]))
    # Baseline participates: if it wins, show that honestly in inference.
    selected = max(models, key=lambda name: validation_metrics[name]["f1_macro"])
    test_metrics, predictions = {}, {}
    for name, model in models.items():
        predicted = model.predict(split.test[feature_columns])
        test_metrics[name] = classification_metrics(split.test["target"], predicted)
        table = split.test[["Date", "LabelDate", "Close", "next_return", "target"]].copy()
        table["prediction"] = predicted
        table["bullish_probability"] = model.predict_proba(split.test[feature_columns])[:, 1]
        predictions[name] = table
    return {"models": models, "selected": selected, "validation_metrics": validation_metrics,
            "test_metrics": test_metrics, "test_predictions": predictions}


def feature_importance(model: Pipeline) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        meaning = "Impurity-based importance; correlated indicators can share or distort importance."
    elif hasattr(estimator, "coef_"):
        values = abs(estimator.coef_[0])
        meaning = "Absolute standardized logistic coefficients; association, not causation."
    else:
        return pd.DataFrame(columns=["feature", "importance", "meaning"])
    return pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": values, "meaning": meaning}).sort_values(
        "importance", ascending=False)
