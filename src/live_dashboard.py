"""Streamlit live workspace; refresh quotes without retraining every minute."""
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.base import clone

from src.analysis import save_analysis
from src.feature_engineering import FEATURE_COLUMNS, create_labels
from src.live_data import fetch_daily, fetch_intraday, utc_now
from src.visualization import price_chart


@st.cache_data(ttl=1800, max_entries=8, show_spinner=False)
def cached_daily(symbol):
    return fetch_daily(symbol)


@st.cache_data(ttl=55, max_entries=8, show_spinner=False)
def cached_intraday(symbol):
    return fetch_intraday(symbol)


def daily_deployment(result):
    """Separate deployment refit after the historical test has been evaluated."""
    labelled = create_labels(result["features"]).dropna(subset=FEATURE_COLUMNS + ["target", "LabelDate"])
    model = clone(result["prediction"]["models"][result["prediction"]["selected"]])
    model.fit(labelled[FEATURE_COLUMNS], labelled.target.astype(int))
    result["daily_deployment_model"] = model
    latest = result["features"].iloc[[-1]]
    return {"direction": "Bullish" if model.predict(latest[FEATURE_COLUMNS])[0] else "Bearish",
            "bullish_probability": float(model.predict_proba(latest[FEATURE_COLUMNS])[0, 1]),
            "as_of": latest.Date.iloc[0].isoformat(), "training_label_end": labelled.LabelDate.max().isoformat(),
            "note": "Deployment clone refitted on all resolved daily labels after historical evaluation; unknown next-session outcome."}


@st.fragment(run_every="60s")
def live_workspace(symbol: str, daily_result=None):
    st.caption("Latest NSE observations refresh every 60 seconds. Historical-window models update when completed daily data changes.")
    if st.button("Refresh latest price", key=f"refresh_{symbol}"):
        cached_intraday.clear()
    snapshot = None
    try:
        snapshot = cached_intraday(symbol)
    except ValueError as exc:
        st.warning(str(exc))
    if daily_result is None:
        st.info("Completed daily history is needed before a price or direction estimate can be calculated.")
        return
    try:
        refreshed = cached_daily(symbol)
        if (daily_result.get("source_hash") is not None
                and hashlib.sha256(refreshed["content"]).hexdigest() != daily_result["source_hash"]):
            st.rerun(scope="app")
    except ValueError:
        st.caption("Daily-history refresh is unavailable; estimates retain the input date shown below.")
    bars = daily_result["bars"]
    forecast = daily_result["price_forecast"]
    direction = daily_result["daily_deployment"]
    months = daily_result["metadata"]["lookback_months"]
    if snapshot is not None:
        latest = snapshot["bars"].iloc[-1]
        quote, quoted_at = float(latest.Close), latest.Date + pd.Timedelta(minutes=5)
        age = max(0, (utc_now() - quoted_at).total_seconds() / 60)
        quote_description = f"{quoted_at:%d %b %Y %H:%M} IST · completed 5-minute bar"
    else:
        quote = float(bars.Close.iloc[-1])
        quoted_at = None
        age = None
        quote_description = "Live feed unavailable · showing the labelled historical daily close"
    cards = st.columns(4)
    cards[0].metric(f"{symbol} · latest available price", f"₹{quote:,.2f}")
    cards[1].metric("Next-session price estimate", f"₹{forecast['estimated_price']:,.2f}", f"{forecast['estimated_return']:+.2%} vs input daily close")
    cards[2].metric("Next-session direction estimate", direction["direction"])
    cards[3].metric("Historical modelling window", f"{months} months")
    st.caption(quote_description)
    if age is not None and age > 15:
        st.caption(f"Latest supplied price is {age:.0f} minutes old: market closure or feed delay. It is not a fresh live quote.")
    st.caption(f"Both estimates use completed daily inputs through {bars.Date.max():%d %b %Y}. "
               "The price estimate is relative to that daily close, not a return applied to an unfinished session's live quote.")
    left, right = st.columns([2.8, 1.15])
    with left, st.container(border=True):
        st.subheader(f"{symbol.removesuffix('.NS')} · {months}-month history")
        st.caption(f"{bars.Date.min():%d %b %Y} – {bars.Date.max():%d %b %Y} · adjusted INR · {len(bars)} observed sessions")
        chart_style = st.radio("Historical chart", ["Line", "Candlestick"], horizontal=True, key=f"history_style_{symbol}")
        st.plotly_chart(price_chart(daily_result["features"], ["sma_20", "sma_50"], kind=chart_style), width="stretch")
    with right, st.container(border=True):
        st.markdown('<div class="mini-label">NEXT TRADING SESSION</div>', unsafe_allow_html=True)
        st.markdown(f"### ₹{forecast['estimated_price']:,.2f}")
        st.caption("Estimated closing price · outcome unknown")
        st.write(f"Price model: **{forecast['selected_model']}**")
        if forecast["selected_model"] == "No price change":
            st.caption("Historical validation preferred the no-change baseline. No learned price improvement is being claimed.")
        st.divider()
        st.write(f"Direction classifier: **{direction['direction']}**")
        st.caption(f"Bullish probability {direction['bullish_probability']:.1%} is uncalibrated. "
                   "The classifier and price regressor are different models and can disagree.")
        selected = daily_result["prediction"]["selected"]
        metrics = daily_result["prediction"]["test_metrics"][selected]
        baseline = daily_result["prediction"]["test_metrics"]["Majority baseline"]
        st.caption(f"Historical direction test: {metrics['accuracy']:.1%} accuracy / {metrics['f1_macro']:.3f} macro F1 "
                   f"over {metrics['n']} sessions. Majority baseline: {baseline['accuracy']:.1%}.")
    st.subheader("How the estimate performed historically")
    price_metrics = forecast["test_metrics"][forecast["selected_model"]]
    base_metrics = forecast["test_metrics"]["No price change"]
    columns = st.columns(3)
    columns[0].metric("Price model · held-out MAE", f"₹{price_metrics['price_mae']:.2f}")
    columns[1].metric("No-change baseline · held-out MAE", f"₹{base_metrics['price_mae']:.2f}")
    columns[2].metric("Held-out observations", str(price_metrics["observations"]))
    st.caption("MAE is average historical absolute price error, not a future prediction interval. "
               "Model selection uses validation only; a separate deployment clone then refits all resolved labels. "
               "Historical test scores belong to the earlier frozen models. A short history does not establish forecasting reliability.")
    with st.expander("Methodology, data quality & report"):
        st.write(f"The selected {months}-month window controls both historical model fitting and the price chart. "
                 "Earlier bars only warm up causal indicators, such as SMA 50; their rows are excluded from fitting/evaluation. "
                 "Targets are next-observation daily return and direction. Chronological splits purge boundary labels. "
                 "The input timestamp always refers to a completed daily session.")
        st.json(daily_result["metadata"]["dataset_source"])
        export_price = {k:v for k,v in forecast.items() if k not in ("deployment_model", "historical_models")}
        report = {"symbol": symbol, "metadata": daily_result["metadata"], "price_forecast": export_price,
                  "direction_forecast": direction, "historical_direction_test": daily_result["prediction"]["test_metrics"]}
        st.download_button("Download price & direction evaluation", json.dumps(report, indent=2, allow_nan=False),
                           f"{symbol}_forecast.json", "application/json")
        if st.button("Save this analysis locally", key=f"save_{symbol}"):
            try:
                output = Path(__file__).resolve().parents[1] / "models" / ("run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
                save_analysis(daily_result, output)
                st.success(f"Saved models/{output.name}")
            except OSError as exc:
                st.error(str(exc))
