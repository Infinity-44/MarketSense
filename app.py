"""Run locally with: streamlit run app.py"""
from datetime import datetime, timezone
import io
import json
import logging
import hashlib
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import Settings
from src.analysis import run_analysis, save_analysis
from src.data_loader import load_csv
from src.feature_engineering import ANOMALY_COLUMNS
from src.market_prediction import feature_importance
from src.sample_data import EXAMPLE_FILENAME, EXAMPLE_URL, load_example
from src.live_data import normalize_nse_symbol
from src.live_dashboard import cached_daily, daily_deployment, live_workspace
from src.price_prediction import estimate_next_price
from src.visualization import confusion_chart, price_chart, series_chart, style_figure

ROOT = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="MarketSense | Market intelligence", page_icon="◈", layout="wide")
st.markdown(f"<style>{(ROOT / 'assets/dashboard.css').read_text()}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_analysis(content: bytes, settings: Settings, symbol: str, deploy_daily=False, lookback_months=None):
    bars, audit = load_csv(io.BytesIO(content))
    result = run_analysis(bars, settings, symbol, audit, lookback_months=lookback_months)
    if deploy_daily:
        result["daily_deployment"] = daily_deployment(result)
        result["price_forecast"] = estimate_next_price(result)
        result["source_hash"] = hashlib.sha256(content).hexdigest()
    return result


def acquisition_help():
    st.markdown('<div class="hero"><div class="mini-label">START YOUR RESEARCH</div>'
                '<h2>A clearer view<br>of the market.</h2>'
                '<p>Explore price behaviour, historical risk and unusual sessions. '
                'Select the real Apple example in the sidebar, or upload your own daily data.</p></div>', unsafe_allow_html=True)
    columns = st.columns(3)
    for col, title, detail in zip(columns, ["↗ Market behaviour", "◒ Historical risk", "⌁ Unusual sessions"],
                                  ["Three classifiers compared with a simple baseline.",
                                   "Volatility, drawdown and historical VaR.", "Outliers with indicator context."]):
        col.markdown(f'<div class="feature-tile"><b>{title}</b><span>{detail}</span></div>', unsafe_allow_html=True)
    with st.expander("CSV format & dataset help"):
        st.write("One daily stock per file, with Date, Open, High, Low, Close, Volume. "
                 "Supply at least 301 valid bars. CSV and comma-separated TXT files work.")
        st.link_button("Get the CC0 stock dataset ↗", "https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/")


def signal_panel(result):
    inference, prediction = result["latest_inference"], result["prediction"]
    selected = prediction["selected"]
    metrics = prediction["test_metrics"][selected]
    baseline = prediction["test_metrics"]["Majority baseline"]
    edge = metrics["accuracy"] - baseline["accuracy"]
    st.markdown('<div class="mini-label">NEXT-OBSERVATION ESTIMATE</div>', unsafe_allow_html=True)
    weak = metrics["accuracy"] <= max(0.5, baseline["accuracy"])
    css = "neutral" if weak else ("down" if inference["direction"] == "Bearish" else "")
    arrow = "↘" if inference["direction"] == "Bearish" else "↗"
    st.markdown(f'<div class="signal {css}">{inference["direction"]} {arrow}</div>', unsafe_allow_html=True)
    st.caption("Outcome unknown · after-close inputs")
    if weak:
        st.caption("Weak held-out results · exploratory estimate")
    st.markdown(f'<div class="insight">Validation-selected model<br><strong>{escape(selected)}</strong></div>', unsafe_allow_html=True)
    st.divider()
    a, b = st.columns(2)
    a.markdown(f'**{metrics["accuracy"]:.1%}**<br><span class="instrument-meta">Test accuracy</span>', unsafe_allow_html=True)
    b.markdown(f'**{baseline["accuracy"]:.1%}**<br><span class="instrument-meta">Baseline</span>', unsafe_allow_html=True)
    st.caption("No accuracy improvement over the majority baseline on this holdout." if edge <= 0 else
               f"{edge * 100:+.1f} percentage points vs baseline. This alone does not establish a reliable forecasting edge.")
    with st.expander("Read this estimate"):
        st.write(f"Bullish probability: {inference['bullish_probability']:.1%}. "
                 "Uncalibrated model output, not measured confidence.")
        st.caption(f"Frozen model fitted through {result['split'].train.Date.max():%d %b %Y}. "
                   "This is the evaluated model, not a newly refitted current-market forecast.")


def metric_table(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame({name: {key: values[key] for key in
                               ("accuracy", "precision_macro", "recall_macro", "f1_macro", "n")}
                         for name, values in metrics.items()}).T.rename_axis("Model")


def main():
    with st.sidebar:
        st.markdown('<div class="brand"><span class="brand-icon">↗</span>MarketSense</div>'
                    '<div class="brand-sub">THE RESEARCH WORKSPACE</div>', unsafe_allow_html=True)
        st.markdown('<div class="nav-label">ANALYZE</div>', unsafe_allow_html=True)
        section = st.radio("Workspace", ["Overview", "Price & Direction", "Market Behaviour", "Risk Analysis",
                                         "Anomaly Detection", "Model Evaluation"], label_visibility="collapsed")
        st.divider()
        st.markdown('<div class="nav-label">DATA SOURCE</div>', unsafe_allow_html=True)
        source = st.selectbox("Dataset", ["NSE · live + historical", "Apple · AAPL (historical example)", "Upload / local CSV"], label_visibility="collapsed")
        is_example = source.startswith("Apple")
        is_live = source.startswith("NSE")
        upload, local = None, None
        if is_live:
            choice = st.selectbox("NSE stock", ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "ITC", "BHARTIARTL", "Custom symbol"])
            live_symbol = st.text_input("NSE symbol", "RELIANCE") if choice == "Custom symbol" else choice
            if st.button("Refresh daily history"):
                cached_daily.clear()
            lookback = st.selectbox("Historical lookback", ["3 months", "5 months", "6 months", "1 year"], index=1)
            lookback_months = {"3 months": 3, "5 months": 5, "6 months": 6, "1 year": 12}[lookback]
            st.caption("The selected months control model training and the history chart. Latest price refreshes separately.")
        if not is_example and not is_live:
            upload = st.file_uploader("Daily OHLCV · CSV or TXT", type=["csv", "txt"])
            available = sorted([*ROOT.joinpath("data/raw").glob("*.csv"),
                                *ROOT.joinpath("data/raw").glob("*.txt")])
            available = [p for p in available if p.name != EXAMPLE_FILENAME]
            local = st.selectbox("Local dataset", available, format_func=lambda p: p.name) if available else None
        default_symbol = "AAPL" if is_example else (live_symbol if is_live else (Path(upload.name).stem.upper() if upload else (local.stem.upper() if local else "YOUR STOCK")))
        symbol = default_symbol if is_example or is_live else st.text_input("Instrument label", default_symbol, max_chars=40)
        with st.expander("Model & risk settings"):
            window = st.slider("Rolling risk window (observations)", 10, min(120, lookback_months * 20 - 20) if is_live else 120, 20)
            low = st.number_input("Low / moderate boundary (annualized %)", 1.0, 99.0, 15.0)
            high = st.number_input("Moderate / high boundary (annualized %)", 2.0, 200.0, 30.0)
            contamination = st.slider("Training outlier fraction (%)", 1, 15, 3)
        st.markdown('<div class="section-note">● Local computation<br>No paid API · No cloud account</div>', unsafe_allow_html=True)
    status = "NSE · CONNECTED FEED" if is_live else "HISTORICAL RESEARCH"
    st.markdown(f'<div class="topline"><div class="breadcrumb">Workspace / <span>{escape(section)}</span></div>'
                f'<div class="status">● {status}</div></div>', unsafe_allow_html=True)
    st.title(section)
    descriptions = {"Overview": "Your market research, in focus.", "Price & Direction": "Historical trends. Next-session price and direction estimates.", "Market Behaviour": "Follow price action. Inspect what the model learned.",
                    "Risk Analysis": "Measure historical downside and changing volatility.", "Anomaly Detection": "Find sessions that stand apart from the training regime.",
                    "Model Evaluation": "Evidence first. Compare every model with a simple baseline."}
    st.markdown(f'<div class="subheading">{descriptions[section]}</div>', unsafe_allow_html=True)
    if not is_example and not is_live and upload is None and local is None:
        acquisition_help()
        return
    try:
        settings = Settings(risk_window=window, low_risk_limit=low / 100,
                            high_risk_limit=high / 100, contamination=contamination / 100,
                            min_observations=50 if is_live else 250)
        if is_live:
            symbol = normalize_nse_symbol(symbol)
            with st.spinner(f"Fetching current daily history for {symbol}…"):
                daily_snapshot = cached_daily(symbol)
            content = daily_snapshot["content"]
        else:
            content = load_example(ROOT / "data/raw") if is_example else (upload.getvalue() if upload is not None else local.read_bytes())
        with st.spinner("Preparing your research · validating bars and evaluating models…"):
            result = cached_analysis(content, settings, symbol, deploy_daily=is_live, lookback_months=lookback_months if is_live else None)
    except (ValueError, OSError) as exc:
        st.error(str(exc))
        if is_live:
            try:
                symbol = normalize_nse_symbol(symbol)
            except ValueError:
                return
            live_workspace(symbol)
        else:
            acquisition_help()
        return

    bars, features = result["bars"], result["features"]
    prediction, risk = result["prediction"], result["risk_summary"]
    metadata, split = result["metadata"], result["split"]
    metadata["dataset_source"] = ({"kind": "current NSE feed", "provider": "Yahoo Finance via yfinance",
                                   "fetched_at": daily_snapshot["fetched_at"], "price_basis": "auto-adjusted OHLC",
                                   "currency": "INR", "provider_audit": daily_snapshot["audit"]} if is_live else
                                  {"kind": "public historical example", "url": EXAMPLE_URL,
                                   "price_basis": "unadjusted OHLC", "currency": "USD"} if is_example else
                                  {"kind": "user-provided CSV", "filename": Path(upload.name).name if upload else local.name})
    name = "Apple Inc." if is_example else symbol
    units = "NSE · adjusted INR · completed daily history" if is_live else ("Plotly public example · unadjusted USD · not current quotes" if is_example else "Your dataset · source price units")
    st.markdown(f'<div class="instrument"><div class="instrument-icon">{escape(symbol[:1])}</div><div>'
                f'<div class="instrument-name">{escape(name)} <span class="small-tag">{escape(symbol)}</span></div>'
                f'<div class="instrument-meta">{bars.Date.min():%d %b %Y} — {bars.Date.max():%d %b %Y}'
                f' &nbsp; / &nbsp; {len(bars):,} observations &nbsp; / &nbsp; {units}</div></div></div>', unsafe_allow_html=True)
    if is_example:
        st.caption("Real historical bars from Plotly’s public Apple example. All scores are computed locally. Switch the data source to analyze your own stock.")
    if is_live and section in ("Overview", "Price & Direction"):
        live_workspace(symbol, result)
        return
    if section == "Price & Direction":
        st.info("Choose NSE · live + historical in the sidebar for price and direction estimates from a selectable historical window.")
        return
    control, view = st.columns([3, 1])
    with control:
        period = st.radio("Chart window", ["3M", "6M", "1Y", "All"], index=2, horizontal=True)
    with view:
        kind = st.selectbox("Chart style", ["Line", "Candlestick"])
    months = {"3M": 3, "6M": 6, "1Y": 12}
    visible = features if period == "All" else features.loc[features.Date >= features.Date.max() - pd.DateOffset(months=months[period])]
    st.caption("Chart controls change the visible window only. Full-history risk summaries and held-out evaluation stay fixed.")
    removed = metadata["audit"]["input_rows"] - metadata["audit"]["clean_rows"]
    if removed:
        st.warning(f"Cleaning removed {removed} rows. Inspect the data-quality audit in Overview.")

    if section == "Overview":
        cards = st.columns(4)
        selected = prediction["selected"]
        performance = prediction["test_metrics"][selected]
        cards[0].metric("Latest historical close" + (" · USD" if is_example else ""), f"{bars.Close.iloc[-1]:,.2f}",
                        f"{features.daily_return.iloc[-1]:+.2%} last session")
        cards[1].metric(f"{window}-bar annualized volatility", f"{risk['latest_rolling_volatility']:.1%}", risk["risk_class"], delta_color="off")
        cards[2].metric("Held-out accuracy", f"{performance['accuracy']:.1%}", f"{performance['n']} test observations", delta_color="off")
        later = result["anomalies"].query("period == 'Later history (out-of-sample)'")
        cards[3].metric("Later-history outliers", f"{later.anomaly.sum():,}", f"of {len(later)} scored sessions", delta_color="off")
        st.write("")
        left, right = st.columns([3.1, 1.15], gap="medium")
        with left, st.container(border=True):
            st.subheader("Price & volume")
            st.caption("Historical price with 20- and 50-observation moving averages")
            st.plotly_chart(price_chart(visible, ["sma_20", "sma_50"], kind=kind), width="stretch")
        with right, st.container(border=True):
            signal_panel(result)
        left, middle, right = st.columns([1.15, 1.1, 1.05])
        with left, st.container(border=True):
            st.subheader("Downside at a glance")
            st.markdown(f"### {risk['maximum_drawdown']:.1%}")
            st.caption("Maximum drawdown · full supplied history")
            risk_visible = result["risk_frame"].loc[result["risk_frame"].Date.isin(visible.Date)]
            figure = series_chart(risk_visible, "drawdown", "", True)
            figure.update_layout(height=160, margin=dict(l=0, r=0, t=5, b=5), showlegend=False)
            st.plotly_chart(figure, width="stretch")
        with middle, st.container(border=True):
            st.subheader("Unusual sessions")
            st.caption("Most recent out-of-sample outliers · full history")
            recent = later.loc[later.anomaly].sort_values("Date", ascending=False).head(3)
            if recent.empty:
                st.caption("No later observations exceed the training threshold.")
            for row in recent.itertuples():
                st.markdown(f"**{row.Date:%d %b %Y}** · {row.daily_return:+.2%}")
                st.caption(f"Relative volume {row.volume_relative_20:.2f}× · score {row.anomaly_score:.3f}")
        with right, st.container(border=True):
            st.subheader("Research integrity")
            st.markdown(f"**{len(features):,}** validated daily bars")
            st.caption(f"{removed} source rows removed · no price interpolation")
            st.markdown("**60 / 20 / 20** chronological split")
            st.caption(f"{split.purged_rows} boundary labels purged · training-only transforms")
            st.markdown("**Historical**, never live")
            st.caption("Every price, score and chart comes from the selected data.")
        with st.expander("Data quality & source observations"):
            st.json(metadata["audit"])
            st.dataframe(bars.tail(50), hide_index=True, width="stretch")

    elif section == "Market Behaviour":
        inference = result["latest_inference"]
        cards = st.columns(3)
        cards[0].metric("Latest-row inference · unknown outcome", inference["direction"])
        cards[1].metric("Bullish model probability", f"{inference['bullish_probability']:.1%}")
        cards[2].metric("Model selected on validation", prediction["selected"])
        st.caption(f"As of {bars.Date.iloc[-1]:%d %b %Y}, for the next trading observation. "
                   "Probabilities are uncalibrated model outputs, not measured confidence or return forecasts. "
                   "The model remains fitted only on the training partition.")
        indicators = st.multiselect("Moving averages", ["sma_10", "sma_20", "sma_50"], ["sma_20", "sma_50"])
        st.plotly_chart(price_chart(visible, indicators, kind=kind), width="stretch")
        left, right = st.columns(2)
        left.plotly_chart(series_chart(visible, "rsi_14", "RSI · 14 observations"), width="stretch")
        right.plotly_chart(series_chart(visible, "macd", "MACD · source price units"), width="stretch")
        st.subheader("Held-out test predictions")
        st.caption("Date is the feature timestamp; LabelDate is when the next-close outcome became known. "
                   "target/prediction: 0 = Bearish (including zero return), 1 = Bullish.")
        table = prediction["test_predictions"][prediction["selected"]]
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button("Download held-out predictions", table.to_csv(index=False), "test_predictions.csv", "text/csv")

    elif section == "Risk Analysis":
        cards = st.columns(4)
        cards[0].metric("Full-period annualized volatility", f"{risk['annualized_volatility']:.1%}")
        cards[1].metric("Full-period max drawdown", f"{risk['maximum_drawdown']:.1%}")
        cards[2].metric("Historical one-day VaR · 95%", f"{risk['historical_var_loss']:.2%}")
        cards[3].metric("Annualized downside deviation", f"{risk['downside_volatility']:.1%}")
        st.write(f"Current classification: **{risk['risk_class']}**, based on the latest **{window}-observation** "
                 f"annualized volatility. Low < {low:.0f}%; moderate {low:.0f}% to < {high:.0f}%; high ≥ {high:.0f}%.")
        st.caption("Thresholds are configurable teaching assumptions, not universal risk bands. "
                   "Annualization assumes 252 trading observations per year. VaR is a historical return quantile "
                   "expressed as a positive loss; actual losses can exceed it.")
        risk_frame = result["risk_frame"].loc[result["risk_frame"].Date.isin(visible.Date)]
        left, right = st.columns(2)
        left.plotly_chart(series_chart(risk_frame, "annualized_volatility", "Rolling annualized volatility", True), width="stretch")
        right.plotly_chart(series_chart(risk_frame, "drawdown", "Historical drawdown", True), width="stretch")
        left, right = st.columns(2)
        left.plotly_chart(series_chart(risk_frame, "var_loss", "Rolling one-day historical VaR · 95%", True), width="stretch")
        right.plotly_chart(series_chart(risk_frame, "downside_volatility", "Rolling annualized downside deviation", True), width="stretch")
        st.caption(f"Full-period daily volatility: {risk['historical_daily_volatility']:.2%}. "
                   "Downside deviation uses a zero return target and includes all returns in the denominator. "
                   "Historical measurements do not guarantee future loss limits.")

    elif section == "Anomaly Detection":
        st.caption(f"Isolation Forest fitted through {result['detector']['fit_end']:%d %b %Y}. "
                   f"Training contamination = {contamination}%; later periods may have a different outlier rate.")
        scope = st.radio("Scoring period", ["Later history (out-of-sample)", "All historical observations"], horizontal=True)
        scored = result["anomalies"]
        if scope.startswith("Later"):
            scored = scored[scored.period == "Later history (out-of-sample)"]
        scored = scored.loc[scored.Date.isin(visible.Date)]
        st.plotly_chart(price_chart(visible, anomalies=scored, kind=kind), width="stretch")
        st.plotly_chart(series_chart(scored, "anomaly_score", "Outlier score · positive values are flagged"), width="stretch")
        flagged = scored.loc[scored.anomaly, ["Date", "Close", "anomaly_score", "period", "indicator_context", *ANOMALY_COLUMNS]]
        st.subheader(f"Flagged observations · {len(flagged):,}")
        st.caption("Selected chart window and scoring period only.")
        st.dataframe(flagged.sort_values("anomaly_score", ascending=False), hide_index=True, width="stretch")
        st.caption("Context lists the two largest absolute deviations from training indicator means. "
                   "These are descriptive comparisons, not explanations of the forest's internal decisions. "
                   "An outlier is not evidence of fraud or manipulation. No labelled anomaly ground truth is available.")
        st.download_button("Download outlier table", flagged.to_csv(index=False), "anomalies.csv", "text/csv")

    else:
        st.subheader("Evidence before predictions")
        st.caption(f"Selected on validation macro F1: {prediction['selected']}. The chart-window control does not change these full-holdout metrics.")
        scorecards = st.columns(3)
        selected_metrics = prediction["test_metrics"][prediction["selected"]]
        scorecards[0].metric("Selected model · test accuracy", f"{selected_metrics['accuracy']:.1%}")
        scorecards[1].metric("Majority baseline · test accuracy", f"{prediction['test_metrics']['Majority baseline']['accuracy']:.1%}")
        scorecards[2].metric("Selected model · test macro F1", f"{selected_metrics['f1_macro']:.3f}")
        with st.expander("Chronological partitions & training protocol"):
            st.dataframe(pd.DataFrame(metadata["split_periods"]).T, width="stretch")
            st.caption(f"60% training / 20% validation / 20% test before purging. {split.purged_rows} boundary rows purged. "
                   "All preprocessing fits training only. Fixed model settings; validation macro-F1 selects the model. "
                   "Test results are reported after selection; no refitting or tuning on test.")
        test_tab, validation_tab = st.tabs(["Held-out test", "Validation · model selection"])
        formats = {"accuracy": "{:.1%}", "precision_macro": "{:.3f}", "recall_macro": "{:.3f}",
                   "f1_macro": "{:.3f}", "n": "{:.0f}"}
        with validation_tab:
            st.dataframe(metric_table(prediction["validation_metrics"]).style.format(formats), width="stretch")
        comparison = metric_table(prediction["test_metrics"])
        with test_tab:
            st.dataframe(comparison.style.format(formats), width="stretch")
        figure = go.Figure()
        for metric, color in [("accuracy", "#B8F275"), ("f1_macro", "#91B5CE")]:
            figure.add_bar(x=comparison.index, y=comparison[metric], name=metric, marker_color=color)
        style_figure(figure, "Held-out test performance")
        figure.update_yaxes(range=[0, 1])
        st.plotly_chart(figure, width="stretch")
        name = st.selectbox("Inspect model", list(prediction["models"]),
                            index=list(prediction["models"]).index(prediction["selected"]))
        left, right = st.columns(2)
        left.plotly_chart(confusion_chart(prediction["test_metrics"][name]["confusion_matrix"],
                                         "Test confusion matrix"), width="stretch")
        with right:
            st.subheader("Test classification report")
            report = prediction["test_metrics"][name]["classification_report"]
            rows = {key: value for key, value in report.items() if isinstance(value, dict)}
            st.dataframe(pd.DataFrame(rows).T.style.format({"precision": "{:.3f}", "recall": "{:.3f}",
                                                           "f1-score": "{:.3f}", "support": "{:.0f}"}), width="stretch")
        importance = feature_importance(prediction["models"][name])
        if not importance.empty:
            st.caption(importance.meaning.iloc[0])
            figure = go.Figure(go.Bar(x=importance.importance, y=importance.feature,
                                      orientation="h", marker_color="#B8F275"))
            style_figure(figure, "Feature associations")
            figure.update_layout(height=420, yaxis_autorange="reversed")
            st.plotly_chart(figure, width="stretch")
        st.caption("Macro metrics weight both direction classes equally. Small, dependent test samples and repeated "
                   "experiments limit conclusions. No transaction costs, trading strategy, or profitability are evaluated.")
        with st.expander("Run metadata and reproducibility"):
            st.json(metadata)
        st.download_button("Download evaluation report", json.dumps({"metadata": metadata,
                            "validation": prediction["validation_metrics"], "test": prediction["test_metrics"]},
                            indent=2), "evaluation.json", "application/json")
        if st.button("Save models and result tables locally"):
            destination = ROOT / "models" / ("run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
            try:
                saved = save_analysis(result, destination)
                st.success(f"Saved to models/{saved.name}")
            except OSError as exc:
                st.error(f"Could not save the run: {exc}")

    st.divider()
    st.markdown('<div class="footer"><span>↗ MarketSense · Understand the evidence.</span><span>Historical analysis · Educational use · No trade execution</span></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
