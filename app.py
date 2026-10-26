"""Run locally with: streamlit run app.py"""
import io
import logging
import hashlib
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import Settings
from src.analysis import run_analysis
from src.data_loader import load_csv
from src.feature_engineering import ANOMALY_COLUMNS
from src.market_prediction import feature_importance
from src.sample_data import EXAMPLE_FILENAME, EXAMPLE_URL, load_example
from src.live_data import normalize_nse_symbol
from src.live_dashboard import cached_daily, live_workspace, render_exports, render_price_evaluation
from src.visualization import confusion_chart, price_chart, series_chart, style_figure

ROOT = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="MarketSense | Market intelligence", page_icon="◈", layout="wide")
st.markdown(f"<style>{(ROOT / 'assets/dashboard.css').read_text()}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_analysis(content: bytes, settings: Settings, symbol: str, lookback_months=None):
    bars, audit = load_csv(io.BytesIO(content))
    result = run_analysis(bars, settings, symbol, audit, lookback_months=lookback_months)
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
                 "Supply at least 300 valid bars for the full-history CSV workflow. CSV and comma-separated TXT files work.")
        st.link_button("Get the CC0 stock dataset ↗", "https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/")


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
            lookback = st.selectbox("Historical lookback", ["3 months", "5 months", "6 months", "1 year"], index=1, help="Changes both the historical model-fitting rows and the price chart. Earlier rows only initialize indicators.")
            lookback_months = {"3 months": 3, "5 months": 5, "6 months": 6, "1 year": 12}[lookback]
            st.caption("The selected months control model training and the history chart. Latest price refreshes separately.")
        if not is_example and not is_live:
            upload = st.file_uploader("Daily OHLCV · CSV or TXT", type=["csv", "txt"])
            available = sorted([*ROOT.joinpath("data/raw").glob("*.csv"),
                                *ROOT.joinpath("data/raw").glob("*.txt")])
            available = [p for p in available if p.name != EXAMPLE_FILENAME]
            local = st.selectbox("Local dataset", available, format_func=lambda p: p.name) if available else None
        csv_currency = st.selectbox("Price currency", ["Source units", "INR", "USD"], help="Display label only; prices are not converted.") if not is_live and not is_example else None
        default_symbol = "AAPL" if is_example else (live_symbol if is_live else (Path(upload.name).stem.upper() if upload else (local.stem.upper() if local else "YOUR STOCK")))
        symbol = default_symbol if is_example or is_live else st.text_input("Instrument label", default_symbol, max_chars=40)
        with st.expander("Model & risk settings"):
            window = st.slider("Rolling risk window (observations)", 10, min(120, lookback_months * 20 - 20) if is_live else 120, 20)
            low = st.number_input("Low / moderate boundary (annualized %)", 1.0, 99.0, 15.0)
            high = st.number_input("Moderate / high boundary (annualized %)", 2.0, 200.0, 30.0)
            contamination = st.slider("Training outlier fraction (%)", 1, 15, 3)
        st.markdown('<div class="section-note">● Local computation<br>No paid API · No cloud account</div>', unsafe_allow_html=True)
    status = "NSE · PROVIDER DATA" if is_live else "HISTORICAL RESEARCH"
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
            result = cached_analysis(content, settings, symbol, lookback_months=lookback_months if is_live else None)
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
                                  {"kind": "user-provided CSV", "filename": Path(upload.name).name if upload else local.name, "currency": csv_currency})
    name = "Apple Inc." if is_example else symbol.removesuffix(".NS")
    units = "NSE · adjusted INR · completed daily history" if is_live else ("Plotly public example · unadjusted USD · not current quotes" if is_example else "Your dataset · source price units")
    st.markdown(f'<div class="instrument"><div class="instrument-icon">{escape(symbol[:1])}</div><div>'
                f'<div class="instrument-name">{escape(name)} <span class="small-tag">{"NSE" if is_live else "EXAMPLE" if is_example else "CSV"}</span></div>'
                f'<div class="instrument-meta">{bars.Date.min():%d %b %Y} — {bars.Date.max():%d %b %Y}'
                f' &nbsp; / &nbsp; {len(bars):,} observations &nbsp; / &nbsp; {units}</div></div></div>', unsafe_allow_html=True)
    if is_example:
        st.caption("Real historical bars from Plotly’s public Apple example. All scores are computed locally. Switch the data source to analyze your own stock.")
    if section in ("Overview", "Price & Direction"):
        live_workspace(symbol, result, live=is_live, detailed=section == "Price & Direction")
        return
    visible = features
    if section != "Model Evaluation":
        control, view = st.columns([3, 1])
        with control:
            period = st.radio("Chart window", ["3M", "5M", "6M", "1Y", "All"], index=4, horizontal=True,
                              help="Display filter only. The sidebar historical lookback controls model fitting.")
        with view:
            kind = st.selectbox("Chart style", ["Line", "Candlestick"])
        months = {"3M": 3, "5M": 5, "6M": 6, "1Y": 12}
        if period != "All":
            visible = features.loc[features.Date >= features.Date.max() - pd.DateOffset(months=months[period])]
        st.caption("Display filter only · evaluation uses the full modelling window selected in the sidebar.")
    removed = metadata["audit"]["input_rows"] - metadata["audit"]["clean_rows"]
    if removed:
        st.warning(f"Cleaning removed {removed} rows. Inspect the data-quality audit in Overview.")

    if section == "Market Behaviour":
        inference = result["daily_deployment"]
        cards = st.columns(3)
        cards[0].metric("Latest-row inference · unknown outcome", inference["direction"])
        cards[1].metric("Bullish model probability", f"{inference['bullish_probability']:.1%}")
        cards[2].metric("Model selected on validation", prediction["selected"])
        st.caption(f"As of {bars.Date.iloc[-1]:%d %b %Y}, for the next trading observation. "
                   "Probabilities are uncalibrated model outputs, not measured confidence or return forecasts. "
                   "This deployment copy refits resolved labels after evaluation. Held-out predictions below come from the earlier frozen test model.")
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
        cards[0].metric("History · annualized volatility", f"{risk['annualized_volatility']:.1%}", help="Sample daily return standard deviation multiplied by √252, across the modelling history.")
        cards[1].metric("History · maximum drawdown", f"{risk['maximum_drawdown']:.1%}", help="Largest decline from a previous observed peak within the selected history.")
        cards[2].metric("Historical one-day VaR · 95%", f"{risk['historical_var_loss']:.2%}", help="Historical fifth-percentile daily return, expressed as a positive loss. Future losses can exceed it.")
        cards[3].metric("Annualized downside deviation", f"{risk['downside_volatility']:.1%}", help="Square root of mean squared negative returns, annualized by √252. Target return is zero.")
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
        if scored.empty:
            st.info("No scored observations in this display window. Choose All or a different scoring period.")
        else:
            figure = series_chart(scored, "anomaly_score", "Outlier score · positive values are flagged")
            figure.add_hline(y=0, line_dash="dot", line_color="#B8F275")
            st.plotly_chart(figure, width="stretch")
        flagged = scored.loc[scored.anomaly, ["Date", "Close", "anomaly_score", "period", "indicator_context", *ANOMALY_COLUMNS]]
        st.subheader(f"Flagged observations · {len(flagged):,}")
        st.caption("Selected chart window and scoring period only.")
        st.dataframe(flagged.sort_values("anomaly_score", ascending=False).style.format({
            "Close": "{:,.2f}", "anomaly_score": "{:.3f}", "daily_return": "{:+.2%}",
            "volatility_20": "{:.2%}", "volume_change": "{:+.1%}", "volume_relative_20": "{:.2f}×",
            "momentum_10": "{:+.2%}", "intraday_range": "{:.2%}"}), hide_index=True, width="stretch")
        st.caption("Context lists the two largest absolute deviations from training indicator means. "
                   "These are descriptive comparisons, not explanations of the forest's internal decisions. "
                   "An outlier is not evidence of fraud or manipulation. No labelled anomaly ground truth is available.")
        st.download_button("Download outlier table", flagged.to_csv(index=False), "anomalies.csv", "text/csv")

    else:
        direction_tab, price_tab = st.tabs(["Direction models", "Price models"])
        with price_tab:
            render_price_evaluation(result)
        with direction_tab:
            st.subheader("Evidence before predictions")
            st.caption(f"Selected on validation macro F1: {prediction['selected']}. Test data evaluates the earlier frozen model; it does not choose the winner.")
            scorecards = st.columns(3)
            selected_metrics = prediction["test_metrics"][prediction["selected"]]
            scorecards[0].metric("Selected model · test accuracy", f"{selected_metrics['accuracy']:.1%}")
            scorecards[1].metric("Majority baseline · test accuracy", f"{prediction['test_metrics']['Majority baseline']['accuracy']:.1%}")
            scorecards[2].metric("Selected model · test macro F1", f"{selected_metrics['f1_macro']:.3f}", help="Average of class-specific F1 scores, weighting Bullish and Bearish equally. F1 balances precision and recall.")
            with st.expander("Chronological partitions & training protocol"):
                st.dataframe(pd.DataFrame(metadata["split_periods"]).T, width="stretch")
                st.caption(f"60% training / 20% validation / 20% test before purging. {split.purged_rows} boundary rows purged. "
                       "All preprocessing fits training only. Fixed model settings; validation macro-F1 selects the model. "
                       "Test results are reported after selection; no refitting or tuning on test.")
            test_tab, validation_tab = st.tabs(["Held-out test", "Validation · model selection"])
            labels = {"accuracy": "Accuracy", "precision_macro": "Macro precision", "recall_macro": "Macro recall",
                      "f1_macro": "Macro F1", "n": "Sessions"}
            formats = {"Accuracy": "{:.1%}", "Macro precision": "{:.3f}", "Macro recall": "{:.3f}",
                       "Macro F1": "{:.3f}", "Sessions": "{:.0f}"}
            with validation_tab:
                st.dataframe(metric_table(prediction["validation_metrics"]).rename(columns=labels).style.format(formats), width="stretch")
            comparison = metric_table(prediction["test_metrics"])
            with test_tab:
                st.dataframe(comparison.rename(columns=labels).style.format(formats), width="stretch")
            with st.expander("What do these metrics mean?"):
                st.write("Accuracy: fraction of correct predictions. Precision: how often a predicted class is correct. "
                         "Recall: how many actual observations of a class are identified. F1 balances precision and recall. "
                         "Macro scores weight Bullish and Bearish equally. Support is the actual sample count for a class.")
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
    render_exports(result)
    st.divider()
    st.markdown('<div class="footer"><span>↗ MarketSense · Understand the evidence.</span><span>Historical analysis · Educational use · No trade execution</span></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
