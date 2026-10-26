"""Run locally with: streamlit run app.py"""
from datetime import datetime, timezone
import io
import json
import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import Settings
from src.analysis import run_analysis, save_analysis
from src.data_loader import load_csv
from src.feature_engineering import ANOMALY_COLUMNS
from src.market_prediction import feature_importance
from src.visualization import confusion_chart, price_chart, series_chart, style_figure

ROOT = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="MarketSense | Market intelligence", page_icon="◈", layout="wide")
st.markdown("""
<style>
.block-container {padding-top:2rem; max-width:1500px;}
[data-testid="stMetric"] {background:#141C30;border:1px solid #25314C;
 border-radius:14px;padding:18px;}
[data-testid="stMetricLabel"] {color:#A8B5CC;}
h1 {letter-spacing:-.045em;} h2,h3 {letter-spacing:-.025em;}
.eyebrow {color:#44D7B6;font-size:12px;letter-spacing:.20em;font-weight:700;}
.intro {color:#A8B5CC;max-width:800px;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_analysis(content: bytes, settings: Settings, symbol: str):
    bars, audit = load_csv(io.BytesIO(content))
    return run_analysis(bars, settings, symbol, audit)


def acquisition_help():
    st.info("Add a real historical dataset to begin. No fabricated prices or model scores are displayed.")
    st.markdown("""
1. Open [Huge Stock Market Dataset on Kaggle](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs).
2. Download and extract the archive (a free Kaggle account may be required).
3. Select one file, such as **Stocks/aapl.us.txt**, and upload it in the sidebar.
   Alternatively, copy it into **data/raw/** and refresh this page.

Expected columns: `Date, Open, High, Low, Close, Volume`. Kaggle's comma-separated
`.txt` files are supported. Supply at least **301 valid daily bars**; several
years are preferable. One instrument per file. This historical release ends in
2017 and is intended for offline study.
""")
    st.code("python train.py --data data/raw/aapl.us.txt --symbol AAPL\nstreamlit run app.py", language="bash")


def metric_table(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame({name: {key: values[key] for key in
                               ("accuracy", "precision_macro", "recall_macro", "f1_macro", "n")}
                         for name, values in metrics.items()}).T.rename_axis("Model")


def main():
    with st.sidebar:
        st.markdown("## ◈ MarketSense")
        st.caption("MARKET BEHAVIOUR · RISK · ANOMALIES")
        section = st.radio("Workspace", ["Overview", "Market Behaviour", "Risk Analysis",
                                         "Anomaly Detection", "Model Evaluation"])
        st.divider()
        upload = st.file_uploader("Upload a single-stock CSV", type=["csv", "txt"])
        available = sorted([*ROOT.joinpath("data/raw").glob("*.csv"),
                            *ROOT.joinpath("data/raw").glob("*.txt")])
        local = st.selectbox("Local dataset", available, format_func=lambda p: p.name) if available else None
        default_symbol = Path(upload.name).stem.upper() if upload else (local.stem.upper() if local else "AAPL")
        symbol = st.text_input("Instrument label", value=default_symbol, max_chars=40)
        with st.expander("Analysis settings"):
            window = st.slider("Rolling risk window (observations)", 10, 120, 20)
            low = st.number_input("Low / moderate boundary (annualized %)", 1.0, 99.0, 15.0)
            high = st.number_input("Moderate / high boundary (annualized %)", 2.0, 200.0, 30.0)
            contamination = st.slider("Training outlier fraction (%)", 1, 15, 3)
        st.caption("All computation runs locally. No paid API or cloud service required.")
    st.markdown('<div class="eyebrow">MARKETSENSE / RESEARCH WORKSPACE</div>', unsafe_allow_html=True)
    st.title(section)
    st.markdown('<p class="intro">Understand historical market behaviour through reproducible '
                'machine learning, transparent risk measurements, and statistical outliers.</p>', unsafe_allow_html=True)
    st.caption("Educational analysis • Daily bars • Predictions use information available after the current close")
    if upload is None and local is None:
        acquisition_help()
        return
    try:
        settings = Settings(risk_window=window, low_risk_limit=low / 100,
                            high_risk_limit=high / 100, contamination=contamination / 100)
        content = upload.getvalue() if upload is not None else local.read_bytes()
        with st.spinner("Validating bars, training on historical data, and evaluating held-out observations…"):
            result = cached_analysis(content, settings, symbol)
    except (ValueError, OSError) as exc:
        st.error(str(exc))
        st.caption("Correct the input or settings and retry. See docs/dataset.md for supported files.")
        return

    bars, features = result["bars"], result["features"]
    prediction, risk = result["prediction"], result["risk_summary"]
    metadata, split = result["metadata"], result["split"]
    st.caption(f"{symbol} · {bars.Date.min():%d %b %Y} — {bars.Date.max():%d %b %Y} · "
               f"{len(bars):,} historical bars · Source units and currency as supplied")
    removed = metadata["audit"]["input_rows"] - len(bars)
    if removed:
        st.warning(f"Cleaning removed {removed} rows. Inspect the data-quality audit in Overview.")

    if section == "Overview":
        cards = st.columns(4)
        cards[0].metric("Latest historical close", f"{bars.Close.iloc[-1]:,.2f}",
                        f"{features.daily_return.iloc[-1]:+.2%} last observed return")
        cards[1].metric(f"{window}-bar annualized volatility", f"{risk['latest_rolling_volatility']:.1%}")
        cards[2].metric("Historical risk category", risk["risk_class"])
        later = result["anomalies"].query("period == 'Later history (out-of-sample)'")
        cards[3].metric("Later-history outliers", f"{later.anomaly.sum():,}", delta_color="off")
        st.plotly_chart(price_chart(features, ["sma_20", "sma_50"]), width="stretch")
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Analysis snapshot")
            st.write(f"Validation selected **{prediction['selected']}**. Its held-out test macro-F1 is "
                     f"**{prediction['test_metrics'][prediction['selected']]['f1_macro']:.3f}** across "
                     f"{len(split.test)} labelled observations.")
            st.write(f"Full-period maximum drawdown is **{risk['maximum_drawdown']:.1%}**. "
                     "Risk categories describe recent historical volatility; they are not a full investment-risk assessment.")
            st.caption("Accuracy above 50% alone does not establish financial predictability or profitable trading.")
        with right:
            st.subheader("Data quality")
            st.json(metadata["audit"])
        with st.expander("Inspect source bars"):
            st.dataframe(bars.tail(100), hide_index=True, width="stretch")

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
        st.plotly_chart(price_chart(features, indicators), width="stretch")
        left, right = st.columns(2)
        left.plotly_chart(series_chart(features, "rsi_14", "RSI · 14 observations"), width="stretch")
        right.plotly_chart(series_chart(features, "macd", "MACD · source price units"), width="stretch")
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
        risk_frame = result["risk_frame"]
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
        st.plotly_chart(price_chart(features, anomalies=scored), width="stretch")
        st.plotly_chart(series_chart(scored, "anomaly_score", "Outlier score · positive values are flagged"), width="stretch")
        flagged = scored.loc[scored.anomaly, ["Date", "Close", "anomaly_score", "period", "indicator_context", *ANOMALY_COLUMNS]]
        st.subheader(f"Flagged observations · {len(flagged):,}")
        st.dataframe(flagged.sort_values("anomaly_score", ascending=False), hide_index=True, width="stretch")
        st.caption("Context lists the two largest absolute deviations from training indicator means. "
                   "These are descriptive comparisons, not explanations of the forest's internal decisions. "
                   "An outlier is not evidence of fraud or manipulation. No labelled anomaly ground truth is available.")
        st.download_button("Download outlier table", flagged.to_csv(index=False), "anomalies.csv", "text/csv")

    else:
        st.subheader("Chronological evaluation")
        st.dataframe(pd.DataFrame(metadata["split_periods"]).T, width="stretch")
        st.caption(f"60% training / 20% validation / 20% test before purging. {split.purged_rows} boundary rows purged. "
                   "All preprocessing fits training only. Fixed model settings; validation macro-F1 selects the model. "
                   "Test results are reported after selection; no refitting or tuning on test.")
        st.subheader("Validation · used for model selection")
        st.dataframe(metric_table(prediction["validation_metrics"]).style.format("{:.3f}"), width="stretch")
        st.subheader("Held-out test · final comparison")
        comparison = metric_table(prediction["test_metrics"])
        st.dataframe(comparison.style.format("{:.3f}"), width="stretch")
        figure = go.Figure()
        for metric, color in [("accuracy", "#44D7B6"), ("f1_macro", "#7DABFF")]:
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
            st.dataframe(pd.DataFrame(prediction["test_metrics"][name]["classification_report"]).T, width="stretch")
        importance = feature_importance(prediction["models"][name])
        if not importance.empty:
            st.caption(importance.meaning.iloc[0])
            st.bar_chart(importance.set_index("feature")["importance"], horizontal=True)
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
    st.caption("MarketSense · Educational historical analysis · Not investment advice or an automated trading system")


if __name__ == "__main__":
    main()
