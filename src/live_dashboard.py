"""Stock research workspace with consistent forecasts, evidence and exports."""
import hashlib
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from src.analysis import save_analysis
from src.live_data import fetch_daily, fetch_intraday, utc_now
from src.reporting import analysis_bundle, markdown_summary, model_assessment, price_unit
from src.visualization import price_chart, price_backtest_chart, series_chart


@st.cache_data(ttl=1800, max_entries=8, show_spinner=False)
def cached_daily(symbol):
    return fetch_daily(symbol)


@st.cache_data(ttl=55, max_entries=8, show_spinner=False)
def cached_intraday(symbol):
    return fetch_intraday(symbol)


def render_exports(result):
    with st.expander("Export analysis & understand the methodology"):
        report_tab, method_tab, data_tab = st.tabs(["Downloads", "How it works", "Data quality"])
        with report_tab:
            left, right = st.columns(2)
            left.download_button("Download research bundle ↓", analysis_bundle(result),
                                 "marketsense_research.zip", "application/zip", width="stretch")
            right.download_button("Download readable summary ↓", markdown_summary(result),
                                  "marketsense_summary.md", "text/markdown", width="stretch")
            st.caption("The bundle includes metrics, forecasts, source observations and held-out prediction tables. No executable model files are included.")
            if st.button("Save trained models & analysis locally"):
                try:
                    root = Path(__file__).resolve().parents[1]
                    folder = save_analysis(result, root / "models" / ("run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
                    st.success(f"Saved to models/{folder.name}")
                except (OSError, ValueError) as exc:
                    st.error(f"Could not save this run: {exc}")
        with method_tab:
            st.write("1. Validate daily OHLCV and compute trailing technical indicators. Earlier data initializes indicators; only the selected lookback is fitted.")
            st.write("2. Split in date order: 60% training, 20% validation, 20% test. Purge labels that reach the next partition. Fit scalers on training only.")
            st.write("3. Select direction by validation macro F1 and price by validation return error, with simple baselines eligible to win.")
            st.write("4. Evaluate frozen models on test. Refit separate deployment copies on resolved labels for the unknown next-session estimates.")
            st.caption("The regressor and classifier can disagree. Probabilities are uncalibrated; historical mean error is not a future uncertainty interval. A single short, dependent holdout limits conclusions.")
        with data_tab:
            st.json(result["metadata"].get("dataset_source", {}))
            st.json(result["metadata"]["audit"])
            st.dataframe(result["bars"].tail(30), hide_index=True, width="stretch")
            st.caption("No interpolated prices. Quote time, daily input date and forecast horizon are distinct. The data hash and split dates are included in the bundle.")


def render_price_evaluation(result):
    forecast = result["price_forecast"]
    prefix, unit = price_unit(result)
    selected = forecast["selected_model"]
    metrics = forecast["test_metrics"][selected]
    baseline = forecast["test_metrics"]["No price change"]
    columns = st.columns(3)
    columns[0].metric("Average price error", f"{prefix}{metrics['price_mae']:.2f}",
                      help="Mean absolute error (MAE) on held-out next-session closes. Lower is better; this is not a future error bound.")
    columns[1].metric("No-change baseline error", f"{prefix}{baseline['price_mae']:.2f}",
                      help="Historical error if every next close were predicted to equal the input close.")
    columns[2].metric("Test sessions", str(metrics["observations"]),
                      help="Chronologically held-out observations. They were not used to fit or select the evaluated model.")
    st.caption(f"{selected} · frozen historical test model · prices in {unit}")
    table = forecast["test_predictions"][selected]
    figure = price_backtest_chart(table)
    figure.update_layout(uirevision="price-test-" + result["metadata"]["cleaned_data_sha256"])
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})
    st.caption("Both lines are historical: estimates were generated without fitting those test rows. Dates are outcome dates, not forecast-input dates.")
    with st.expander("Compare price models & inspect predictions"):
        comparison = pd.DataFrame(forecast["test_metrics"]).T
        comparison["validation_return_mae"] = pd.Series(forecast["validation_return_mae"])
        comparison = comparison.rename(columns={"price_mae": f"Test price MAE ({unit})", "return_mae": "Test return MAE",
                                                 "observations": "Test sessions", "validation_return_mae": "Validation return MAE"})
        st.dataframe(comparison.style.format({f"Test price MAE ({unit})": "{:.2f}", "Test return MAE": "{:.2%}",
                                              "Validation return MAE": "{:.2%}", "Test sessions": "{:.0f}"}), width="stretch")
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button("Download price test predictions ↓", table.to_csv(index=False), "price_test_predictions.csv", "text/csv")


@st.fragment(run_every="60s")
def live_workspace(symbol: str, daily_result=None, live=True, detailed=False):
    snapshot = None
    if live:
        controls = st.columns([3, 1])
        with controls[0]:
            auto = st.toggle("Auto-refresh quote · 60 seconds", value=True, key=f"auto_quote_{symbol}",
                             help="Refreshes the latest supplied completed bar. Historical models change only when completed daily data changes.")
        with controls[1]:
            refresh = st.button("Refresh price", key=f"refresh_{symbol}", width="stretch")
        quote_key = f"last_quote_{symbol}"
        if refresh:
            cached_intraday.clear(symbol)
        if auto or refresh or quote_key not in st.session_state:
            try:
                snapshot = cached_intraday(symbol)
                st.session_state[quote_key] = snapshot
            except ValueError as exc:
                st.warning(str(exc))
        else:
            snapshot = st.session_state[quote_key]
    if daily_result is None:
        st.info("Completed daily history is needed to calculate estimates. Retry the history request or choose Upload / local CSV.")
        return
    result = daily_result
    if live:
        try:
            refreshed = cached_daily(symbol)
            if hashlib.sha256(refreshed["content"]).hexdigest() != result.get("source_hash"):
                st.rerun(scope="app")
        except ValueError:
            st.caption("Daily-history refresh is unavailable. Forecasts retain the input date shown below.")
    bars, forecast, direction = result["bars"], result["price_forecast"], result["daily_deployment"]
    prefix, unit = price_unit(result)
    quality = model_assessment(result)
    if snapshot is not None:
        latest = snapshot["bars"].iloc[-1]
        quote = float(latest.Close)
        available_at = latest.Date + pd.Timedelta(minutes=5)
        age = max(0, (utc_now() - available_at).total_seconds() / 60)
        status = "Recent supplied quote" if age <= 15 else "Latest session / delayed"
        quote_time = f"{available_at:%d %b %Y · %H:%M} IST"
        tone = "good" if age <= 15 else "caution"
        quote_label = "Latest available price"
    else:
        quote = float(bars.Close.iloc[-1])
        status, tone = ("Quote unavailable", "caution") if live else ("Historical dataset", "neutral")
        quote_time = f"{bars.Date.iloc[-1]:%d %b %Y} · daily close"
        quote_label = "Last completed daily close"
    st.markdown(f'<div class="context-strip"><span class="pill {tone}">● {status}</span>'
                f'<span>Quote: <b>{quote_time}</b></span><span>Forecast input: <b>{bars.Date.max():%d %b %Y}</b></span>'
                '<span>Horizon: <b>next trading session</b></span></div>', unsafe_allow_html=True)
    cards = st.columns(4)
    cards[0].metric(quote_label, f"{prefix}{quote:,.2f}", help=f"Price in {unit}. Free provider quotes may be delayed. If no quote is available, the historical daily close is explicitly labelled.")
    cards[1].metric("Next-session price estimate", f"{prefix}{forecast['estimated_price']:,.2f}",
                    f"{forecast['estimated_return']:+.2%} vs daily input close",
                    help="A separate refitted regression model estimates the next close from completed daily inputs. This is not a guaranteed target.")
    cards[2].metric("Next-session direction estimate", direction["direction"],
                    help="A separate refitted classifier estimates Bullish for a positive next-session return, Bearish for zero or negative. It may disagree with the price regressor.")
    cards[3].metric("Historical risk", result["risk_summary"]["risk_class"],
                    help="A configurable rule based on rolling historical annualized volatility, not a learned prediction of future losses.")
    left, right = st.columns([2.8, 1.15], gap="medium")
    with left, st.container(border=True):
        months = result["metadata"]["lookback_months"]
        window = f"{months}-month history" if months else "Historical price"
        st.subheader(f"{symbol.removesuffix('.NS')} · {window}")
        st.caption(f"{bars.Date.min():%d %b %Y} – {bars.Date.max():%d %b %Y} · {len(bars):,} sessions · {unit}")
        chart_controls = st.columns([2, 1])
        with chart_controls[0]:
            chart_style = st.radio("Historical chart", ["Line", "Candlestick"], horizontal=True, key=f"history_style_{symbol}", label_visibility="collapsed")
        with chart_controls[1]:
            indicators = st.toggle("Moving averages", value=True, key=f"averages_{symbol}")
        figure = price_chart(result["features"], ["sma_20", "sma_50"] if indicators else [], kind=chart_style)
        figure.update_layout(uirevision=f"history-{symbol}-{months}")
        st.plotly_chart(figure, width="stretch", config={"displaylogo": False})
    with right, st.container(border=True):
        st.markdown('<div class="mini-label">HISTORICAL EVIDENCE</div>', unsafe_allow_html=True)
        for title, value in [("Price model", quality["price_status"]), ("Direction model", quality["direction_status"])]:
            css = "good" if value in ("Lower test error", "Higher test F1") else "caution"
            st.markdown(f'<div class="quality-row"><span>{title}</span><span class="pill {css}">{value}</span></div>', unsafe_allow_html=True)
        st.caption("Comparisons describe this historical holdout; they do not establish future reliability.")
        st.divider()
        st.markdown(f'<div class="model-label">PRICE ESTIMATOR</div><div class="model-name">{escape(forecast["selected_model"])}</div>'
                    f'<div class="model-label">DIRECTION CLASSIFIER</div><div class="model-name">{escape(result["prediction"]["selected"])}</div>', unsafe_allow_html=True)
        st.caption(f"Bullish model output: {direction['bullish_probability']:.1%}. Uncalibrated; not a confidence level.")
        if quality["small_holdout"]:
            st.markdown(f'<div class="sample-note">Small holdout · {quality["test_observations"]} sessions<br>Consider a longer history to inspect more test observations.</div>', unsafe_allow_html=True)
        with st.expander("Read these estimates"):
            st.write("Both estimates use the completed daily input date above. A newer intraday quote does not become an unfinished daily training row. Classifier and regressor estimates can disagree.")
            st.caption("Evaluated models remain frozen. Separate deployment copies refit resolved labels after evaluation; their next-session outcomes are unknown.")
    if detailed:
        st.subheader("Historical price estimates versus observations")
        render_price_evaluation(result)
    else:
        columns = st.columns(3)
        risk = result["risk_summary"]
        with columns[0], st.container(border=True):
            st.subheader("Historical downside")
            st.metric("Largest peak-to-trough decline", f"{risk['maximum_drawdown']:.1%}", help="Maximum drawdown within the selected daily history, shown as a negative return.")
            figure = series_chart(result["risk_frame"], "drawdown", "", True)
            figure.update_layout(height=135, margin=dict(l=0, r=0, t=5, b=5))
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
        with columns[1], st.container(border=True):
            st.subheader("Unusual sessions")
            later = result["anomalies"].query("period == 'Later history (out-of-sample)'")
            flagged = later.loc[later.anomaly].sort_values("Date", ascending=False)
            st.caption(f"{len(flagged)} outliers / {len(later)} later observations · Isolation Forest")
            if flagged.empty:
                st.write("No later observation exceeds the training threshold.")
            for row in flagged.head(3).itertuples():
                st.markdown(f"**{row.Date:%d %b %Y}** · {row.daily_return:+.2%}")
                st.caption(f"Relative volume {row.volume_relative_20:.2f}× · score {row.anomaly_score:.3f}")
        with columns[2], st.container(border=True):
            st.subheader("Evaluation at a glance")
            selected = result["prediction"]["test_metrics"][result["prediction"]["selected"]]
            st.metric("Direction test accuracy", f"{selected['accuracy']:.1%}", help="Correct held-out predictions divided by the test sample size. Compare the baseline on Model Evaluation.")
            st.caption(f"Macro F1 {selected['f1_macro']:.3f} · {selected['n']} test sessions")
            error = forecast["test_metrics"][forecast["selected_model"]]["price_mae"]
            st.write(f"Average historical price error: **{prefix}{error:.2f}**")
            st.caption("Open Price & Direction for the held-out price chart and full regression comparison.")
    render_exports(result)
    st.markdown('<div class="footer"><span>↗ MarketSense · Historical evidence, clearly presented.</span><span>Educational research · No trade execution</span></div>', unsafe_allow_html=True)
