# NSE history-based price and direction estimates

The default source is **NSE · live + historical**. Choose a stock and a **3-month, 5-month, 6-month or 1-year historical lookback**. That window controls both the price chart and the rows used for model training and evaluation. The app displays the latest supplied NSE price, a next-session closing-price estimate and a separate Bullish/Bearish next-session direction estimate.

## Free local implementation

The app uses [yfinance](https://github.com/ranaroussi/yfinance) and Yahoo Finance's public feed without API keys, subscriptions or billing configuration. Yahoo's personal/research-use terms apply. [Streamlit is free and open-source](https://github.com/streamlit/streamlit). No broker, order, paid API or paid hosting is integrated. Hosted CI is manual-only. The free provider may delay, rate-limit or fail requests; this is not an exchange-grade tick feed.

## Historical modelling window

Fetch sufficient earlier daily history to initialize trailing features such as SMA 50, then filter the feature-complete rows to the chosen number of months. **Earlier rows warm up indicators only and are excluded from model fitting and evaluation.** A 3-month window typically has about 60 trading sessions and very few held-out observations. The live mode permits at least 50 labelled rows; the standalone full-history CSV workflow retains its default 250-label minimum. Short-window results are limited experiments, not evidence of reliable financial forecasting.

Keep exchange dates in Asia/Kolkata. A current-day daily bar is eligible only when provider metadata identifies that same date's actual regular-session end and another 15 minutes has elapsed. Otherwise it is excluded conservatively. Prices consistently use the provider's auto-adjusted OHLC; adjusted closes are not mixed with raw high/low/open.

Use chronological 60/20/20 train/validation/test partitions and purge next-close labels that reach the following partition. Fit every scaler and evaluated model on training only. No design choice is changed after looking at the final test results.

## Two separately evaluated estimates

**Direction:** compare majority baseline, scaled Logistic Regression, Random Forest and Gradient Boosting using validation macro F1. Report held-out accuracy, class support, macro precision/recall/F1 and confusion matrices. After evaluation, a separate deployment clone refits all resolved daily labels inside the chosen window and estimates direction from the latest completed feature row. Probabilities are uncalibrated.

**Price:** predict the next daily return using a no-price-change baseline, scaled Ridge regression and Random Forest regression. Select by lowest validation return MAE. Evaluate frozen models on test, including price MAE = mean absolute difference between true next close and `current_close * (1 + predicted_return)`. A separate deployment clone then refits resolved labels and estimates `latest_completed_close * (1 + estimated_return)`. The baseline can win; a no-change estimate is a valid result and is labelled honestly.

The classifier and regressor are different models and may disagree. A next-session estimate has an unknown outcome and no guaranteed accuracy. MAE is historical average error, **not** a future prediction interval. Reported test scores describe frozen historical models, not measured accuracy of deployment clones refitted after evaluation.

## Latest price versus forecast inputs

The latest available price comes from the last completed five-minute bar and refreshes every 60 seconds, using a 55-second feed cache. A running bar is excluded. The UI shows the bar completion date/time in IST and flags age greater than 15 minutes as possible market closure or delay. Five-minute bars supply the latest price only: **they do not define a prediction horizon or replace the selected months of daily training data.**

Both forecasts are based on the daily input date shown in the UI. During the session, an intraday price can be newer than that input date. The app does not pass an unfinished daily candle into a daily-close model or imply that the estimate is calculated from the entire day's still-unknown OHLCV. Daily history caches for 30 minutes, has a manual refresh, and triggers recomputation when a changed history snapshot is detected. No synthetic or sample prices replace a failed live feed. If only the latest-price request fails, the last historical daily close is explicitly labelled as such.

## Actual verification

On 26 October 2026, the provider returned **1,236 daily** and **1,394 five-minute RELIANCE.NS observations**. The 5-month modelling window contained **105 daily sessions** (26 May–26 October); the 3-month window contained **63**. These came from actual requests, not fixtures.

For the 5-month price experiment, validation selected Ridge regression. It produced a next-session estimate of approximately **₹1,201.56** from an input close of **₹1,201.80**. Its historical test MAE was **₹16.42** over **21 sessions**, compared with **₹15.77** for no price change. This weak result is shown rather than hidden, and it does not guarantee the next price. The selected direction classifier estimated Bearish. All results are snapshot-specific and will change with the selected stock, window and provider data.

## Modules and storage

- `src/live_data.py`: keyless NSE provider adapter, symbol validation, IST date/bar handling.
- `src/analysis.py`: causal indicator context, selected-month filtering, purged daily classification/risk/anomaly analysis and artifacts.
- `src/price_prediction.py`: price-model comparison, MAE and separate deployment regression.
- `src/live_dashboard.py`: cached feeds, current quote, historical chart, price/direction cards and exports.
- `app.py`: stock/lookback selectors and original analysis sections.

Provider cookies/timezone caches remain under ignored `.cache/yfinance/`. Downloaded observations and saved artifacts stay in ignored data/models directories. Code contains no provider credentials. Saved live analyses include historical classifier/regressor models, deployment clones, metadata and both forecasts. Tests cover selected-window fitting, indicator context, training-only scaling, validation selection, MAE conventions, provider timestamps, partial bars, UI paths, failure states and artifact round-trips.
