# MarketSense demonstration — 8–10 minutes

## Before the presentation

Use Python 3.11 or 3.12 and install `requirements.txt`. Start from the project root:

```bash
python -m pytest -q
python -m streamlit run app.py
```

Open http://localhost:8501. Check internet access once with **NSE · live + historical**, **RELIANCE** and **5 months**. Download the research bundle before presenting. Keep it as an honest backup if the provider is unavailable. The optional Apple example caches its real historical data locally; it is clearly labelled historical and does not represent current NSE prices. Avoid switching feeds repeatedly if Yahoo rate limits requests.

## Presentation sequence

| Time | Screen / action | What to say |
|---|---|---|
| 0:00–0:45 | Overview | “MarketSense combines historical direction classification, price regression, risk measurements and unsupervised anomaly detection.” |
| 0:45–1:30 | Select stock / 5 months | “The selected window controls training and evaluation rows. Older observations initialize causal indicators only.” |
| 1:30–2:30 | Quote, input-date and next-session cards | “The quote can be newer than the daily forecast inputs. The outcome of the next-session estimate is unknown. Free quotes can be delayed.” |
| 2:30–3:30 | Price & Direction | Show historical actual-versus-estimated closes. Explain MAE and compare the selected model with no price change. Both chart lines are held-out historical results. |
| 3:30–4:30 | Model Evaluation → Direction models | Show accuracy, precision, recall, macro F1, class support and confusion matrix. The validation metric selects the winner; test does not. |
| 4:30–5:15 | Model Evaluation → Price models | Compare regression models, validation return MAE and held-out price MAE. A baseline can legitimately win. |
| 5:15–6:00 | Market Behaviour | Show indicators. Explain that the displayed direction is the same deployment estimate as Overview; the held-out prediction table belongs to the earlier frozen model. |
| 6:00–6:45 | Risk Analysis | Explain historical volatility, drawdown and VaR. Risk categories are configurable rules, not another trained model. |
| 6:45–7:30 | Anomaly Detection | Inspect one flagged date and its training-statistic context. An outlier is not proof of manipulation. |
| 7:30–8:15 | Change 5 months → 3 months | Show date/row-count changes and explain that shorter histories leave very small holdouts. Do not select a window merely because its test score looks better. |
| 8:15–9:00 | Export analysis | Download the ZIP and readable Markdown summary; optionally save models locally. Show dates, versions and data hash in the report. |
| 9:00–10:00 | Tests and limitations | Mention causality, boundary purging, training-only preprocessing, forecast consistency, saved-model round-trips and failure-state checks. |

## Explain the models accurately

- **Direction:** Logistic Regression, Random Forest Classifier and Gradient Boosting Classifier versus a majority-class baseline. Select by validation macro F1.
- **Price:** Ridge and Random Forest regression versus no price change. Select by validation return MAE; convert predicted return into next-session close.
- **Anomalies:** training-only Isolation Forest, following Liu, Ting and Zhou's 2008 isolation methodology. This is an application of the algorithm, not reproduction of the paper's benchmark results.
- **Risk:** descriptive formulas and configurable thresholds.
- **Deployment:** separate clones refit resolved labels after the frozen historical evaluation. Their future performance has not been measured yet.

## If something unexpected happens

- **Market closed / old quote:** point to the timestamp. Refreshing cannot invent a new market observation.
- **Quote request fails:** the app explicitly labels the historical daily close; it can still show daily forecasts if history was fetched successfully.
- **History request fails:** retry or select the cached historical example / a genuine full-history CSV. Use the previously exported report to discuss the actual NSE experiment.
- **Baseline wins or model loses on test:** report it honestly. The project demonstrates an auditable ML experiment, not guaranteed financial predictability.
- **Price and direction disagree:** they are separately trained regression/classification models; their objectives and validation selection differ.

## Four-person speaking roles

1. Data source, preprocessing and causal indicators.
2. Classification/regression targets, chronological splits and evaluation.
3. Risk equations, Isolation Forest and research alignment.
4. Dashboard, exports, verification and limitations.

All four should understand the end-to-end flow and distinguish historical scores, current quotes and unknown future outcomes.
