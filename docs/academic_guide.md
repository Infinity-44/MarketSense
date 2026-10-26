# Academic report and viva guide

## Problem statement and objectives

Historical stock data contains price trends, changing volatility and unusual observations that are difficult to assess from raw tables. Build an understandable, reproducible application that predicts the next observed direction, measures historical risk, identifies statistical outliers and visualizes evidence without claiming investment certainty.

Objectives: validate a real OHLCV dataset; generate causal indicators; compare three supervised classifiers and a simple baseline; quantify interpretable risk; apply historical-training-only Isolation Forest; present held-out results and limitations through a local interactive dashboard.

## System design and workflow

Use the architecture diagram in README. The reusable Python modules contain the analysis, `src/analysis.py` orchestrates it, `train.py` exposes reproducible runs, and `app.py` presents them. The CSV contract avoids an API-key dependency. Data quality, feature calculations, splitting, fitting, selection, testing, saving and presentation remain individually explainable.

## EDA procedure

Use `notebooks/exploration.ipynb` after acquiring real data. Inspect date coverage, cleaning counts, volume zeros, descriptive statistics, close/return/volume plots and training-period feature correlations. Investigate large returns against corporate actions and source quality. Perform model-design EDA on the training period; avoid using the final test period to choose indicators or thresholds. Full-history plots may be shown as descriptive context, clearly labelled.

## Model comparison for the report

Run the pipeline once after fixing the methodology, then copy the genuine validation and test comparisons from the dashboard or saved JSON. Record the data hash, date periods and row counts with the table. Explain whether the baseline won; a weak model is a valid academic result if evaluation is honest. Include the selected model's confusion matrix and per-class support, rather than reporting accuracy alone. Do not fill the report with fixture scores.

## Research foundation

The feasible primary paper is Liu, Ting and Zhou's Isolation Forest (ICDM 2008). Implement random-tree isolation through the scikit-learn estimator and evaluate its application to held-out market observations. Discuss shorter isolation paths, subsampling, threshold selection, and distribution drift. Compare qualitative outlier context and score distributions across periods. Do not claim published benchmark reproduction or anomaly accuracy without labelled ground truth. See methodology.md for primary links and precise scope.

Historical volatility, drawdown and return-quantile VaR supply a descriptive risk foundation. Direction classification supplies a standard supervised time-series experiment. Explain these as complementary modules with different evaluation requirements; the risk category is a transparent rule, not another trained ML model.

## Four-person team allocation

| Member | Main responsibility | Review responsibility |
|---|---|---|
| 1 | Dataset provenance, loader, cleaning audit, EDA notebook | Check feature inputs and dates |
| 2 | Indicators, target, splits, classifiers and metrics | Review leakage tests and selection |
| 3 | Risk formulas, Isolation Forest and score context | Check mathematical conventions |
| 4 | Streamlit, Plotly, integration, CI and report assembly | Verify visible timestamps and caveats |

All four should run the CLI and explain one end-to-end observation. Use small branches/PRs; each PR gets a teammate review. Keep datasets and artifacts local. Share reproducible run metadata and agreed data-source/version instructions rather than credentials.

## Demonstration sequence

1. Start the dashboard and load a real stock file; show the source and date coverage.
2. Explain the cleaning audit and after-close prediction timestamp.
3. Show historical indicators and distinguish unknown latest inference from held-out test predictions.
4. Show fixed chronological partitions, the majority baseline and selection on validation macro-F1.
5. Explain the test confusion matrix, class balance and uncalibrated probabilities.
6. Show historical risk formulas and change a risk threshold transparently.
7. Show later-period anomaly markers and the training-fit cutoff; inspect indicator context.
8. Save a run, inspect metadata, and show leakage/formula tests.

## Viva questions to prepare

- Why random shuffling leaks time-series information, and why the last row cannot be labelled yet.
- Why features can use the current close only when prediction happens after that close.
- Why a boundary row is purged even when its inputs are historical.
- Why the scaler fits training only and the baseline is allowed to win.
- Why macro-F1 and class supports matter when Bullish labels dominate.
- Why 55% test accuracy does not establish profitable prediction.
- How daily volatility differs from annualized volatility and drawdown.
- Why historical VaR can be exceeded and why downside deviation uses a zero target.
- Why statistical outliers are not necessarily fraud, and contamination is not an accuracy measure.
- How random isolation trees relate to the selected paper and what was not reproduced.

## Limitations and future scope

Current scope: binary daily direction and next-session price regression, one instrument at a time, fixed holdouts/parameters, a free provider feed with possible delay, no calibrated probabilities, no deployment service, no anomaly ground truth, and historical teaching risk bands. Dataset vintage, corporate-action adjustments, survivorship bias, serial dependence, regime changes and repeated test inspection limit conclusions.

Future work: expanding-window validation with purging; multiple instruments with per-symbol features and careful grouped time splits; stability checks across regimes; probability calibration using training-only time folds; explicit Neutral thresholds based on a documented economic/noise rationale; walk-forward retraining; persistent frozen-model inference; LOF novelty detection; and independently labelled anomaly benchmarks. Trading backtests and paid data integration are separate projects with additional assumptions.
