# ML methodology and equations

## Prediction timestamp and label

Observation t becomes available after the supplied day's close, including that day's volume. Forecast the next observed trading close: `r[t+1] = Close[t+1]/Close[t] - 1`, `y[t] = 1 if r[t+1] > 0 else 0`. Zero belongs to Bearish by the binary convention, not an economic assertion. The final target is missing, not zero. `LabelDate` records when the target can be known.

Only `FEATURE_COLUMNS` enter the classifier; Date, next return, label date and target are never inputs. Features use trailing windows and `ewm(adjust=False)`. The 50-bar SMA determines the normal 49-row warm-up. RSI uses recursive alpha=1/14 smoothing seeded from the first change; this may differ slightly from libraries seeded by a 14-day simple average. Flat prices produce RSI 50 and gains without losses produce RSI 100. MACD is EMA12 minus EMA26, signal EMA9 of MACD. Momentum is the 10-observation percentage price change. All ratios, returns and volatility use decimals internally.

## Chronology and selection

1. Build causal features on the complete chronological series; earlier history may legitimately provide context for later holdouts.
2. Create next-close labels separately. Remove warm-up and unknown-label rows for supervised evaluation.
3. Allocate 60% training, 20% validation, 20% test in date order.
4. Purge rows at the first two boundaries whenever `LabelDate >= next_partition_start`. Under a one-step horizon this removes one row per boundary.
5. Fit all fixed candidate models only on training. Logistic Regression's StandardScaler lives in its Pipeline and never fits validation/test.
6. Select highest validation macro-F1. Include the majority baseline; it can be the honest selected model. No hyperparameter search and no refitting are performed.
7. Evaluate all models on test for the final comparison. Do not change the model after reading test scores. A future experiment must reserve a fresh holdout or clearly disclose test reuse.
8. Predict from the newest feature-complete bar with the selected training-period model. Its outcome is unknown and its training regime can be stale. Uploading another CSV creates a new experiment, rather than evaluating it against a previously saved frozen model.

Simple holdout validation is time-aware, but it measures only one regime. Expanding-window validation, with label purging inside each fold, is a future extension. Random K-fold is unsuitable here. A one-step purge prevents label overlap but does not eliminate serial dependence between adjacent feature windows.

## Classifiers and metrics

The majority baseline tests whether the learned approach improves over class imbalance. Logistic Regression is a readable linear baseline (scaled inputs, max_iter 2000). Random Forest models nonlinear interactions (200 trees, depth 5, leaf size 10). Gradient Boosting adds shallow trees sequentially (100 trees, depth 2, learning rate .05, leaf size 10). All seeds are 42; tree estimators use one worker for predictable local resource use.

Accuracy is correct predictions / observations. Precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2PR/(P+R). Macro scores average class-specific scores equally. Zero denominators use zero and explicit class supports reveal absent classes. Confusion matrices have actual classes on rows and predicted classes on columns, always `[Bearish, Bullish]`. Model probabilities are not calibrated. Do not interpret 60% model probability as empirically verified 60% certainty.

Feature importance is absolute standardized coefficients for Logistic Regression, or impurity reduction for the tree models. Correlation, feature scale choices, and biased impurity importances limit interpretation. Baseline importance is unavailable.

## Historical risk

For close-to-close returns r:

- Daily volatility: sample standard deviation `std(r, ddof=1)`.
- Annualized volatility: daily volatility × `sqrt(252)`. The default assumes daily trading observations; it is inappropriate for intraday or irregularly sampled data.
- Drawdown[t]: `Close[t]/max(Close[0:t]) - 1`. Maximum drawdown is the most negative value, reported as a negative return.
- Historical one-day 95% VaR: `max(0, -quantile(r, .05))`, reported as a positive loss fraction, using pandas' default linear interpolation. This is not expected shortfall or a maximum loss guarantee.
- Annualized downside deviation: `sqrt(mean(min(r,0)^2)) * sqrt(252)`. Denominator includes all valid returns; target return is zero.

Rolling versions use the configurable observation window (default 20) and require a full window. Drawdown is expanding from the first supplied price. Classification uses **latest rolling annualized volatility**, whereas full-period metrics summarize all supplied history. Low <.15, Moderate [.15,.30), High ≥.30 by default. These are explainable project assumptions, not validated universal thresholds. Changing the window or thresholds changes the historical classification.

## Anomaly detection and paper alignment

Primary paper: Fei Tony Liu, Kai Ming Ting and Zhi-Hua Zhou, **Isolation Forest**, ICDM 2008, pp. 413–422, DOI [10.1109/ICDM.2008.17](https://doi.org/10.1109/ICDM.2008.17). [Author-hosted paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf).

The paper's central method isolates observations using random feature/split trees. Shorter average isolation paths indicate unusual observations. We apply scikit-learn's IsolationForest with 200 trees, subsampling (`max_samples='auto'`, up to 256 observations), seed 42 and training contamination .03. See the [official estimator documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).

Fit only feature-complete training observations, using returns, rolling volatility, volume change, relative volume, momentum and price range. Apply the same fitted detector to subsequent observations. Distinguish fitted training rows, earlier rows not used in fitting, and later out-of-sample rows. The score shown is **negative decision_function**, so positive means flagged; it is neither a probability nor the paper's raw score. Training contamination sets a threshold and does not enforce a 3% later outlier rate.

Top two absolute z deviations from training means provide context. Constant training indicators have undefined z scores and are omitted. These statistics do not explain the tree's decisions causally. No true anomaly labels exist in this dataset, so classification accuracy/F1/AUROC for anomalies cannot be honestly reported. A shock test verifies software sensitivity only. This is an application of the paper's methodology, **not reproduction of its published benchmark results**. Merlion is an optional future framework, not a dependency.

## Limits on conclusions

Financial labels are noisy, serially dependent and nonstationary. A single stock/time range supports limited generalization. Majority-class comparison is necessary but insufficient evidence of forecasting value. Retrospective adjusted prices, dataset selection, stale training, missing sessions, repeated experiments, and no independent anomaly labels remain limitations. There is no backtested trading strategy, transaction cost model, significance analysis or evidence of profitable trading. MarketSense reports descriptive historical risk and statistical predictions for teaching.
