# Implementation record

The workspace was empty at inspection. The architecture, data strategy, ML protocol and major leakage/data risks were explained before coding. Development proceeded in the following phases.

| Phase | Objective and completed work | Check / command | Remaining dependency |
|---|---|---|---|
| 1 · setup/loading | Config, dependencies, CSV schema contract | `python -m pytest tests/test_data.py -q` | Acquire a real stock CSV |
| 2 · preprocessing/EDA | Conservative cleaning, audit, notebook workflow | Same ingestion tests; run EDA notebook | Actual-data EDA needs acquisition |
| 3 · features | Causal indicators, unknown final label | `python -m pytest tests/test_features_prediction.py -q` | None in software |
| 4 · classification | Pipelines, purged 60/20/20, baseline, validation selection, test metrics | Same prediction tests; `python train.py --data data/raw/aapl.us.txt --symbol AAPL` | Real financial evaluation needs acquisition |
| 5 · risk | Historical/rolling risk, configurable categories | `python -m pytest tests/test_risk_anomalies.py -q` | Empirical interpretation on real data |
| 6 · anomalies | Training-only forest, later scores, indicator context | Same risk/anomaly tests | No ground-truth anomaly labels supplied |
| 7 · dashboard | Five sections, Plotly, downloads, save action, empty state | `streamlit run app.py`; AppTest in integration tests | Real-data demonstration |
| 8 · verification/docs | Pipeline round-trip, UI navigation, docs, GitHub CI, credential preflight | `python -m pytest -q`; `python scripts/check_secrets.py` | Remote CI runs only after publication |

Tests intentionally use deterministic synthetic fixtures to verify software behaviour. They do not establish financial-model performance. A fixture shock test only checks that extreme indicator values can be flagged. No dataset or benchmark scores are silently manufactured.

Local verification: **15 tests passed on Python 3.12**, including all dashboard sections. A Streamlit-internal pandas deprecation warning is present; it does not fail the checks. Python 3.11 and Windows execution are configured in the manual CI matrix but have not yet been verified remotely. The workflow does not run on push, avoiding automatic hosted compute usage. The tracked-file credential preflight passed before initial publication; it is a heuristic, not a guarantee of detecting all secrets.

The EDA notebook imports reusable source functions and contains no duplicated training implementation. CLI runs write new artifact directories rather than changing source data. All result charts and tables derive from the selected input. No screenshots with invented financial data are included.

## Reproduction checklist

Acquire the documented dataset, record its provenance, run the tests, run a named CLI experiment, inspect `report.json`, and open the dashboard on the same file/settings. Keep train/validation/test dates and hashes with the final report. Fix all modelling choices before inspecting held-out test scores. Save exact installed versions alongside the experiment if reproducing it on another machine.

## Repository publishing

The repository should contain only source, configuration, tests, the unexecuted notebook and docs. `.gitignore` excludes raw/processed data, trained model outputs, environments and common credential locations. Review `git diff --cached` and run the tracked-file credential preflight before pushing. The remote URL must contain no credentials. Use existing GitHub authentication via the connector or normal Git credential helper; never embed tokens in project files.
