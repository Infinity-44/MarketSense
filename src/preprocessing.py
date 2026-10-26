"""Conservative OHLCV cleaning: no interpolation or backward filling."""
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def validate_ohlcv(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return a single instrument's daily bars and an auditable cleaning report.

    Missing/invalid bars are removed; conflicting dates raise instead of silently
    selecting a stock or a price. Identical repeated bars are safe to collapse.
    """
    if raw.empty:
        raise ValueError("The CSV contains no observations.")
    frame = raw.copy()
    frame.columns = [str(c).strip().title() for c in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names after normalization.")
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    for column in ("Symbol", "Ticker", "Name"):
        if column in frame and frame[column].nunique(dropna=False) > 1:
            raise ValueError("Provide exactly one stock per CSV; filter the symbol first.")
    frame = frame[REQUIRED_COLUMNS].copy()
    count = len(frame)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    for column in REQUIRED_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    missing_mask = frame.isna().any(axis=1)
    missing_count = int(missing_mask.sum())
    frame = frame.loc[~missing_mask].copy()
    valid = ((frame[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
             & (frame["Volume"] >= 0)
             & (frame["High"] >= frame[["Open", "Low", "Close"]].max(axis=1))
             & (frame["Low"] <= frame[["Open", "High", "Close"]].min(axis=1)))
    invalid_count = int((~valid).sum())
    frame = frame.loc[valid].copy()
    duplicates = int(frame.duplicated().sum())
    frame = frame.drop_duplicates()
    if frame["Date"].duplicated().any():
        raise ValueError("Conflicting observations share a date. Use one daily bar per stock.")
    frame = frame.sort_values("Date").reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError("Fewer than two valid OHLCV observations remain after cleaning.")
    return frame, {
        "input_rows": count, "clean_rows": len(frame),
        "missing_or_nonfinite_rows_removed": missing_count,
        "invalid_price_or_volume_rows_removed": invalid_count,
        "exact_duplicates_removed": duplicates,
        "policy": "Drop invalid bars; collapse exact duplicates; reject conflicting dates; no imputation.",
    }
