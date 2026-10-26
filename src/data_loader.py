"""CSV ingestion, independent of Streamlit and any paid service."""
import logging
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from src.preprocessing import validate_ohlcv

logger = logging.getLogger(__name__)


def load_csv(source: str | Path | BinaryIO) -> tuple[pd.DataFrame, dict]:
    try:
        raw = pd.read_csv(source)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError("Unable to read CSV. Supply a UTF-8, comma-separated OHLCV file.") from exc
    cleaned, audit = validate_ohlcv(raw)
    logger.info("Loaded %s valid bars from %s input rows", len(cleaned), audit["input_rows"])
    return cleaned, audit
