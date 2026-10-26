"""Download a real, publicly hosted example; never fabricate demo observations.

The source is the Apple OHLCV example used in Plotly's official documentation.
Downloaded data stays in ignored local storage rather than being redistributed.
"""
import hashlib
import io
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd

from src.preprocessing import validate_ohlcv

EXAMPLE_URL = "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv"
EXAMPLE_DOCS = "https://plotly.com/python/candlestick-charts/"
EXAMPLE_FILENAME = "aapl_example.csv"


def normalize_example(content: bytes) -> bytes:
    raw = pd.read_csv(io.BytesIO(content))
    raw = raw.rename(columns={f"AAPL.{column}": column for column in
                              ("Open", "High", "Low", "Close", "Volume")})
    bars, audit = validate_ohlcv(raw)
    if len(bars) < 301 or audit["invalid_price_or_volume_rows_removed"]:
        raise ValueError("The public example failed OHLCV validation.")
    return bars.to_csv(index=False, date_format="%Y-%m-%d").encode()


def load_example(cache_directory: Path) -> bytes:
    destination = cache_directory / EXAMPLE_FILENAME
    if destination.exists():
        return destination.read_bytes()
    try:
        with urlopen(EXAMPLE_URL, timeout=15) as response:
            original = response.read(2_000_001)
    except (URLError, TimeoutError, OSError) as exc:
        raise ValueError("The example could not be downloaded. Check your connection or upload a CSV.") from exc
    if len(original) > 2_000_000:
        raise ValueError("Unexpected example size. Please upload a CSV instead.")
    content = normalize_example(original)
    cache_directory.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    destination.with_suffix(".source.json").write_text(json.dumps({
        "source": EXAMPLE_URL, "documentation": EXAMPLE_DOCS,
        "source_sha256": hashlib.sha256(original).hexdigest(),
        "description": "Historical AAPL example; unadjusted OHLC, USD, not live quotes.",
    }, indent=2), encoding="utf-8")
    return content
