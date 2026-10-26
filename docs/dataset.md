# Dataset acquisition and provenance

## Recommended source

- Publisher: Boris Marjanovic, **Huge Stock Market Dataset**.
- Source: https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/
- Publisher-listed license: **CC0 / Public Domain**.
- Data card: daily US stock/ETF OHLCV, adjusted for dividends and splits, last updated November 2017. Treat it as historical research data, not current quotes.
- Expected file: `Stocks/aapl.us.txt` (or another sufficiently long single-instrument file). Extra `OpenInt` is ignored.

Open the dataset page, use Download, and extract the archive. A free account may be necessary. Copy only the selected instrument file into `data/raw/`; the complete archive is large and is unnecessary for one experiment. Alternatively upload that file directly in the sidebar. No API credentials are needed by MarketSense. A direct single-file endpoint attempted during implementation returned HTTP 404, so manual download is the reliable documented route.

Record the actual filename, stock, original download date, dataset version, adjustment policy, and source link in your report. Retain the original file locally. Each saved run records the cleaned-data SHA-256 and date range; also hash the original file if required by your supervisor. Do not confuse the date of a historical price with today's date.

## Input contract

One daily instrument per UTF-8 comma-separated file, with `Date,Open,High,Low,Close,Volume`. Column names are whitespace/case normalized. Dates should use ISO `YYYY-MM-DD`. At least 301 valid bars are needed under defaults; five or more years gives more useful holdouts. Currency is not inferred from a filename, so charts use source units.

The loader converts dates and numerics; removes missing/nonfinite rows, nonpositive prices, negative volumes and impossible OHLC ranges; collapses exact duplicate bars; rejects conflicting repeated dates; and sorts chronologically. If Symbol/Ticker/Name identifies multiple instruments it rejects the file. Mixed instruments without identifiers may be impossible to recognize: the user must supply one stock. Every removal is counted in an audit.

There is no interpolation, backward filling or fabricated weekend data. Holidays stay absent. A dropped invalid bar can make the next observed close span more than one trading session; inspect the audit before interpreting the target as strictly next-day. Zero volume is allowed; its percentage change from a zero denominator is encoded as zero, with relative volume preserved. An all-zero-volume asset has no useful volume signal.

## Adjustment caveats

Use consistently adjusted Open/High/Low/Close. Do not replace only Close with Adjusted Close while leaving raw OHLC. Splits, dividends, stale prices, vendor errors, and retrospective adjustment can alter apparent returns and anomalies. The dataset is not a point-in-time universe: survivorship and selection bias can remain. A fixed vintage is reproducible but does not prove modern deployment performance.

Other legally reusable CSV sources work if they meet the contract. Record their actual license and adjustment policy; do not assume all free-to-view financial data permits redistribution. Raw data is excluded from Git by default. Without acquired real data, no financial benchmark or anomaly-accuracy claim is made.
