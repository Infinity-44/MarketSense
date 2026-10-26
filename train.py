"""Example: python train.py --data data/raw/aapl.us.txt --symbol AAPL"""
import argparse
from datetime import datetime, timezone
import logging

from config.settings import Settings
from src.analysis import run_analysis, save_analysis
from src.data_loader import load_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Train MarketSense on a single stock's OHLCV CSV.")
    parser.add_argument("--data", required=True, help="CSV or Kaggle comma-separated .txt file")
    parser.add_argument("--symbol", default="Selected stock")
    parser.add_argument("--output", help="New output directory; existing directories are protected")
    parser.add_argument("--risk-window", type=int, default=20)
    parser.add_argument("--low-risk", type=float, default=0.15)
    parser.add_argument("--high-risk", type=float, default=0.30)
    parser.add_argument("--contamination", type=float, default=0.03)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        settings = Settings(risk_window=args.risk_window, low_risk_limit=args.low_risk,
                            high_risk_limit=args.high_risk, contamination=args.contamination)
        bars, audit = load_csv(args.data)
        result = run_analysis(bars, settings, args.symbol, audit)
        output = args.output or "models/run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        folder = save_analysis(result, output)
    except (ValueError, OSError) as exc:
        logging.error("%s", exc)
        return 1
    print(f"Saved run: {folder}")
    print(f"Validation-selected model: {result['prediction']['selected']}")
    for name, metrics in result["prediction"]["test_metrics"].items():
        print(f"{name}: held-out accuracy={metrics['accuracy']:.3f}, macro-F1={metrics['f1_macro']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
