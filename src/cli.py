import argparse
import sys
import pandas as pd
from pathlib import Path
from src.inference import SpaceRadiationForecaster
from src.live_service import run_live_forecast_cycle

def main():
    parser = argparse.ArgumentParser(
        description="Operational Space Weather Relativistic Electron Forecast CLI"
    )
    parser.add_argument(
        "--mode",
        choices=["live", "demo", "batch"],
        default="live",
        help="Operational mode: 'live' (NOAA SWPC stream), 'demo' (recent test window), or 'batch'"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to input Parquet or CSV file (required for batch mode)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="forecast_output.csv",
        help="Output file path for batch mode (default: forecast_output.csv)"
    )

    args = parser.parse_args()

    if args.mode == "live":
        run_live_forecast_cycle()
    elif args.mode == "demo":
        from src.inference import run_demo_inference
        run_demo_inference()
    elif args.mode == "batch":
        if not args.input or not Path(args.input).exists():
            print(f"Error: A valid input file is required for batch mode. Provided: {args.input}")
            sys.exit(1)
        forecaster = SpaceRadiationForecaster()
        df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)
        preds = forecaster.predict(df)
        preds.to_csv(args.output)
        print(f"[✓] Batch inference complete ({len(preds):,} records). Saved to {args.output}")

if __name__ == "__main__":
    main()
