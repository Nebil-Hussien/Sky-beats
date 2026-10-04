#!/usr/bin/env python3
"""
datapreprocessing_w8.py
-----------------------
RQ8-specific preprocessing: collapse the shared, general-processed song-level
dataset to **one row per day** (mean audio profile + that day's weather) and
attach the calendar features the models need: `month` (1-12, the baseline
predictor) and `season` (meteorological, for reporting).

Reuses the shared helper in `src/utils.py`:
  - aggregate_daily : collapse song-level rows to one row per date

Usage
-----
  python datapreprocessing_w8.py <input_csv> -o <output_csv>
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Make the shared src/utils.py importable when run as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from utils import aggregate_daily  # noqa: E402

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

SEASONS = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}


def setup_logger():
    """Configure logging to ./logs relative to this script (CWD-independent)."""
    log_path = Path(__file__).resolve().parent / "logs" / "datapreprocessing_w8.log"
    logging.basicConfig(filename=log_path, filemode="w", level=logging.DEBUG)
    return logging.getLogger(__name__)


def build_daily_features(df):
    """Aggregate to one row per day and attach month + season columns.

    `utils.aggregate_daily` only keeps the weather columns and audio features,
    so month and season are re-derived from the date afterwards.

    Args:
        df (pd.DataFrame): song-level rows with a 'date' column

    Returns:
        pd.DataFrame: one row per date, sorted by date, with audio profile,
            weather, month (int) and season (str)
    """
    daily = aggregate_daily(df).copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["month"] = daily["date"].dt.month
    daily["season"] = daily["month"].map(SEASONS)
    return daily


def main(input_csv, output_csv):
    setup_logger()
    LOGGER.info("Loading %s", input_csv)
    try:
        df = pd.read_csv(input_csv, parse_dates=["date"])
    except FileNotFoundError:
        raise SystemExit(f"ERROR: input file not found: {input_csv}")
    except pd.errors.EmptyDataError:
        raise SystemExit(f"ERROR: input file is empty: {input_csv}")

    required = {"date", "valence", "sunshine_hours",
                "precipitation_mm", "temperature_mean_c"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: input missing required columns: {sorted(missing)}")

    daily = build_daily_features(df)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_csv, index=False)

    LOGGER.info("Saved %d daily rows -> %s (%s to %s)", len(daily), output_csv,
                daily["date"].min().date(), daily["date"].max().date())
    print(f"Daily rows: {len(daily)} ({daily['date'].min().date()} to {daily['date'].max().date()})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ8 preprocessing: daily audio profile + month/season features")
    parser.add_argument("input_csv", help="shared general-processed song-level CSV")
    parser.add_argument("-o", "--output", required=True, help="output daily CSV")
    args = parser.parse_args()
    main(args.input_csv, args.output)
