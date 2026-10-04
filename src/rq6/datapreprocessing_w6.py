#!/usr/bin/env python3
"""
datapreprocessing_w6.py
-----------------------
RQ6-specific preprocessing: aggregate the shared, general-processed dataset to a
**daily audio profile** (mean of each audio feature + that day's weather) and tag
each day as `gloomy` / `clear` / `other` using percentile cut-points of the daily
sunshine and precipitation distributions.

Reuses the shared helpers in `src/utils.py`:
  - aggregate_daily     : collapse song-level rows to one row per date
  - label_gloomy_clear  : percentile-based gloomy/clear/other labelling

Usage
-----
  python datapreprocessing_w6.py <input_csv> -o <output_csv>
      [--sun-low-pct 25 --sun-high-pct 75 --precip-low-pct 25 --precip-high-pct 75]
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Make the shared src/utils.py importable when run as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from utils import aggregate_daily, label_gloomy_clear  # noqa: E402

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


def setup_logger():
    """Configure logging to ./logs relative to this script (CWD-independent)."""
    log_path = Path(__file__).resolve().parent / "logs" / "datapreprocessing_w6.log"
    logging.basicConfig(filename=log_path, filemode="w", level=logging.DEBUG)
    return logging.getLogger(__name__)


def build_daily_profile(df, sun_low_pct, sun_high_pct, precip_low_pct, precip_high_pct):
    """Aggregate to one row per day (audio profile + weather), then tag day_type."""
    daily = aggregate_daily(df)
    daily = label_gloomy_clear(
        daily,
        sun_low_pct=sun_low_pct, sun_high_pct=sun_high_pct,
        precip_low_pct=precip_low_pct, precip_high_pct=precip_high_pct,
    )
    return daily


def main(input_csv, output_csv, sun_low_pct, sun_high_pct, precip_low_pct, precip_high_pct):
    setup_logger()
    LOGGER.info("Loading %s", input_csv)
    try:
        df = pd.read_csv(input_csv, parse_dates=["date"])
    except FileNotFoundError:
        raise SystemExit(f"ERROR: input file not found: {input_csv}")
    except pd.errors.EmptyDataError:
        raise SystemExit(f"ERROR: input file is empty: {input_csv}")

    required = {"date", "sunshine_hours", "precipitation_mm"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: input missing required columns: {sorted(missing)}")

    daily = build_daily_profile(df, sun_low_pct, sun_high_pct, precip_low_pct, precip_high_pct)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_csv, index=False)

    counts = daily["day_type"].value_counts().to_dict()
    LOGGER.info("Saved %d daily rows -> %s (day_type counts: %s)", len(daily), output_csv, counts)
    print(f"Daily rows: {len(daily)} | day_type counts: {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ6 preprocessing: daily audio profile + gloomy/clear labels")
    parser.add_argument("input_csv", help="shared general-processed song-level CSV")
    parser.add_argument("-o", "--output", required=True, help="output daily labelled CSV")
    parser.add_argument("--sun-low-pct", type=float, default=25,
                        help="sunshine percentile at/below which a day is 'dark' (default 25)")
    parser.add_argument("--sun-high-pct", type=float, default=75,
                        help="sunshine percentile at/above which a day is 'sunny' (default 75)")
    parser.add_argument("--precip-low-pct", type=float, default=25,
                        help="precip percentile at/below which a day is 'dry' (default 25)")
    parser.add_argument("--precip-high-pct", type=float, default=75,
                        help="precip percentile at/above which a day is 'wet' (default 75)")
    args = parser.parse_args()
    main(args.input_csv, args.output, args.sun_low_pct, args.sun_high_pct,
         args.precip_low_pct, args.precip_high_pct)
