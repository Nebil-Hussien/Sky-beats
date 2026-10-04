#!/usr/bin/env python3
"""
datapreprocessing_w9.py
-----------------------
RQ9-specific preprocessing (the reverse of RQ6): build the **daily audio
profile**, tag each day `gloomy` / `clear` / `other` using the SAME percentile
definition as RQ6, and then keep **only the gloomy and clear days** so the result
is a ready-to-train, two-class classification table.

The gloomy/clear labelling is shared with RQ6 through `utils.label_gloomy_clear`,
so the two workflows can never disagree on what "gloomy" means. Days that are
neither gloomy nor clear ("other") are the majority and are irrelevant to the
gloomy-vs-clear classification, so they are dropped here (the dropped count is
logged, never silently discarded).

Reuses the shared helpers in `src/utils.py`:
  - aggregate_daily     : collapse song-level rows to one row per date
  - label_gloomy_clear  : percentile-based gloomy/clear/other labelling

Usage
-----
  python datapreprocessing_w9.py <input_csv> -o <output_csv>
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
    log_path = Path(__file__).resolve().parent / "logs" / "datapreprocessing_w9.log"
    logging.basicConfig(filename=log_path, filemode="w", level=logging.DEBUG)
    return logging.getLogger(__name__)


def build_classification_table(df, sun_low_pct, sun_high_pct,
                               precip_low_pct, precip_high_pct):
    """Aggregate to daily rows, label gloomy/clear/other, keep only the two
    target classes.

    Percentiles are computed on the FULL daily distribution first, so the
    gloomy/clear cut-points are unaffected by the subsequent filtering.
    Returns the two-class table plus the count of dropped "other" days.
    """
    daily = aggregate_daily(df)
    daily = label_gloomy_clear(
        daily,
        sun_low_pct=sun_low_pct, sun_high_pct=sun_high_pct,
        precip_low_pct=precip_low_pct, precip_high_pct=precip_high_pct,
    )

    n_total = len(daily)
    two_class = daily[daily["day_type"].isin(["gloomy", "clear"])].reset_index(drop=True)
    n_dropped = n_total - len(two_class)
    return two_class, n_dropped


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

    two_class, n_dropped = build_classification_table(
        df, sun_low_pct, sun_high_pct, precip_low_pct, precip_high_pct)

    if two_class.empty:
        raise SystemExit(
            "ERROR: no gloomy or clear days after labelling; check the percentile "
            "thresholds and that the input covers a range of weather.")

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    two_class.to_csv(output_csv, index=False)

    counts = two_class["day_type"].value_counts().to_dict()
    LOGGER.info("Dropped %d 'other' days; kept %d gloomy/clear days (%s) -> %s",
                n_dropped, len(two_class), counts, output_csv)
    print(f"Kept {len(two_class)} gloomy/clear days "
          f"(dropped {n_dropped} 'other'); class counts: {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ9 preprocessing: daily audio profile, labelled and filtered to gloomy/clear")
    parser.add_argument("input_csv", help="shared general-processed song-level CSV")
    parser.add_argument("-o", "--output", required=True, help="output two-class labelled CSV")
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
