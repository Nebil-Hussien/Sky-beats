#!/usr/bin/env python3
"""
genre_shift.py
--------------
RQ3: Which music genres become more or less popular in the rainiest months
compared to the driest months?

Reads the shared pre-processed Sky Beats dataset (song-level rows with weather
columns), ranks the N rainiest and N driest calendar months by mean daily
precipitation, and computes each genre's popularity shift (rainy - dry) by two
metrics:
  - stream share : % of the group's total streams going to each genre
  - count  share : % of the group's charting song-rows in each genre
A positive shift means the genre is relatively more popular in rainy months.
The per-genre shift table is written to CSV; plotting is a separate component
(plot_genre_shift.py).

Note on genre data
------------------
Genre labels cover only the ~22%-matched subset of charting songs and are skewed
toward electronic genres; results should be read as *relative* shifts between
rainy and dry months within this sample, not absolute genre popularity.

Usage
-----
  python genre_shift.py <input_csv> -o <output_csv> [--n-months 12]
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


def setup_logger():
    """Configure logging to ./logs relative to this script (CWD-independent)."""
    log_path = Path(__file__).resolve().parent / "logs" / "genre_shift.log"
    logging.basicConfig(filename=log_path, filemode="w", level=logging.DEBUG)
    return logging.getLogger(__name__)


def genre_shares(df):
    """Return per-genre stream share and song-count share (each summing to 1)."""
    by_streams = df.groupby("track_genre")["streams"].sum()
    by_counts = df.groupby("track_genre").size()
    return pd.DataFrame({
        "stream_share": by_streams / by_streams.sum(),
        "count_share": by_counts / by_counts.sum(),
    })


def compute_genre_shift(df, n_months):
    """
    Split the data into the N rainiest and N driest calendar months and return a
    table of per-genre popularity shifts (rainy - dry) for both metrics.
    """
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M")

    # Average precipitation per month (one weather value per day, so de-dup by day first).
    monthly_precip = (
        df.drop_duplicates(subset="date")
          .groupby("month")["precipitation_mm"].mean()
          .sort_values()
    )

    n_available = len(monthly_precip)
    if n_months * 2 > n_available:
        raise SystemExit(
            f"ERROR: n_months={n_months} needs {n_months * 2} distinct months, "
            f"but only {n_available} are available. Lower n_months."
        )

    driest_months = monthly_precip.head(n_months).index
    rainiest_months = monthly_precip.tail(n_months).index
    LOGGER.info("Driest months:   %s", ", ".join(str(m) for m in driest_months))
    LOGGER.info("Rainiest months: %s", ", ".join(str(m) for m in rainiest_months))

    dry = genre_shares(df[df["month"].isin(driest_months)])
    rainy = genre_shares(df[df["month"].isin(rainiest_months)])

    # Align on the full genre set (a genre missing in one group counts as 0 share).
    shift = pd.DataFrame(index=sorted(set(dry.index) | set(rainy.index)))
    shift["rainy_stream_share"] = rainy["stream_share"].reindex(shift.index).fillna(0)
    shift["dry_stream_share"] = dry["stream_share"].reindex(shift.index).fillna(0)
    shift["stream_shift"] = shift["rainy_stream_share"] - shift["dry_stream_share"]
    shift["rainy_count_share"] = rainy["count_share"].reindex(shift.index).fillna(0)
    shift["dry_count_share"] = dry["count_share"].reindex(shift.index).fillna(0)
    shift["count_shift"] = shift["rainy_count_share"] - shift["dry_count_share"]

    return shift.sort_values("stream_shift", ascending=False).round(4)


def main(input_csv, output_csv, n_months):
    setup_logger()
    LOGGER.info("Loading %s", input_csv)
    try:
        df = pd.read_csv(input_csv, parse_dates=["date"])
    except FileNotFoundError:
        raise SystemExit(f"ERROR: input file not found: {input_csv}")
    except pd.errors.EmptyDataError:
        raise SystemExit(f"ERROR: input file is empty: {input_csv}")

    for col in ("track_genre", "streams", "precipitation_mm", "date"):
        if col not in df.columns:
            raise SystemExit(f"ERROR: expected column '{col}' missing from input.")

    shift = compute_genre_shift(df, n_months)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    shift.to_csv(output_csv)
    LOGGER.info("Saved shift table -> %s", output_csv)

    print("\nTop 5 genres gaining popularity in rainy months (by stream share):")
    print(shift.head(5)[["rainy_stream_share", "dry_stream_share", "stream_shift"]].to_string())
    print("\nTop 5 genres gaining popularity in dry months (by stream share):")
    print(shift.tail(5)[["rainy_stream_share", "dry_stream_share", "stream_shift"]].to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ3: per-genre popularity shift, rainiest vs driest months")
    parser.add_argument("input_csv", help="pre-processed song-level merged CSV")
    parser.add_argument("-o", "--output", required=True,
                        help="output CSV for the per-genre shift table")
    parser.add_argument("--n-months", type=int, default=12,
                        help="number of rainiest/driest months to compare (default 12)")
    args = parser.parse_args()
    main(args.input_csv, args.output, args.n_months)
