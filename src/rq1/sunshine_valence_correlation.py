#!/usr/bin/env python3
"""
sunshine_valence_correlation.py
-------------------------------
RQ1 analysis: Spearman correlation between daily sunshine duration and the mean
valence of the German Spotify Top 200 — computed for the whole date range
("all") and separately per meteorological season, so the seasonal variation of
the relationship can be assessed.

Spearman is used (consistent with Workflow 2) because it is robust to the
skewed, bounded distribution of daily sunshine hours and does not assume a
linear relationship.

Output: a CSV with one row per group:
    group, n_days, rho, p_value, significant

Usage
-----
  python sunshine_valence_correlation.py <daily_csv> -o <output_csv>
      [-x sunshine_hours] [-y valence] [--alpha 0.05]
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

SEASON_ORDER = ["winter", "spring", "summer", "autumn"]


def setup_logger():
    """Configure logging to ./logs relative to this script (CWD-independent)."""
    log_path = Path(__file__).resolve().parent / "logs" / "sunshine_valence_correlation.log"
    logging.basicConfig(filename=log_path, filemode="w", level=logging.DEBUG)
    return logging.getLogger(__name__)


def correlate_by_season(df, x_variable, y_variable, alpha=0.05):
    """Spearman correlation of x vs y, overall and per season.

    Args:
        df (pd.DataFrame): daily rows with x_variable, y_variable, 'season'
        x_variable (str): weather column (e.g. sunshine_hours)
        y_variable (str): audio feature column (e.g. valence)
        alpha (float): significance level for the 'significant' flag

    Returns:
        pd.DataFrame: one row per group (all + each season) with
            n_days, rho, p_value, significant
    """
    rows = []
    groups = [("all", df)] + [
        (season, df[df["season"] == season]) for season in SEASON_ORDER
    ]
    for name, group in groups:
        group = group[[x_variable, y_variable]].dropna()
        if len(group) < 3:
            LOGGER.warning("Group '%s' has only %d rows; skipping test", name, len(group))
            rows.append({"group": name, "n_days": len(group),
                         "rho": float("nan"), "p_value": float("nan"),
                         "significant": False})
            continue
        rho, p_value = spearmanr(group[x_variable], group[y_variable])
        LOGGER.debug("%s: n=%d rho=%.4f p=%.4g", name, len(group), rho, p_value)
        rows.append({"group": name, "n_days": len(group),
                     "rho": rho, "p_value": p_value,
                     "significant": bool(p_value < alpha)})
    return pd.DataFrame(rows)


def main(input_csv, output_csv, x_variable, y_variable, alpha):
    setup_logger()
    LOGGER.info("Loading %s", input_csv)
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        raise SystemExit(f"ERROR: input file not found: {input_csv}")
    except pd.errors.EmptyDataError:
        raise SystemExit(f"ERROR: input file is empty: {input_csv}")

    required = {x_variable, y_variable, "season"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: input missing required columns: {sorted(missing)}")

    result = correlate_by_season(df, x_variable, y_variable, alpha)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    LOGGER.info("Saved correlation table -> %s", output_csv)
    print(result.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ1: Spearman correlation of sunshine vs valence, overall and per season")
    parser.add_argument("input_csv", help="daily CSV from datapreprocessing_w1.py")
    parser.add_argument("-o", "--output", required=True, help="output correlation CSV")
    parser.add_argument("-x", "--x_variable", default="sunshine_hours",
                        help="weather column (default: sunshine_hours)")
    parser.add_argument("-y", "--y_variable", default="valence",
                        help="audio feature column (default: valence)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="significance level (default: 0.05)")
    args = parser.parse_args()
    main(args.input_csv, args.output, args.x_variable, args.y_variable, args.alpha)
