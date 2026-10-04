#!/usr/bin/env python3
"""
audio_profile.py
----------------
RQ6 analysis: compare the audio profile of gloomy vs clear days.

For each audio feature, compute the mean on gloomy days and on clear days, their
difference (gloomy − clear), and a Mann-Whitney U test (non-parametric — the
per-day feature distributions are not necessarily normal) of whether the two
groups differ. Writes a comparison table to CSV.

Usage
-----
  python audio_profile.py <daily_labeled_csv> -o <output_csv>
      [--features valence,energy,danceability,acousticness,tempo] [--alpha 0.05]
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu

DEFAULT_FEATURES = ["valence", "energy", "danceability", "acousticness", "tempo"]

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


def setup_logger():
    """Configure logging to ./logs relative to this script (CWD-independent)."""
    log_path = Path(__file__).resolve().parent / "logs" / "audio_profile.log"
    logging.basicConfig(filename=log_path, filemode="w", level=logging.DEBUG)
    return logging.getLogger(__name__)


def compare_profiles(df, features, alpha=0.05):
    """Per feature: gloomy/clear means, difference, and Mann-Whitney U test.

    Returns a DataFrame with one row per feature.
    """
    gloomy = df[df["day_type"] == "gloomy"]
    clear = df[df["day_type"] == "clear"]
    if gloomy.empty or clear.empty:
        raise SystemExit(
            f"ERROR: need both groups; got {len(gloomy)} gloomy and {len(clear)} clear rows.")

    rows = []
    for feat in features:
        if feat not in df.columns:
            raise SystemExit(f"ERROR: feature column '{feat}' not in input.")
        g = gloomy[feat].dropna()
        c = clear[feat].dropna()
        u_stat, p_value = mannwhitneyu(g, c, alternative="two-sided")
        rows.append({
            "feature": feat,
            "gloomy_mean": round(g.mean(), 4),
            "clear_mean": round(c.mean(), 4),
            "diff_gloomy_minus_clear": round(g.mean() - c.mean(), 4),
            "u_statistic": round(float(u_stat), 1),
            "p_value": round(float(p_value), 4),
            "significant": bool(p_value < alpha),
        })
    return pd.DataFrame(rows)


def main(input_csv, output_csv, features, alpha):
    setup_logger()
    LOGGER.info("Loading %s", input_csv)
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        raise SystemExit(f"ERROR: input file not found: {input_csv}")
    except pd.errors.EmptyDataError:
        raise SystemExit(f"ERROR: input file is empty: {input_csv}")

    if "day_type" not in df.columns:
        raise SystemExit("ERROR: input missing 'day_type' column (run datapreprocessing_w6 first).")

    result = compare_profiles(df, features, alpha)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    LOGGER.info("Saved comparison -> %s", output_csv)
    print(result.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ6: compare audio profile between gloomy and clear days")
    parser.add_argument("input_csv", help="daily labelled CSV from datapreprocessing_w6.py")
    parser.add_argument("-o", "--output", required=True, help="output comparison CSV")
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES),
                        help="comma-separated audio feature columns to compare")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="significance level for the Mann-Whitney test (default 0.05)")
    args = parser.parse_args()
    feats = [f.strip() for f in args.features.split(",") if f.strip()]
    main(args.input_csv, args.output, feats, args.alpha)
