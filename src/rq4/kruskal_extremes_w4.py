#!/usr/bin/env python3
"""
Kruskal-Wallis test comparing an audio feature across three temperature
extreme groups: Cold extreme, Normal, Hot extreme (based on
temperature_deviation_extreme from RQ4 preprocessing).

If significant, Dunn's post-hoc test with Bonferroni correction is applied.

Configurable: minimum group size, significance level, log file, log level.
"""

import os
import sys
import argparse
import logging
import json
from itertools import combinations

import pandas as pd
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn

_HERE = os.path.dirname(os.path.abspath(__file__))

# Fixed order of temperature extreme groups
GROUP_ORDER = ["Cold extreme", "Normal", "Hot extreme"]

# Default effect size bins (epsilon-squared thresholds)
EFFECT_SIZE_BINS = [
    (0.01, "negligible"),
    (0.04, "small"),
    (0.16, "moderate"),
    (0.36, "large"),
    (1.01, "very large"),
]

# Columns added by general preprocessing (to validate the input)
GENERAL_PREPROCESSING_COLUMNS = {'year', 'month', 'day', 'weekday',
                                 'is_weekend', 'season'}
AUDIO_FEATURE_COLS = ['acousticness', 'energy', 'tempo', 'valence', 'danceability']


def setup_logger(log_file, log_level=logging.DEBUG):
    """Configure a logger with file and console output."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('kruskal_extremes')
    logger.setLevel(log_level)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(ch)

    return logger


def effect_size_label(epsilon_sq, effect_bins=None):
    """Map epsilon-squared to a qualitative strength label."""
    if effect_bins is None:
        effect_bins = EFFECT_SIZE_BINS
    for threshold, label in effect_bins:
        if epsilon_sq < threshold:
            return label
    return "very large"


def epsilon_squared(h_stat, n):
    """Calculate Kruskal-Wallis effect size: eps^2 = H / (n - 1)."""
    if n <= 1:
        return 0.0
    return max(0.0, min(1.0, h_stat / (n - 1)))


def load_and_clean(file_path, cat_column, y_variable, logger,
                   aggregate_daily=True, date_column='date'):
    """
    Load the CSV and return a clean dataframe with only the needed columns.

    Args:
        file_path (str): Path to the RQ4-processed CSV.
        cat_column (str): The temperature extreme category column.
        y_variable (str): The audio feature column to test.
        logger (logging.Logger): Logger instance.
        aggregate_daily (bool): If True, aggregate to one row per day (default: True).
        date_column (str): Name of the date column for aggregation (default: 'date').

    Raises:
        ValueError: If file is empty, unreadable, or has insufficient data.
        KeyError: If required columns are missing.

    Returns:
        pd.DataFrame: Cleaned two-column dataframe.
        int: Number of rows dropped due to missing values.
    """
    # Check general preprocessing columns exist
    try:
        header_cols = set(pd.read_csv(file_path, nrows=0).columns)
    except Exception as e:
        logger.error(f"Could not read header from {file_path}: {e}")
        raise ValueError(f"Could not read file {file_path}: {e}")

    missing_general = GENERAL_PREPROCESSING_COLUMNS - header_cols
    if missing_general:
        logger.error(f"Missing general preprocessing columns: {sorted(missing_general)}")
        raise ValueError(
            f"Input CSV missing general preprocessing columns: {sorted(missing_general)}. "
            "Run dataprocessing.py on this file first."
        )

    # Read the data
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file is empty: {file_path}")
        raise ValueError(f"CSV file is empty: {file_path}")
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {file_path} ({e})")
        raise ValueError(f"Error parsing CSV file: {file_path}")

    # Check required columns
    missing = [c for c in (cat_column, y_variable) if c not in df.columns]
    if missing:
        logger.error(f"Missing columns in CSV file: {missing}")
        raise KeyError(f"Missing columns in CSV file: {missing}")

    # daily aggregation
    if aggregate_daily:
        # Ensure date column exists
        if date_column not in df.columns:
            logger.error(f"Date column '{date_column}' not found for aggregation.")
            raise KeyError(f"Date column '{date_column}' missing from CSV.")
        logger.info(f"Aggregating data by '{date_column}' (one row per day)")
        agg_dict = {cat_column: 'first', y_variable: 'mean'}
        df = df.groupby(date_column, as_index=False).agg(agg_dict)
        logger.debug(f"Aggregated to {len(df)} days")

    # Check for NaNs in audio feature
    if df[y_variable].isna().any():
        logger.warning(f"{y_variable} contains NaNs; they will be dropped.")

    # Subset and clean
    sub = df[[cat_column, y_variable]].copy()
    sub[y_variable] = pd.to_numeric(sub[y_variable], errors='coerce')

    before = len(sub)
    sub = sub.dropna()
    n_dropped = before - len(sub)
    if n_dropped > 0:
        logger.info(f"Dropped {n_dropped}/{before} rows with missing values")

    # Filter to known categories
    unknown = set(sub[cat_column].unique()) - set(GROUP_ORDER)
    if unknown:
        logger.warning(f"Unrecognized group labels found: {unknown} — dropping")
        sub = sub[sub[cat_column].isin(GROUP_ORDER)]

    if len(sub) == 0:
        logger.error("No valid rows after cleaning.")
        raise ValueError("No valid data for analysis.")

    # Check if y_variable is constant
    if sub[y_variable].nunique() == 1:
        logger.error(f"{y_variable} is constant; Kruskal-Wallis test is invalid.")
        raise ValueError(f"{y_variable} is constant in this dataset.")

    return sub, n_dropped


def group_summary(df, cat_column, y_variable):
    """Per-group n, median, and IQR in the defined group order."""
    summary = {}
    for label in GROUP_ORDER:
        vals = df.loc[df[cat_column] == label, y_variable]
        if len(vals) == 0:
            continue
        summary[label] = {
            "n": int(len(vals)),
            "median": round(float(vals.median()), 4),
            "q1": round(float(vals.quantile(0.25)), 4),
            "q3": round(float(vals.quantile(0.75)), 4),
        }
    return summary


def run_kruskal(df, cat_column, y_variable, min_group_size, logger):
    """
    Run Kruskal-Wallis across all sufficiently-sized groups.

    Args:
        df (pd.DataFrame): Cleaned dataframe.
        cat_column (str): The temperature extreme category column.
        y_variable (str): The audio feature column.
        min_group_size (int): Minimum observations per group to include.
        logger (logging.Logger): Logger instance.

    Raises:
        ValueError: If fewer than 2 usable groups remain.

    Returns:
        tuple: (H statistic, p-value, total n, list of used group labels)
    """
    groups, used_labels = [], []
    for label in GROUP_ORDER:
        vals = df.loc[df[cat_column] == label, y_variable]
        if len(vals) < min_group_size:
            logger.warning(
                f"Group '{label}' has only {len(vals)} rows "
                f"(< {min_group_size}) — excluded from test"
            )
            continue
        groups.append(vals)
        used_labels.append(label)

    if len(groups) < 2:
        logger.error("Fewer than 2 usable groups after size filtering.")
        raise ValueError(f"Need at least 2 groups with >= {min_group_size} observations each")

    h_stat, p_value = kruskal(*groups)
    total_n = sum(len(g) for g in groups)

    logger.debug(f"H={h_stat:.4f}, p={p_value:.4g}, n={total_n}, groups={used_labels}")
    return float(h_stat), float(p_value), total_n, used_labels


def run_dunn(df, cat_column, y_variable, used_labels, logger):
    """
    Run Dunn's post-hoc test with Bonferroni correction on the used groups.

    Args:
        df (pd.DataFrame): Cleaned dataframe.
        cat_column (str): The temperature extreme category column.
        y_variable (str): The audio feature column.
        used_labels (list): Groups included in the Kruskal-Wallis test.
        logger (logging.Logger): Logger instance.

    Returns:
        dict: "{group_a}_vs_{group_b}" -> adjusted p-value.
    """
    sub = df[df[cat_column].isin(used_labels)]
    matrix = posthoc_dunn(sub, val_col=y_variable, group_col=cat_column,
                          p_adjust='bonferroni')
    logger.debug(f"Post-hoc pairwise matrix:\n{matrix}")

    pairwise = {}
    for a, b in combinations(used_labels, 2):
        pairwise[f"{a}_vs_{b}"] = round(float(matrix.loc[a, b]), 4)
    return pairwise


def process(file_path, y_variable, cat_column, min_group_size, alpha,
            aggregate_daily, date_column, logger):
    """
    Full comparison pipeline: load -> summarize -> Kruskal-Wallis -> post-hoc.

    Args:
        file_path (str): Path to the RQ4-processed CSV.
        y_variable (str): The audio feature column to test.
        cat_column (str): The temperature extreme category column.
        min_group_size (int): Minimum observations per group to include.
        alpha (float): Significance level.
        aggregate_daily (bool): If True, aggregate to one row per day.
        date_column (str): Name of the date column for aggregation.
        logger (logging.Logger): Logger instance.

    Returns:
        dict: Full result payload.
    """
    df, n_dropped = load_and_clean(file_path, cat_column, y_variable, logger,
                                   aggregate_daily, date_column)
    summary = group_summary(df, cat_column, y_variable)

    h_stat, p_value, total_n, used_labels = run_kruskal(
        df, cat_column, y_variable, min_group_size, logger
    )

    eps_sq = epsilon_squared(h_stat, total_n)
    strength = effect_size_label(eps_sq)
    significant = p_value < alpha

    posthoc = {}
    if significant:
        logger.debug("Omnibus test significant — running Dunn's post-hoc")
        posthoc = run_dunn(df, cat_column, y_variable, used_labels, logger)
    else:
        logger.debug("Omnibus test not significant — skipping post-hoc")

    return {
        "y_variable": y_variable,
        "cat_column": cat_column,
        "n_dropped": n_dropped,
        "buckets_used": used_labels,
        "group_summary": summary,
        "h_statistic": round(h_stat, 4),
        "p_value": float(p_value),
        "significant_at_0.05": bool(significant),
        "effect_size": {"epsilon_squared": round(eps_sq, 4), "label": strength},
        "posthoc_pairwise_p": posthoc,
        "alpha_used": alpha,
        "aggregated_daily": aggregate_daily,
    }


def main(csvfile, y_col, cat_col, output, min_group_size, alpha,
         log_file, log_level, aggregate_daily, date_column):
    # Convert log level
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
    }
    log_level = level_map.get(log_level.upper(), logging.DEBUG)

    logger = setup_logger(log_file, log_level)
    logger.info(f"Running Kruskal-Wallis for {y_col} across {cat_col}")
    # Log aggregation status
    if aggregate_daily:
        logger.info(f"Aggregating data daily by '{date_column}'")

    # Create output directory if needed
    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        result = process(csvfile, y_col, cat_col, min_group_size, alpha,
                         aggregate_daily, date_column, logger)
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Result saved to {output}")

        # Print summary to console
        agg_msg = " (daily aggregated)" if aggregate_daily else ""
        logger.info(
            f"{y_col}{agg_msg}: H={result['h_statistic']:.3f}, "
            f"p={result['p_value']:.4g}, "
            f"effect={result['effect_size']['label']}, "
            f"{'significant' if result['significant_at_0.05'] else 'not significant'}"
        )
    except Exception as e:
        logger.error(f"Failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kruskal-Wallis test for RQ4 temperature extreme groups "
                    "(Cold extreme, Normal, Hot extreme), with Dunn's post-hoc."
    )
    parser.add_argument('csvfile', help='RQ4-processed CSV')
    parser.add_argument('-y', '--y_variable', required=True,
                        help='Audio feature column to test')
    parser.add_argument('-c', '--cat_column', default='temperature_deviation_extreme',
                        help='Temperature extreme category column (default: temperature_deviation_extreme)')
    parser.add_argument('-o', '--output', required=True,
                        help='Output JSON file path')
    parser.add_argument('--min-group-size', type=int, default=5,
                        help='Minimum observations per group to include (default: 5)')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level (default: 0.05)')
    parser.add_argument('--log', type=str, default=None,
                        help='Path to log file (default: logs/kruskal_extremes_<y_variable>.log)')
    parser.add_argument('--log-level', type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='DEBUG', help='Logging level (default: DEBUG)')
    parser.add_argument('--no-aggregate', action='store_true',
                        help='Disable daily aggregation (use raw song-level data, not recommended)')
    parser.add_argument('--date-column', type=str, default='date',
                        help='Name of the date column for aggregation (default: "date")')

    args = parser.parse_args()

    # Validate arguments
    if args.min_group_size < 2:
        parser.error("--min-group-size must be at least 2")
    if not (0 < args.alpha < 1):
        parser.error("--alpha must be between 0 and 1")

    # Set log default if not provided
    if args.log is None:
        args.log = f"{_HERE}/logs/kruskal_extremes_{args.y_variable}.log"

    aggregate_daily = not args.no_aggregate

    main(args.csvfile, args.y_variable, args.cat_column, args.output,
         args.min_group_size, args.alpha, args.log, args.log_level,
         aggregate_daily, args.date_column)
