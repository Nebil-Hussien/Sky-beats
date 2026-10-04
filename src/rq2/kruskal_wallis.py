#!/usr/bin/env python3
"""
Conducts a Kruskal-Wallis test comparing an audio feature across
precipitation categories (No rain, Low, Moderate, High). If significant,
Dunn's post-hoc test with Bonferroni correction is applied.

The script is configurable: minimum group size, significance level,
logging, and daily aggregation options can be set via command-line arguments.
"""

import pandas as pd
import argparse
import logging
import json
import os
import sys
from itertools import combinations
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn

_HERE = os.path.dirname(os.path.abspath(__file__))

# Fixed order of precipitation buckets
PRECIPITATION_CATEGORY_ORDER = ["No rain", "Low", "Moderate", "High"]

# Default minimum group size (can be overridden by CLI)
MINIMUM_GROUP_SIZE = 5

# Default significance level (can be overridden by CLI)
ALPHA = 0.05

# Effect size benchmarks (Cohen's guidelines adapted for epsilon-squared)
EFFECT_SIZE_BINS = [
    (0.01, "negligible"),
    (0.04, "small"),
    (0.16, "moderate"),
    (0.36, "large"),
    (1.01, "very large"),
]


def setup_logger(y_variable, log_file, log_level=logging.DEBUG):
    """Configure and return a logger with file output.

    Args:
        y_variable (str): name of the audio feature (used in log filename)
        log_file (str): full path to the log file
        log_level (int): logging level (e.g., logging.DEBUG)

    Returns:
        logging.Logger: configured logger
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(f'precip_bucket_{y_variable}')
    logger.setLevel(log_level)
    logger.handlers.clear()
    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(handler)
    # Also add a console handler for immediate feedback
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console)
    return logger


def effect_size_label(epsilon_sq):
    """Map epsilon-squared to a qualitative strength label.

    Args:
        epsilon_sq (float): epsilon-squared effect size

    Returns:
        str: qualitative effect size label
    """
    for threshold, label in EFFECT_SIZE_BINS:
        if epsilon_sq < threshold:
            return label
    return "very large"


def epsilon_squared(h_stat, n):
    """Calculate Kruskal-Wallis effect size: eps^2 = H / (n - 1).

    Args:
        h_stat (float): Kruskal-Wallis H statistic
        n (int): total sample size across all groups

    Returns:
        float: epsilon-squared, bounded [0, 1]
    """
    if n <= 1:
        return 0.0
    return max(0.0, min(1.0, h_stat / (n - 1)))


def load_and_aggregate(file_path, cat_column, y_variable,
                       aggregate_daily, date_column, logger):
    """Load the CSV, aggregate to daily level, and clean.

    Args:
        file_path (str): path to the w2-processed CSV
        cat_column (str): the precipitation bucket column
        y_variable (str): the audio feature column to test
        aggregate_daily (bool): if True, aggregate to one row per date
        date_column (str): name of the date column (used if aggregate_daily)
        logger (logging.Logger): logger instance

    Raises:
        ValueError: if file is empty or unreadable, or date column missing
        KeyError: if either cat_column or y_variable is missing

    Returns:
        pd.DataFrame: cleaned two-column dataframe (one row per day if aggregated)
        int: number of rows dropped due to missing values
    """
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file is empty: {file_path}")
        raise ValueError(f"CSV file is empty: {file_path}")
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {file_path} ({e})")
        raise ValueError(f"Error parsing CSV file: {file_path}")

    # Ensure required columns exist
    required = [cat_column, y_variable]
    if aggregate_daily:
        required.append(date_column)
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"Missing columns in CSV file: {missing}")
        raise KeyError(f"Missing columns in CSV file: {missing}")

    # Aggregate daily
    if aggregate_daily:
        logger.info(f"Aggregating data by '{date_column}' (one row per day)")
        agg_dict = {
            cat_column: 'first',
            y_variable: 'mean'
        }
        df = df.groupby(date_column, as_index=False).agg(agg_dict)
        logger.debug(f"Aggregated from {len(df)} days")

    # Keep only the two relevant columns and drop missing
    sub = df[[cat_column, y_variable]].copy()
    sub[y_variable] = pd.to_numeric(sub[y_variable], errors='coerce')
    n_before = len(sub)
    sub = sub.dropna()
    n_dropped = n_before - len(sub)
    if n_dropped > 0:
        logger.warning(f"Dropped {n_dropped}/{n_before} rows with "
                       f"missing {cat_column}/{y_variable}")

    unknown = set(sub[cat_column].unique()) - set(PRECIPITATION_CATEGORY_ORDER)
    if unknown:
        logger.warning(f"Unrecognized bucket labels found, dropping: {unknown}")
        sub = sub[sub[cat_column].isin(PRECIPITATION_CATEGORY_ORDER)]

    return sub, n_dropped


def group_summary(df, cat_column, y_variable):
    """Compute per-bucket n, median, and IQR in ascending order.

    Args:
        df (pd.DataFrame): cleaned dataframe
        cat_column (str): the precipitation bucket column
        y_variable (str): the audio feature column

    Returns:
        dict: bucket -> {n, median, q1, q3}
    """
    summary = {}
    for label in PRECIPITATION_CATEGORY_ORDER:
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


def run_kruskal_wallis(df, cat_column, y_variable, min_group_size, logger):
    """Run Kruskal-Wallis across all non-empty, sufficiently-sized buckets.

    Args:
        df (pd.DataFrame): cleaned dataframe
        cat_column (str): the precipitation bucket column
        y_variable (str): the audio feature column
        min_group_size (int): minimum observations per bucket to include
        logger (logging.Logger): logger instance

    Raises:
        ValueError: if fewer than 2 usable groups remain

    Returns:
        tuple: (H statistic, p-value, total n, list of used bucket labels)
    """
    groups, used_labels, total_n = [], [], 0
    for label in PRECIPITATION_CATEGORY_ORDER:
        vals = df.loc[df[cat_column] == label, y_variable]
        if len(vals) < min_group_size:
            logger.warning(f"Bucket '{label}' has only {len(vals)} rows "
                           f"(< {min_group_size}) — excluded from test")
            continue
        groups.append(vals)
        used_labels.append(label)
        total_n += len(vals)

    if len(groups) < 2:
        logger.error("Fewer than 2 usable buckets; cannot run Kruskal-Wallis")
        raise ValueError(f"Need at least 2 buckets with >= {min_group_size} observations each")

    h_stat, p_value = kruskal(*groups)
    logger.debug(f"H={h_stat:.4f}, p={p_value:.4g}, n={total_n}, "
                 f"buckets used={used_labels}")
    return float(h_stat), float(p_value), total_n, used_labels


def run_posthoc(df, cat_column, y_variable, used_labels, logger):
    """Run Dunn's post-hoc test with Bonferroni correction on the used buckets.

    Args:
        df (pd.DataFrame): clean dataframe
        cat_column (str): the precipitation bucket column
        y_variable (str): the audio feature column
        used_labels (list): buckets included in the Kruskal-Wallis test
        logger (logging.Logger): logger instance

    Returns:
        dict: "{bucket_a}_vs_{bucket_b}" -> adjusted p-value
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
    """Full comparison pipeline: load -> aggregate -> summarize -> Kruskal-Wallis -> post-hoc.

    Args:
        file_path (str): path to the w2-processed CSV
        y_variable (str): the audio feature column to test
        cat_column (str): the precipitation bucket column
        min_group_size (int): minimum observations per bucket
        alpha (float): significance level for omnibus test
        aggregate_daily (bool): if True, aggregate to daily level first
        date_column (str): name of the date column for aggregation
        logger (logging.Logger): logger instance

    Returns:
        dict: full result payload with all statistics and effect sizes
    """
    df, n_dropped = load_and_aggregate(
        file_path, cat_column, y_variable,
        aggregate_daily, date_column, logger
    )
    summary = group_summary(df, cat_column, y_variable)

    h_stat, p_value, total_n, used_labels = run_kruskal_wallis(
        df, cat_column, y_variable, min_group_size, logger)

    eps_sq = epsilon_squared(h_stat, total_n)
    strength = effect_size_label(eps_sq)
    significant = p_value < alpha

    posthoc = {}
    if significant:
        logger.debug("Omnibus test significant — running Dunn's post-hoc")
        posthoc = run_posthoc(df, cat_column, y_variable, used_labels, logger)
    else:
        logger.debug("Omnibus test not significant — skipping post-hoc")

    return {
        "y_variable": y_variable,
        "cat_column": cat_column,
        "n_dropped": n_dropped,
        "buckets_used": used_labels,
        "group_summary": summary,
        "h_statistic": round(h_stat, 4),
        "p_value": p_value,
        "significant_at_0.05": bool(significant),
        "effect_size": {"epsilon_squared": round(eps_sq, 4), "label": strength},
        "posthoc_pairwise_p": posthoc,
        "aggregated_daily": aggregate_daily,
    }


def main(csvfile: str, y_col: str, cat_col: str, output: str,
         min_group_size: int, alpha: float, log_file: str, log_level: str,
         aggregate_daily: bool, date_column: str):
    # Convert log level string to logging constant
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
    }
    log_level = level_map.get(log_level.upper(), logging.DEBUG)

    logger = setup_logger(y_col, log_file, log_level)
    logger.info(f"Performing Kruskal-Wallis bucket comparison for {y_col}")
    if aggregate_daily:
        logger.info(f"Aggregating data daily by '{date_column}'")

    # Create output directory if needed
    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        result = process(csvfile, y_col, cat_col, min_group_size, alpha,
                         aggregate_daily, date_column, logger)
        logger.info(f"Result: {result}")
        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Wrote result to {output}")

        # Print summary to console
        agg_msg = " (daily aggregated)" if aggregate_daily else ""
        logger.info(f"{y_col} across {cat_col}{agg_msg}: H={result['h_statistic']} "
                    f"(buckets={result['buckets_used']}, "
                    f"{result['effect_size']['label']}, "
                    f"p={result['p_value']:.4g}, "
                    f"{'significant' if result['significant_at_0.05'] else 'not significant'})")
    except Exception as e:
        logger.error(f"Failed to perform bucket comparison: {e}")
        raise
    logger.info("Finished bucket comparison")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare an audio feature across 4 precipitation buckets "
                    "using Kruskal-Wallis + Dunn's post-hoc, saving result as JSON. "
                    "data is aggregated daily"
    )
    parser.add_argument('csvfile', type=str,
                        help="Path to the w2-processed dataset (output of datapreprocessing_w2.py)")
    parser.add_argument('-y', '--y_variable', type=str, required=True,
                        help="The audio feature column to test")
    parser.add_argument('-c', '--cat_column', type=str, default='precipitation_mm_cat',
                        help="The precipitation bucket column (default: precipitation_mm_cat)")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Output JSON file path for the full result")
    parser.add_argument('--min-group-size', type=int, default=MINIMUM_GROUP_SIZE,
                        help=f"Minimum number of observations per bucket to include (default: {MINIMUM_GROUP_SIZE})")
    parser.add_argument('--alpha', type=float, default=ALPHA,
                        help=f"Significance level for omnibus test (default: {ALPHA})")
    parser.add_argument('--log', type=str,
                        default=None,
                        help="Path to log file (default: logs/precip_bucket_comparison_<y>.log)")
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='DEBUG', help="Logging level (default: DEBUG)")
    parser.add_argument('--no-aggregate', action='store_true',
                        help="Disable daily aggregation")
    parser.add_argument('--date-column', type=str, default='date',
                        help="Name of the date column for aggregation")

    args = parser.parse_args()

    # Validate arguments
    if args.min_group_size < 2:
        parser.error("--min-group-size must be at least 2")
    if not (0 < args.alpha < 1):
        parser.error("--alpha must be between 0 and 1")

    # Default log file if not provided
    if args.log is None:
        args.log = f'{_HERE}/logs/precip_bucket_comparison_{args.y_variable}.log'

    # Determine aggregation flag
    aggregate_daily = not args.no_aggregate

    main(args.csvfile, args.y_variable, args.cat_column, args.output,
         args.min_group_size, args.alpha, args.log, args.log_level,
         aggregate_daily, args.date_column)
