#!/usr/bin/env python3
"""
Runs a Spearman rank correlation test between a continuous weather variable
(x_variable) and an audio feature (y_variable). Saves the result as JSON.

The data is aggregated daily (using the mean of the audio feature and the mean of the weather variable).

Configurable: sample size threshold, significance level, confidence level,
logging level, log file path, aggregation options.
"""

import pandas as pd
import numpy as np
import argparse
import logging
import json
import os
import sys
from scipy.stats import spearmanr, norm

_HERE = os.path.dirname(os.path.abspath(__file__))

# Defaults this values can be overridden via CLI
DEFAULT_MIN_SAMPLE_SIZE = 8
DEFAULT_ALPHA = 0.05
DEFAULT_CONFIDENCE = 0.95

# Cohen-style thresholds for effect size absolute rho
EFFECT_SIZE_BINS = [
    (0.10, "negligible"),
    (0.30, "weak"),
    (0.50, "moderate"),
    (0.70, "strong"),
    (1.01, "very strong"),
]


def setup_logger(y_variable, log_file, log_level=logging.DEBUG):
    """Configure a logger with file and console output."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(f'spearman_{y_variable}')
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


def effect_size_label(rho):
    """Map |rho| to a qualitative strength label."""
    abs_rho = abs(rho)
    for threshold, label in EFFECT_SIZE_BINS:
        if abs_rho < threshold:
            return label
    return "very strong"


def rho_confidence_interval(rho, n, confidence=0.95):
    """Calculate 95% CI for Spearman's rho using Fisher z-transformation."""
    if n < 4 or abs(rho) >= 1.0:
        return None, None
    z = np.arctanh(rho)
    se = 1 / np.sqrt(n - 3)
    z_crit = norm.ppf(1 - (1 - confidence) / 2)
    lo, hi = z - z_crit * se, z + z_crit * se
    return float(np.tanh(lo)), float(np.tanh(hi))


def load_and_aggregate(file_path, x_variable, y_variable,
                       aggregate_daily, date_column,
                       min_sample_size, logger):
    """Load the CSV, aggregate to daily level, and clean.

    Args:
        file_path (str): path to the CSV
        x_variable (str): continuous weather column
        y_variable (str): audio feature column
        aggregate_daily (bool): if True, aggregate to one row per date
        date_column (str): name of the date column
        min_sample_size (int): minimum valid rows required
        logger (logging.Logger): logger instance

    Raises:
        ValueError: if file is empty, unreadable, or insufficient data.
        KeyError: if columns are missing.

    Returns:
        tuple: (x_series, y_series, n_dropped)
    """
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file is empty: {file_path}")
        raise ValueError(f"CSV file is empty: {file_path}")
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {file_path} ({e})")
        raise ValueError(f"Error parsing CSV file: {file_path}")

    missing = [c for c in (x_variable, y_variable) if c not in df.columns]
    if aggregate_daily and date_column not in df.columns:
        missing.append(date_column)
    if missing:
        logger.error(f"Missing columns in CSV file: {missing}")
        raise KeyError(f"Missing columns in CSV file: {missing}")

    # Aggregate daily
    if aggregate_daily:
        logger.info(f"Aggregating data by '{date_column}' (one row per day)")
        agg_dict = {x_variable: 'mean', y_variable: 'mean'}
        df = df.groupby(date_column, as_index=False).agg(agg_dict)
        logger.debug(f"Aggregated to {len(df)} days")

    # Clean the aggregated data
    sub = df[[x_variable, y_variable]].apply(pd.to_numeric, errors='coerce')
    n_before = len(sub)
    sub = sub.dropna()
    n_after = len(sub)
    n_dropped = n_before - n_after
    if n_dropped > 0:
        logger.warning(f"Dropped {n_dropped}/{n_before} rows with "
                       f"missing or non-numeric {x_variable}/{y_variable}")

    if n_after < min_sample_size:
        logger.error(f"Only {n_after} valid rows remain "
                     f"(minimum {min_sample_size} required)")
        raise ValueError(f"Insufficient data: {n_after} valid rows "
                         f"(need at least {min_sample_size})")

    if sub[x_variable].nunique() == 1 or sub[y_variable].nunique() == 1:
        logger.error(
            "One or both columns are constant — correlation is undefined")
        raise ValueError(f"{x_variable} or {y_variable} has zero variance "
                         "in this data; correlation is undefined")

    return sub[x_variable], sub[y_variable], n_dropped


def compute_spearman(x, y, alpha, confidence,
                     logger=None, *, min_sample_size=8):
    """Compute Spearman's rho, p-value, CI, and effect size."""
    x = pd.Series(x).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    if len(x) != len(y):
        msg = f"Mismatched lengths: x has {len(x)} values and y has {len(y)}"
        if logger:
            logger.error(msg)
        raise ValueError(msg)
    n = len(x)

    if n < min_sample_size:
        msg = f"Insufficient data: {n} valid pairs (need at least {min_sample_size})"
        if logger:
            logger.error(msg)
        raise ValueError(msg)
    if x.nunique() == 1 or y.nunique() == 1:
        msg = "One or both variables are constant hence correlation is undefined"
        if logger:
            logger.error(msg)
        raise ValueError(msg)

    rho, p_value = spearmanr(x, y)
    ci_lo, ci_hi = rho_confidence_interval(rho, n, confidence)
    strength = effect_size_label(rho)
    significant = bool(p_value < alpha)

    if logger:
        logger.debug(f"rho={rho:.4f}, p={p_value:.4g}, n={n}, "
                     f"95% CI=({ci_lo}, {ci_hi}), strength={strength}")

    return {
        "n": n,
        "rho": round(float(rho), 4),
        "p_value": float(p_value),
        "significant_at_0.05": significant,
        "direction": "positive" if rho > 0 else "negative" if rho < 0 else "none",
        "effect_size": strength,
        "ci_95": [
            round(ci_lo, 4) if ci_lo is not None else None,
            round(ci_hi, 4) if ci_hi is not None else None
        ],
    }


def spearman_correlation_test(file_path, x_variable, y_variable,
                              min_sample_size, alpha, confidence,
                              aggregate_daily, date_column, logger):
    """Full pipeline: load (aggregate), clean, compute, and assemble result."""
    x, y, n_dropped = load_and_aggregate(
        file_path, x_variable, y_variable,
        aggregate_daily, date_column,
        min_sample_size, logger
    )
    logger.debug(f"Running Spearman correlation on {x_variable} vs "
                 f"{y_variable} (n={len(x)})")

    result = compute_spearman(
        x, y, alpha, confidence, logger, min_sample_size=min_sample_size
    )
    result["x_variable"] = x_variable
    result["y_variable"] = y_variable
    result["n_dropped"] = n_dropped
    result["aggregated_daily"] = aggregate_daily
    return result


def main(
        csvfile: str,
        x_col: str,
        y_col: str,
        output: str,
        min_sample_size: int,
        alpha: float,
        confidence: float,
        log_file: str,
        log_level: str,
        aggregate_daily: bool,
        date_column: str):

    # Convert log level
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
    }
    log_level = level_map.get(log_level.upper(), logging.DEBUG)

    logger = setup_logger(y_col, log_file, log_level)
    logger.info(f"Performing Spearman correlation test: {x_col} vs {y_col}")
    if aggregate_daily:
        logger.info(f"Aggregating data daily by '{date_column}'")

    # Create output directory if needed
    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        result = spearman_correlation_test(
            csvfile, x_col, y_col,
            min_sample_size, alpha, confidence,
            aggregate_daily, date_column, logger
        )
        logger.info(f"Result: {result}")
        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Wrote result to {output}")

        # Print summary to console
        agg_msg = " (daily aggregated)" if aggregate_daily else ""
        logger.info(f"{x_col} vs {y_col}{agg_msg}: rho={result['rho']} "
                    f"(n={result['n']}, {result['effect_size']}, "
                    f"p={result['p_value']:.4g}, "
                    f"{'significant' if result['significant_at_0.05'] else 'not significant'})")
    except Exception as e:
        logger.error(f"Failed to perform Spearman correlation test: {e}")
        raise
    logger.info("Finished Spearman correlation test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Spearman correlation between a weather variable and "
                    "an audio feature, saving result as JSON. "
                    "data is aggregated daily"
    )
    parser.add_argument('csvfile', type=str, help="Path to the dataset CSV")
    parser.add_argument(
        '-x', '--x_variable', type=str, required=True,
        help="The continuous weather column (e.g., precipitation_mm)")
    parser.add_argument('-y', '--y_variable', type=str, required=True,
                        help="The audio feature column to test")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Output JSON file path for the full result")
    parser.add_argument(
        '--min-sample-size', type=int, default=DEFAULT_MIN_SAMPLE_SIZE,
        help=f"Minimum number of valid rows (default: {DEFAULT_MIN_SAMPLE_SIZE})")
    parser.add_argument(
        '--alpha', type=float, default=DEFAULT_ALPHA,
        help=f"Significance level for determining significance (default: {DEFAULT_ALPHA})")
    parser.add_argument(
        '--confidence', type=float, default=DEFAULT_CONFIDENCE,
        help=f"Confidence level for the CI (default: {DEFAULT_CONFIDENCE})")
    parser.add_argument(
        '--log', type=str, default=None,
        help="Path to log file (default: logs/spearman_correlation_<y>.log)")
    parser.add_argument(
        '--log-level', type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='DEBUG',
        help="Logging level (default: DEBUG)")

    # New arguments for daily aggregation
    parser.add_argument(
        '--no-aggregate', action='store_true',
        help="Disable daily aggregation (use raw song-level data, not recommended)")
    parser.add_argument(
        '--date-column', type=str, default='date',
        help="Name of the date column for aggregation (default: 'date')")

    args = parser.parse_args()

    # Validate arguments
    if args.min_sample_size < 2:
        parser.error("--min-sample-size must be at least 2")
    if not (0 < args.alpha < 1):
        parser.error("--alpha must be between 0 and 1")
    if not (0 < args.confidence < 1):
        parser.error("--confidence must be between 0 and 1")

    # If log not set, use default with y_variable
    if args.log is None:
        args.log = f'{_HERE}/logs/spearman_correlation_{args.y_variable}.log'

    # Determine aggregation flag
    aggregate_daily = not args.no_aggregate

    main(args.csvfile, args.x_variable, args.y_variable, args.output,
         args.min_sample_size, args.alpha, args.confidence,
         args.log, args.log_level,
         aggregate_daily, args.date_column)
