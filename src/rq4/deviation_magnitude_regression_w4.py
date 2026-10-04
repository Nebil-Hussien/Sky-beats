#!/usr/bin/env python3
"""
RQ4 secondary analysis: does the magnitude of temperature deviation
(abs(deviation)) predict an audio feature?

Performs OLS regression and saves:
1. Full result as JSON (slope, intercept, p-value, R², effect size, etc.)
2. A TXT flag for significance (for Snakemake workflow integration)

Configurable: significance level, log file, log level.
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import statsmodels.api as sm

_HERE = os.path.dirname(os.path.abspath(__file__))

# Default effect size bins (R² thresholds) - can be extended
DEFAULT_EFFECT_BINS = [
    (0.01, "negligible"),
    (0.09, "small"),
    (0.25, "moderate"),
    (0.49, "large"),
    (1.01, "very large"),
]

# Columns added by general preprocessing (to validate the input)
GENERAL_PREPROCESSING_COLUMNS = {'year', 'month', 'day', 'weekday',
                                 'is_weekend', 'season'}
AUDIO_FEATURE_COLS = ['acousticness', 'energy', 'tempo', 'valence', 'danceability']


def setup_logger(log_file, log_level=logging.DEBUG):
    """Configure a logger with file and console output."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('devmag_regression')
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


def r2_label(r2, effect_bins=None):
    """Map R² to a qualitative label using provided effect size bins."""
    if effect_bins is None:
        effect_bins = DEFAULT_EFFECT_BINS
    for threshold, label in effect_bins:
        if r2 < threshold:
            return label
    return "very large"


def load_and_prepare(file_path, y_variable, logger):
    """
    Load the W4 processed CSV and prepare data for regression.

    Args:
        file_path (str): Path to the W4-processed CSV.
        y_variable (str): Audio feature column to regress.
        logger (logging.Logger): Logger instance.

    Raises:
        FileNotFoundError: If CSV is missing.
        ValueError: If CSV is empty, unreadable, or missing required columns.
        KeyError: If required columns are missing.

    Returns:
        pd.DataFrame: With 'abs_deviation' column and y_variable.
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

    # Check for required audio feature columns (should be median-filled)
    missing_audio = [c for c in AUDIO_FEATURE_COLS if c not in header_cols]
    if missing_audio:
        logger.warning(f"CSV missing some audio columns: {missing_audio}")
        # Not a hard error - the user might only have some features

    # Read the data
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file is empty: {file_path}")
        raise ValueError(f"CSV file is empty: {file_path}")
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {file_path} ({e})")
        raise ValueError(f"Error parsing CSV file: {file_path}")

    # Check required regression columns
    required = ['date', 'temperature_deviation', y_variable]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        raise KeyError(f"Missing required columns: {missing}")

    # Check for NaNs in audio feature (should have been median-filled)
    if df[y_variable].isna().any():
        logger.warning(f"{y_variable} contains NaNs; they will be dropped.")
        # They will be dropped in the next step

    # Subset and drop NaNs
    sub = df[required].copy()
    before = len(sub)
    sub = sub.dropna()
    dropped = before - len(sub)
    if dropped:
        logger.info(f"Dropped {dropped} rows with missing values")

    if len(sub) == 0:
        logger.error("No valid rows after dropping NaNs.")
        raise ValueError("No valid data for regression.")

    # Create absolute deviation
    sub['abs_deviation'] = sub['temperature_deviation'].abs()

    # Check if abs_deviation is constant
    if sub['abs_deviation'].nunique() == 1:
        logger.error("Absolute deviation is constant; regression is undefined.")
        raise ValueError("Absolute deviation is constant in this dataset.")

    # Check if y_variable is constant
    if sub[y_variable].nunique() == 1:
        logger.error(f"{y_variable} is constant; regression is undefined.")
        raise ValueError(f"{y_variable} is constant in this dataset.")

    return sub


def run_regression(df, y_variable, alpha, effect_bins, logger):
    """
    Regress y on abs(temperature_deviation).

    Args:
        df (pd.DataFrame): Prepared data with 'abs_deviation' and y_variable.
        y_variable (str): Audio feature column.
        alpha (float): Significance level.
        effect_bins (list): Effect size bins for R² labeling.
        logger (logging.Logger): Logger instance.

    Returns:
        dict: All regression results (native Python types).
    """
    X = sm.add_constant(df['abs_deviation'])
    y = df[y_variable]
    model = sm.OLS(y, X).fit()

    n = len(df)
    slope = float(model.params['abs_deviation'])
    p_value = float(model.pvalues['abs_deviation'])
    r_squared = float(model.rsquared)
    intercept = float(model.params['const'])
    significant = bool(p_value < alpha)
    effect_label = r2_label(r_squared, effect_bins)

    logger.debug(f"n={n}, slope={slope:.4f}, p={p_value:.4g}, R²={r_squared:.4f}, "
                 f"significant={significant}, effect={effect_label}")

    return {
        "n": n,
        "y_variable": y_variable,
        "intercept": round(intercept, 4),
        "slope": round(slope, 4),
        "slope_p_value": round(p_value, 6),
        "r_squared": round(r_squared, 4),
        "effect_label": effect_label,
        "significant_at_0.05": significant,
        "alpha_used": alpha,
    }


def main(csvfile, y_variable, output_json, output_txt,
         alpha, log_file, log_level, effect_bins):
    logger = setup_logger(log_file, log_level)
    logger.info(f"Starting absolute-deviation regression for {y_variable}")

    # Create output directories
    for out_path in [output_json, output_txt]:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    try:
        df = load_and_prepare(csvfile, y_variable, logger)
        result = run_regression(df, y_variable, alpha, effect_bins, logger)

        # Write JSON result
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        logger.info(f"JSON result saved: {output_json}")

        # Write significance flag
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write('True' if result['significant_at_0.05'] else 'False')
        logger.info(f"Significance flag saved: {output_txt}")

        # Print summary to console
        print(f"{y_variable}: slope={result['slope']:.4f}, "
              f"p={result['slope_p_value']:.4g}, "
              f"R²={result['r_squared']:.3f} ({result['effect_label']})")

    except Exception as e:
        logger.error(f"Regression failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regress audio feature on abs(temperature_deviation)."
    )
    parser.add_argument('csvfile', help='W4 processed CSV')
    parser.add_argument('-y', '--y_variable', required=True,
                        help='Audio feature column to regress')
    parser.add_argument('-o', '--output_json', required=True,
                        help='Output JSON file path')
    parser.add_argument('-s', '--significance', required=True,
                        help='Output TXT file path for significance flag')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level (default: 0.05)')
    parser.add_argument('--log', type=str, default=None,
                        help='Path to log file')
    parser.add_argument('--log-level', type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='DEBUG', help='Logging level (default: DEBUG)')

    args = parser.parse_args()

    # Validate alpha
    if not (0 < args.alpha < 1):
        parser.error("--alpha must be between 0 and 1")

    # Set log default if not provided
    if args.log is None:
        args.log = f"{_HERE}/logs/deviation_magnitude_regression_{args.y_variable}.log"

    # Use default effect size bins
    effect_bins = DEFAULT_EFFECT_BINS

    main(args.csvfile, args.y_variable, args.output_json, args.significance,
         args.alpha, args.log, args.log_level, effect_bins)
