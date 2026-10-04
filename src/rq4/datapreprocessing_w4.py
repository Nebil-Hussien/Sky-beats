#!/usr/bin/env python3
"""
RQ4 preprocessing:
- Compute temperature deviation from monthly climatology.
- Flag extreme hot/cold days as 5th/95th percentiles (configurable).
- Add week/month period labels.

The script expects a CSV that has already been through the general
preprocessing step (dataprocessing.py), which adds columns like year,
month, season, etc. It validates the presence of those columns.
"""

import pandas as pd
import argparse
import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

# Columns required from the general-processed CSV
COLUMNS = ['date', 'temperature_mean_c', 'acousticness',
           'energy', 'tempo', 'valence', 'danceability']

AUDIO_FEATURE_COLS = [
    'acousticness',
    'energy',
    'tempo',
    'valence',
    'danceability']

# These columns are added by dataprocessing.py's general preprocessing
GENERAL_PREPROCESSING_COLUMNS = {'year', 'month', 'day', 'weekday',
                                 'is_weekend', 'season'}

# Default percentile cutoffs
DEFAULT_LOW_PCT = 5
DEFAULT_HIGH_PCT = 95
DEVIATION_LABELS = ["Cold extreme", "Normal", "Hot extreme"]
# Backward-compatible public name used by earlier workflow versions.
DEV_LABELS = DEVIATION_LABELS


def setup_logger(log_file, log_level=logging.DEBUG):
    """Configure a logger with file and console output."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('datapreprocessing_w4')
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


def add_temperature_deviation(df, temp_col='temperature_mean_c'):
    """
    Compute monthly climatological mean and the deviation from it.

    Args:
        df (pd.DataFrame): Must have a 'date' column and temp_col.
        temp_col (str): Name of the temperature column.

    Returns:
        pd.DataFrame: With a new column 'temperature_deviation'.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    monthly_mean = df.groupby(df['date'].dt.month)[temp_col].transform('mean')
    df['temperature_deviation'] = df[temp_col] - monthly_mean
    return df


def flag_extreme_deviation(df, dev_col='temperature_deviation',
                           low_pct=DEFAULT_LOW_PCT, high_pct=DEFAULT_HIGH_PCT,
                           labels=DEVIATION_LABELS):
    """
    Create a three‑level categorical column based on percentiles of the deviation.

    Args:
        df (pd.DataFrame): Must have dev_col.
        dev_col (str): Deviation column name.
        low_pct (float): Lower percentile cutoff (e.g., 5).
        high_pct (float): Upper percentile cutoff (e.g., 95).
        labels (list): Three labels for categories.

    Returns:
        pd.DataFrame: With new column 'temperature_deviation_extreme'.
    """
    cold_cut = df[dev_col].quantile(low_pct / 100)
    hot_cut = df[dev_col].quantile(high_pct / 100)
    # If the cold_cut is greater than or equal to hot_cut, it means all values
    # are identical or very close, so we classify all as "Normal".
    if cold_cut >= hot_cut:
        df['temperature_deviation_extreme'] = labels[1]  # "Normal"
        return df
    bins = [-float('inf'), cold_cut, hot_cut, float('inf')]
    df['temperature_deviation_extreme'] = pd.cut(
        df[dev_col], bins=bins, labels=labels, duplicates='drop')
    return df


def add_hotcold_binary(df, extreme_col='temperature_deviation_extreme'):
    """
    Add a binary column: 1 for Hot extreme, 0 for Cold extreme, NaN for Normal.

    Args:
        df (pd.DataFrame): Must have extreme_col.
        extreme_col (str): Name of the extreme category column.

    Returns:
        pd.DataFrame: With new column '<extreme_col>_hotcold'.
    """
    mapping = {"Hot extreme": 1, "Cold extreme": 0, "Normal": None}
    df[f'{extreme_col}_hotcold'] = df[extreme_col].map(mapping)
    return df


def add_period_columns(df):
    """
    Add week_label (ISO week) and month_label (YYYY-MM) columns from the date.

    Args:
        df (pd.DataFrame): Must have a 'date' column.

    Returns:
        pd.DataFrame: With 'week_label' and 'month_label' columns.
    """
    df['date'] = pd.to_datetime(df['date'])
    iso = df['date'].dt.isocalendar()
    df['week_label'] = iso['year'].astype(
        str) + '-W' + iso['week'].astype(str).str.zfill(2)
    df['month_label'] = df['date'].dt.to_period('M').astype(str)
    return df


def process_csvfile_w4(filename, low_pct, high_pct, logger):
    """
    Full RQ4 preprocessing pipeline.

    Args:
        filename (str): Path to the general‑processed CSV.
        low_pct (float): Lower percentile cutoff.
        high_pct (float): Upper percentile cutoff.
        logger (logging.Logger): Logger instance.

    Raises:
        ValueError: If the CSV is missing required columns or has NaNs in audio features.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    # Check that the CSV has the general preprocessing columns
    try:
        header_cols = set(pd.read_csv(filename, nrows=0).columns)
    except Exception as e:
        logger.error(f"Could not read header from {filename}: {e}")
        raise ValueError(f"Could not read file {filename}: {e}")

    missing_general = GENERAL_PREPROCESSING_COLUMNS - header_cols
    if missing_general:
        logger.error(
            f"Missing general preprocessing columns: {
                sorted(missing_general)}")
        raise ValueError(
            f"Input CSV missing general preprocessing columns: {
                sorted(missing_general)}. "
            "Run dataprocessing.py on this file first."
        )

    # Read the data
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        logger.error(f"Could not read columns {COLUMNS} from {filename}: {e}")
        raise ValueError(f"Could not read data from {filename}: {e}")

    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])

    # Verify audio feature columns have no NaNs (should have been
    # median‑filled)
    cols_with_nans = [c for c in AUDIO_FEATURE_COLS if df[c].isna().any()]
    if cols_with_nans:
        logger.error(f"Audio columns with NaNs: {cols_with_nans}")
        raise ValueError(
            f"Audio feature columns still have NaNs: {cols_with_nans}. "
            "The general preprocessing step should have median‑filled them."
        )

    # Drop rows with missing essential values
    essential = ['date', 'temperature_mean_c'] + AUDIO_FEATURE_COLS
    before = len(df)
    df = df.dropna(subset=essential)
    dropped = before - len(df)
    if dropped:
        logger.info(f"Dropped {dropped} rows with missing essential values")

    # Add deviation and extreme categories
    df = add_temperature_deviation(df)
    df = flag_extreme_deviation(df, low_pct=low_pct, high_pct=high_pct)
    df = add_hotcold_binary(df)

    # Add period labels
    df = add_period_columns(df)

    logger.debug(f"Processed df head:\n{df.head()}")
    return df


def main(csvfile, outfile, low_pct, high_pct, log_file, log_level):
    logger = setup_logger(log_file, log_level)
    logger.info(f"Starting RQ4 preprocessing on {csvfile}")

    if not csvfile.endswith('.csv'):
        logger.error(f"Input file {csvfile} is not a CSV")
        raise OSError("Input file must have .csv extension")

    # Create output directory if needed
    out_dir = os.path.dirname(outfile)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        df = process_csvfile_w4(csvfile, low_pct, high_pct, logger)
        df.to_csv(outfile, index=False)
        logger.info(f"Saved processed CSV to {outfile}")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

    logger.info("Finished RQ4 preprocessing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ4 preprocessing: temperature deviation and extreme flagging.")
    parser.add_argument('-i', '--processedcsvfile', required=True,
                        help="Input CSV (must have general preprocessing)")
    parser.add_argument('-o', '--outfile', required=True,
                        help="Output CSV file name")
    parser.add_argument(
        '--low-pct',
        type=float,
        default=DEFAULT_LOW_PCT,
        help=f"Lower percentile cutoff (default: {DEFAULT_LOW_PCT})")
    parser.add_argument(
        '--high-pct',
        type=float,
        default=DEFAULT_HIGH_PCT,
        help=f"Upper percentile cutoff (default: {DEFAULT_HIGH_PCT})")
    parser.add_argument('--log', type=str,
                        default=f"{_HERE}/logs/dataprocessing_w4.log",
                        help="Path to log file")
    parser.add_argument('--log-level', type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='DEBUG', help="Logging level (default: DEBUG)")

    args = parser.parse_args()

    # Validate percentiles
    if not (0 < args.low_pct < args.high_pct < 100):
        parser.error("Percentiles must satisfy 0 < low_pct < high_pct < 100")

    main(args.processedcsvfile, args.outfile,
         args.low_pct, args.high_pct,
         args.log, args.log_level)
