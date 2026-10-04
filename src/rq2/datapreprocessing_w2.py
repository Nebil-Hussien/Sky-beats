#!/usr/bin/env python3
"""
Processes the general-processed Sky Beats dataset for Workflow 2 (RQ2) analysis.
It adds RQ2-specific features:
- Bucketing daily precipitation into 4 categories using dataset percentiles
- Adding week/month period labels for downstream averaging

The percentile cut points are user-configurable (default: 25th, 50th, 75th).
"""

import pandas as pd
import argparse
import logging
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# Columns expected from the general preprocessing step
COLUMNS = ['date', 'precipitation_mm', 'acousticness',
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

PRECIPITATION_AMOUNT_LABELS = ["No rain", "Low", "Moderate", "High"]
# Backward-compatible public name used by earlier workflow versions.
PRECIP_LABELS = PRECIPITATION_AMOUNT_LABELS


def setup_logger(log_file):
    """Configure and return a logger with file output."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('datapreprocessing_w2')
    logger.setLevel(logging.DEBUG)
    # Remove any existing handlers to avoid duplication
    if logger.handlers:
        logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    return logger


def create_precipitation_bins_percentile(
        df, column, p1, p2, p3, labels=PRECIPITATION_AMOUNT_LABELS):
    """Bucket a precipitation column into 4 categories using custom percentiles.

    Args:
        df (pd.DataFrame): input dataframe
        column (str): name of the precipitation column
        p1, p2, p3 (float): percentile cut points (0-100)
        labels (list): 4 category labels

    Returns:
        pd.DataFrame: df with an added f'{column}_cat' column
    """
    # Validate percentiles are increasing and in range
    if not (0 < p1 < p2 < p3 < 100):
        raise ValueError(
            f"Percentiles must be strictly increasing. Got {p1}, {p2}, {p3}")

    # Compute cuts
    cuts = [df[column].quantile(p / 100) for p in (p1, p2, p3)]

    # Build bin edges
    min_val = df[column].min()
    first_edge = min_val - 1e-6 if min_val > 0 else -0.001
    bins = [first_edge] + cuts + [float("inf")]
    unique_bins = sorted(set(bins))  # Remove duplicate edges
    # if there are fewer than 2 unique bins, assign all to the first label
    if len(unique_bins) < 2:
        df[f'{column}_cat'] = labels[0]
        return df

    # Adjust labels to match number of bins
    n_bins = len(unique_bins)
    n_labels_needed = n_bins - 1

    if n_labels_needed > len(labels):
        labels = labels + \
            [f"cat_{i}" for i in range(len(labels), n_labels_needed)]
    else:
        labels = labels[:n_labels_needed]
    df[f'{column}_cat'] = pd.cut(
        df[column], bins=unique_bins, labels=labels, duplicates='drop')
    return df


# Backward-compatible alias used by earlier workflow versions.
create_precip_bins_percentile = create_precipitation_bins_percentile


def add_period_columns(df):
    """Add week and month labels (e.g., '2023-W05', '2023-01').

    Args:
        df (pd.DataFrame): must have a 'date' column

    Returns:
        pd.DataFrame: with 'week_label' and 'month_label' columns
    """
    iso = df['date'].dt.isocalendar()
    df['week_label'] = iso['year'].astype(str) + '-W' + \
        iso['week'].astype(str).str.zfill(2)
    df['month_label'] = df['date'].dt.to_period('M').astype(str)
    return df


def process_csvfile_rq2(filename, p1, p2, p3, logger):
    """Perform RQ2-specific preprocessing: percentile bucketing + period labels.

    Args:
        filename (str): path to the general-processed CSV
        p1, p2, p3 (float): percentile cut points
        logger (logging.Logger): logger instance

    Raises:
        ValueError: if the CSV hasn't been general-preprocessed or contains NaNs

    Returns:
        pd.DataFrame: processed dataframe
    """
    # a Basic sanity check: file name contains 'processed'
    if 'processed' not in filename:
        logger.warning(
            f"Filename '{filename}' does not contain 'processed' – proceeding anyway, but content will be checked.")

    # Check the actual column set from the general preprocessing
    try:
        header_cols = set(pd.read_csv(filename, nrows=0).columns)
    except Exception as e:
        logger.error(f"Could not read CSV header: {e}")
        raise ValueError(f"Could not read file {filename}: {e}")

    missing = GENERAL_PREPROCESSING_COLUMNS - header_cols
    if missing:
        logger.error(
            f"Missing general preprocessing columns: {
                sorted(missing)}")
        raise ValueError(
            f"Input CSV missing columns added by general preprocessing: {
                sorted(missing)}. "
            f"Run dataprocessing.py on this file first."
        )

    # Read the required columns
    try:
        df = pd.read_csv(filename, usecols=COLUMNS)
    except Exception as e:
        logger.error(f"Could not read CSV columns: {e}")
        raise ValueError(
            f"Could not read columns {COLUMNS} from {filename}: {e}")

    # Verify audio features have no NaNs, they should have been median-filled
    cols_with_nans = [c for c in AUDIO_FEATURE_COLS if df[c].isna().any()]
    if cols_with_nans:
        logger.error(f"Audio columns with NaNs: {cols_with_nans}")
        raise ValueError(
            f"Audio feature columns still contain missing values that the general "
            f"preprocessing step should have median-filled: {cols_with_nans}. "
            f"This file may not be fully processed."
        )

    # Convert date and drop any remaining rows with NaNs in essential columns
    df['date'] = pd.to_datetime(df['date'])
    initial_len = len(df)
    df.dropna(inplace=True)
    dropped = initial_len - len(df)
    if dropped:
        logger.info(
            f"Dropped {dropped} rows with missing values (besides audio features)")

    logger.debug(f"Data loaded: {df.head()}")

    # Apply percentile bucketing
    df = create_precipitation_bins_percentile(
        df, 'precipitation_mm', p1, p2, p3)
    logger.debug(
        f"Bucketing applied with percentiles {p1}, {p2}, {p3}: {
            df.head()}")

    # Add period columns
    df = add_period_columns(df)
    logger.debug(f"Period columns added: {df.head()}")

    return df


def main(csvfile: str, outfile: str, p1: float,
         p2: float, p3: float, log_file: str):
    logger = setup_logger(log_file)
    logger.info(f"Starting RQ2 preprocessing on {csvfile}")

    if not csvfile.endswith('.csv'):
        logger.error(f"Input file {csvfile} is not a CSV")
        raise OSError("Input file must have .csv extension")

    try:
        df_processed = process_csvfile_rq2(csvfile, p1, p2, p3, logger)
        df_processed.to_csv(outfile, index=False)
        logger.info(f"Saved processed CSV to {outfile}")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

    logger.info("Finished RQ2 preprocessing")
    return df_processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Applies RQ2-specific preprocessing: 4-bin precipitation bucketing "
        "using configurable percentiles, plus week/month labels.")
    parser.add_argument('-i', '--processedcsvfile', required=True,
                        help='General-processed CSV file to process')
    parser.add_argument('-o', '--outfile', required=True,
                        help='Output CSV file name')
    parser.add_argument('--p1', type=float, default=25.0,
                        help='First percentile cut point (default: 25)')
    parser.add_argument('--p2', type=float, default=50.0,
                        help='Second percentile cut point (default: 50)')
    parser.add_argument('--p3', type=float, default=75.0,
                        help='Third percentile cut point (default: 75)')
    parser.add_argument(
        '--log',
        type=str,
        default=f'{_HERE}/logs/dataprocessing_w2.log',
        help='Path to log file (default: logs/dataprocessing_w2.log)')

    args = parser.parse_args()

    # Validate percentiles order
    if not (0 < args.p1 < args.p2 < args.p3 < 100):
        parser.error(
            f"Percentiles must be strictly increasing and between 0 and 100. Got {
                args.p1}, {
                args.p2}, {
                args.p3}")

    main(
        args.processedcsvfile,
        args.outfile,
        args.p1,
        args.p2,
        args.p3,
        args.log)
