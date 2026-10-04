#!/usr/bin/env python3
"""
RQ5 preprocessing: derives a meteorological season column from the date.

The script processes the general-processed Sky Beats dataset for RQ5
(seasonal energy analysis). It adds a 'season' column based on meteorological
seasons (Winter: Dec-Feb, Spring: Mar-May, Summer: Jun-Aug, Autumn: Sep-Nov).

Configurable: log file, log level.
"""

import os
import sys
import argparse
import logging
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))

# Columns required from the general-processed CSV
# RQ5 focuses on energy, but we include all audio features for completeness
COLUMNS = ['date', 'temperature_mean_c', 'precipitation_mm',
           'sunshine_hours', 'acousticness', 'energy', 'tempo',
           'valence', 'danceability']

AUDIO_FEATURE_COLS = ['acousticness', 'energy', 'tempo', 'valence', 'danceability']

# Columns added by general preprocessing
GENERAL_PREPROCESSING_COLUMNS = {'year', 'month', 'day', 'weekday',
                                 'is_weekend', 'season'}

# Meteorological season definition (Northern Hemisphere)
MONTH_TO_SEASON = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn',
}
SEASON_ORDER = ['Winter', 'Spring', 'Summer', 'Autumn']


def setup_logger(log_file, log_level=logging.DEBUG):
    """Configure a logger with file and console output."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('datapreprocessing_w5')
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


def add_season_column(df):
    """
    Derive a meteorological season from the date column.

    The season is stored as an ordered categorical with Winter as the
    reference level for downstream regression encoding.

    Args:
        df (pd.DataFrame): DataFrame with a 'date' column.

    Returns:
        pd.DataFrame: DataFrame with an added 'season' column.
    """
    df = df.copy()
    df['season'] = df['date'].dt.month.map(MONTH_TO_SEASON)
    df['season'] = pd.Categorical(df['season'], categories=SEASON_ORDER,
                                  ordered=False)
    return df


def process_csvfile_rqenergy(filename, logger):
    """
    Perform RQ5 preprocessing: derive meteorological season.

    Args:
        filename (str): Path to the general-processed CSV file.
        logger (logging.Logger): Logger instance.

    Raises:
        ValueError: If the CSV is empty, unreadable, or missing required columns.
        KeyError: If required columns are missing.

    Returns:
        pd.DataFrame: Processed DataFrame with 'season' column.
    """
    logger.info(f"Processing RQ5 (season derivation) on {filename}")

    # Check general preprocessing columns exist
    try:
        header_cols = set(pd.read_csv(filename, nrows=0).columns)
    except Exception as e:
        logger.error(f"Could not read header from {filename}: {e}")
        raise ValueError(f"Could not read file {filename}: {e}")

    missing_general = GENERAL_PREPROCESSING_COLUMNS - header_cols
    if missing_general:
        logger.error(f"Missing general preprocessing columns: {sorted(missing_general)}")
        raise ValueError(
            f"Input CSV missing general preprocessing columns: {sorted(missing_general)}. "
            "Run dataprocessing.py on this file first."
        )

    # Read the data
    try:
        df = pd.read_csv(filename)
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file is empty: {filename}")
        raise ValueError(f"CSV file is empty: {filename}")
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {filename} ({e})")
        raise ValueError(f"Error parsing CSV file: {filename}")

    # Convert date column
    df['date'] = pd.to_datetime(df['date'])

    # Verify audio features have no NaNs (should be median-filled by general preprocessing)
    cols_with_nans = [c for c in AUDIO_FEATURE_COLS if c in df.columns and df[c].isna().any()]
    if cols_with_nans:
        logger.warning(f"Audio columns with NaNs: {cols_with_nans} — will be dropped.")
        # They will be dropped in the next step

    # Drop rows with missing values
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with missing values")

    if len(df) == 0:
        logger.error("No data remaining after cleaning")
        raise ValueError("No valid data for processing.")

    # Add season column
    df = add_season_column(df)
    logger.debug(f"Season column added: {df['season'].value_counts().to_dict()}")

    return df


def main(csvfile, outfile, log_file, log_level):
    logger = setup_logger(log_file, log_level)
    logger.info(f"Starting RQ5 preprocessing on {csvfile}")

    if not csvfile.endswith('.csv'):
        logger.error(f"Input file {csvfile} is not a CSV")
        raise OSError("Input file must have .csv extension")

    # Create output directory if needed
    out_dir = os.path.dirname(outfile)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        df_processed = process_csvfile_rqenergy(csvfile, logger)
        df_processed.to_csv(outfile, index=False)
        logger.info(f"Saved processed CSV to {outfile}")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

    logger.info("Finished RQ5 preprocessing")
    return df_processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ5 preprocessing: derive meteorological season from date."
    )
    parser.add_argument('-i', '--processedcsvfile', required=True,
                        help="Input CSV (must have general preprocessing)")
    parser.add_argument('-o', '--outfile', required=True,
                        help="Output CSV file name")
    parser.add_argument('--log', type=str,
                        default=f"{_HERE}/logs/dataprocessing_w5.log",
                        help="Path to log file")
    parser.add_argument('--log-level', type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='DEBUG', help="Logging level (default: DEBUG)")

    args = parser.parse_args()

    main(args.processedcsvfile, args.outfile, args.log, args.log_level)
