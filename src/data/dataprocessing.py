#!/usr/bin/env python3
"""The script performs a general processing of the input dataset for the Sky Beats Spotify/weather dataset."""

import argparse
import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

REQUIRED_COLUMNS = {
    "date",
    "streams",
    "track_id",
    "danceability",
    "energy",
    "tempo",
    "valence",
    "acousticness",
    "popularity",
    "temperature_mean_c",
    "precipitation_mm",
    "sunshine_hours",
}

NUMERIC_COLUMNS = [
    "streams", "energy", "danceability", "tempo", "valence", "acousticness", "popularity",
    "temperature_mean_c", "precipitation_mm", "sunshine_hours",
]

FILLNA_COLUMNS = ["streams", "energy", "danceability", "tempo", "valence", "acousticness", "popularity"]

DAILY_REQUIRED_COLUMNS = {
    "date",
    "avg_valence",
    "avg_energy",
    "avg_danceability",
    "avg_tempo",
    "avg_acousticness",
    "total_streams",
    "unique_songs",
    "temperature_mean_c",
    "precipitation_mm",
    "sunshine_hours",
}

DAILY_NUMERIC_COLUMNS = [
    "avg_valence", "avg_energy", "avg_danceability", "avg_tempo", "avg_acousticness",
    "total_streams", "unique_songs", "temperature_mean_c", "precipitation_mm", "sunshine_hours",
]

# Registry of known dataset grains, selectable via the --schema CLI flag.
SCHEMAS = {
    "song": (REQUIRED_COLUMNS, NUMERIC_COLUMNS, FILLNA_COLUMNS),
    "daily": (DAILY_REQUIRED_COLUMNS, DAILY_NUMERIC_COLUMNS, DAILY_NUMERIC_COLUMNS),
}


def convert_to_datetime(date_value, fmt="%Y-%m-%d"):
    """Parse date column string into a pandas datetime object for possible future manupliation."""
    if pd.isna(date_value):
        return pd.NaT
    if isinstance(date_value, pd.Timestamp):
        return date_value.to_pydatetime()
    return pd.to_datetime(date_value, format=fmt, errors="coerce")


def get_season(month):
    """Setting up numerical months to a meteorological seasons."""
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"


def validate_file_path(filename):
    """Validate that the input CSV file exists and is not empty."""
    if filename is None:
        raise FileNotFoundError("Input file path cannot be invalid/None")

    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {filename}")
    if path.suffix.lower() != ".csv":
        raise OSError("Provide a csv file with .csv extension")
    if path.stat().st_size == 0:
        raise ValueError(f"File is empty: {filename}")
    return path


def validate_dataframe(df, required_columns=None):
    """Ensure the dataframe has the expected schema and a contiguous date range."""
    if df is None:
        raise ValueError("Input dataframe cannot be None")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame")

    required_columns = required_columns if required_columns is not None else REQUIRED_COLUMNS
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("Input dataframe contains invalid date values")

    normalized_dates = pd.DatetimeIndex(parsed_dates.dt.normalize().dropna().sort_values())
    if normalized_dates.empty:
        raise ValueError("Input dataframe contains invalid date values")

    expected_dates = pd.date_range(normalized_dates[0], normalized_dates[-1], freq="D")
    missing_dates = [date.strftime("%Y-%m-%d") for date in expected_dates.difference(normalized_dates)]
    if missing_dates:
        raise ValueError(
            "Input dataframe dates are not contiguous; missing dates: " + ", ".join(missing_dates[:10])
        )

    return df


def preprocess_dataframe(df, required_columns=None, numeric_columns=None, fillna_columns=None):
    """Clean the Sky Beats merged dataset for analysis."""
    required_columns = required_columns if required_columns is not None else REQUIRED_COLUMNS
    numeric_columns = numeric_columns if numeric_columns is not None else NUMERIC_COLUMNS
    fillna_columns = fillna_columns if fillna_columns is not None else FILLNA_COLUMNS

    validate_dataframe(df, required_columns)

    cleaned = df.copy()
    cleaned["date"] = cleaned["date"].apply(convert_to_datetime)
    cleaned = cleaned.dropna(subset=["date"])
    cleaned = cleaned.sort_values("date").reset_index(drop=True)
    cleaned["year"] = cleaned["date"].dt.year
    cleaned["month"] = cleaned["date"].dt.month
    cleaned["day"] = cleaned["date"].dt.day
    cleaned["weekday"] = cleaned["date"].dt.day_name()
    cleaned["is_weekend"] = cleaned["date"].dt.dayofweek.isin([5, 6])
    cleaned["season"] = cleaned["month"].apply(get_season)

    for column in numeric_columns:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    for column in fillna_columns:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    if "track_id" in cleaned.columns:
        cleaned["track_id"] = cleaned["track_id"].astype(str).str.strip()

    if "title" in cleaned.columns:
        cleaned["title"] = cleaned["title"].fillna("")

    if "artist" in cleaned.columns:
        cleaned["artist"] = cleaned["artist"].fillna("")

    return cleaned


def process_csvfile(filename, required_columns=None, numeric_columns=None, fillna_columns=None):
    """Load a CSV file and return the preprocessed dataframe."""
    path = validate_file_path(filename)
    df = pd.read_csv(path, engine="python")
    LOGGER.debug("Preprocessing merged Spotify/weather dataset")
    return preprocess_dataframe(df, required_columns, numeric_columns, fillna_columns)


def main(csvfile: str, outfile: str, schema: str = "song"):
    log_path = Path(__file__).resolve().parent / "logs" / "dataprocessing.log"
    logging.basicConfig(filename=log_path, filemode="w")
    LOGGER.info("Started preprocessing (schema=%s)", schema)
    required_columns, numeric_columns, fillna_columns = SCHEMAS[schema]
    df_processed = process_csvfile(csvfile, required_columns, numeric_columns, fillna_columns)
    LOGGER.info("Saving processed csv")
    df_processed.to_csv(outfile, index=False)
    LOGGER.info("End")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the Sky Beats Spotify/weather dataset")
    parser.add_argument("-i", "--csvfile", required=True, type=str, help="input csv file")
    parser.add_argument("-o", "--outfile", required=True, type=str, help="output csv file")
    parser.add_argument("--schema", choices=list(SCHEMAS), default="song",
                        help="dataset grain: 'song' (per-track rows) or 'daily' (per-day aggregates)")
    args = parser.parse_args()
    main(args.csvfile, args.outfile, args.schema)
