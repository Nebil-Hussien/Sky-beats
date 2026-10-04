#!/usr/bin/env python3
"""
datapreprocessing_w7.py
-------------
A modular preprocessing component for the weather-music pipeline.
Handles daily data aggregation and calendar feature engineering.
"""

import pandas as pd
import holidays


def add_calendar_features(df, date_col='date'):
    """
    Takes a dataframe and appends day_of_week, is_weekend, month,
    and German public holidays based on a date column.
    """
    # Force date conversion
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Extract structural dates
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x in [5, 6] else 0)
    df['month'] = df[date_col].dt.month

    # Initialize German holiday locator dynamically across years in dataset
    years = df[date_col].dt.year.unique()
    de_holidays = holidays.Germany(years=list(years))

    # Flag holiday matches
    df['is_holiday'] = df[date_col].apply(lambda x: 1 if x in de_holidays else 0)

    return df


def generate_daily_averages(song_level_df, weather_df):
    """
    Example placeholder function for your aggregation logic:
    Combines track streams to a single daily average row and merges with weather.
    """
    # Your grouping logic goes here:
    # daily_songs = song_level_df.groupby('date').mean(numeric_only=True)
    # merged = daily_songs.merge(weather_df, on='date')
    pass


if __name__ == "__main__":
    import argparse

    # 1. Grab the file paths passed by Snakemake
    parser = argparse.ArgumentParser(description="Run preprocessing steps")
    parser.add_argument("input_csv", help="Path to raw merged daily file")
    parser.add_argument("output_csv", help="Path to save processed daily file")
    args = parser.parse_args()

    print(f"Reading raw data from: {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    # 2. Run feature engineering (input is already validated/imputed by
    #    the rq7_dataprocessing rule, via the shared dataprocessing.py --schema daily)
    print("Engineering calendar and holiday features...")
    df_clean = add_calendar_features(df, date_col='date')

    # 3. Save the file where Snakemake expects it
    df_clean.to_csv(args.output_csv, index=False)
    print(f"Success! Processed data saved to: {args.output_csv}")
