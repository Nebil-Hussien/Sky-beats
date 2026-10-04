#!/usr/bin/env python3
"""Join Spotify and DWD weather data and create song- and daily-grain CSVs."""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--spotify", type=Path, help="Override Spotify input CSV")
    parser.add_argument("--weather", type=Path, help="Override DWD input CSV")
    parser.add_argument("--song-output", type=Path, help="Override song-level output")
    parser.add_argument("--daily-output", type=Path, help="Override daily output")
    return parser.parse_args()


def main(config_path, spotify_path=None, weather_path=None,
         song_output=None, daily_output=None):
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    root = Path(__file__).resolve().parents[2]
    data = root / config["paths"]["output_dir"]
    spotify_path = spotify_path or data / config["spotify"]["output_file"]
    weather_path = weather_path or data / config["dwd"]["output_file"]
    song_output = song_output or data / config["merge"]["output_merged"]
    daily_output = daily_output or data / config["merge"]["output_daily"]

    spotify = pd.read_csv(spotify_path)
    weather = pd.read_csv(weather_path)
    spotify_required = {"date", "track_id", "streams", "valence", "energy",
                        "danceability", "tempo", "acousticness"}
    weather_required = {"date", "precipitation_mm", "sunshine_hours",
                        "temperature_mean_c"}
    if missing := spotify_required.difference(spotify.columns):
        raise ValueError(f"Spotify input is missing columns: {sorted(missing)}")
    if missing := weather_required.difference(weather.columns):
        raise ValueError(f"weather input is missing columns: {sorted(missing)}")
    spotify["date"] = pd.to_datetime(spotify["date"])
    weather["date"] = pd.to_datetime(weather["date"])
    spotify = spotify[spotify["date"].between(config["dates"]["start"], config["dates"]["end"])]
    songs = spotify.merge(weather, on="date", how="left")
    daily = (spotify.groupby("date").agg(
        avg_valence=("valence", "mean"), avg_energy=("energy", "mean"),
        avg_danceability=("danceability", "mean"), avg_tempo=("tempo", "mean"),
        avg_acousticness=("acousticness", "mean"),
        total_streams=("streams", lambda values: values.sum(min_count=1)),
        unique_songs=("track_id", "nunique"),
    ).reset_index().round(3).merge(weather, on="date", how="inner"))
    song_output.parent.mkdir(parents=True, exist_ok=True)
    daily_output.parent.mkdir(parents=True, exist_ok=True)
    songs.to_csv(song_output, index=False)
    daily.to_csv(daily_output, index=False)
    LOGGER.info("Saved %s song rows to '%s'", len(songs), song_output)
    LOGGER.info("Saved %s daily rows to '%s'", len(daily), daily_output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    main(args.config, args.spotify, args.weather, args.song_output, args.daily_output)
