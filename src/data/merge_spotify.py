#!/usr/bin/env python3
"""Join Spotify charts and audio features using Spotify track IDs."""

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
    parser.add_argument("--charts", type=Path, help="Override charts input CSV")
    parser.add_argument("--features", type=Path, help="Override audio-features input CSV")
    parser.add_argument("--output", type=Path, help="Override merged output CSV")
    return parser.parse_args()


def main(config_path, charts_path=None, features_path=None, output=None):
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    root = Path(__file__).resolve().parents[2]
    raw = root / config["paths"]["raw_dir"]
    charts_path = charts_path or raw / config["spotify"]["charts_file"]
    features_path = features_path or raw / config["spotify"]["features_file"]
    output = output or root / config["paths"]["output_dir"] / config["spotify"]["output_file"]

    charts = pd.read_csv(charts_path)
    chart_columns = {"region", "date", "title", "artist", "rank", "streams", "trend", "url"}
    missing = chart_columns.difference(charts.columns)
    if missing:
        raise ValueError(f"charts input is missing columns: {sorted(missing)}")
    charts = charts.loc[charts["region"] == config["spotify"]["region"],
                        ["date", "title", "artist", "rank", "streams", "trend", "url"]].copy()
    charts["track_id"] = charts.pop("url").str.split("/track/").str[-1]
    features = pd.read_csv(features_path)
    feature_columns = set(config["spotify"]["audio_features"])
    missing = feature_columns.difference(features.columns)
    if missing:
        raise ValueError(f"features input is missing columns: {sorted(missing)}")
    features = features[list(config["spotify"]["audio_features"])]
    features = features.drop_duplicates(subset=["track_id"])
    merged = charts.merge(features, on="track_id", how="left")
    required = ["danceability", "energy", "tempo", "valence", "acousticness"]
    unmatched = merged[required[0]].isna().sum()
    merged = merged.dropna(subset=required)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    LOGGER.info("Saved %s rows to '%s'; dropped %s unmatched rows", len(merged), output, unmatched)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    main(args.config, args.charts, args.features, args.output)
