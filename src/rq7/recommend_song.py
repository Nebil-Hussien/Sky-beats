#!/usr/bin/env python3
"""
weather_music_pipeline.py
-------------------------
RQ7: Can daily weather attributes predict music mood, and can this be
used to recommend songs?

This script contains PART B of the original pipeline:

PART B:
-------
Use the trained model's predicted mood profile to recommend songs
using k-NN similarity in audio-feature space.

(PART A -- training the weather -> mood model -- now lives in
train_model.py.)

Inputs:
    - trained model(s) from train_model.py
    - song-level dataset for recommendation

Outputs:
    - recommended songs

"Current weather" for the recommendation step is NOT passed as CLI flags
-- it's read from a small YAML file, e.g.:

    # current_weather.yaml
    precipitation: 5.0
    sunshine_hours: 1.5
    temperature_mean_c: 9.0

Usage:
    python recommend_songs.py model.pkl songs.csv \
        -r recommendations.csv \
        -w current_weather.yaml
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml
import joblib

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


# =========================
# LOGGER
# =========================
def setup_logger():
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "weather_music.log"
    logging.basicConfig(
        filename=log_path,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


# =========================
# VALIDATION
# =========================
def validate_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: missing columns: {missing}")


def load_current_weather(yaml_path):
    """
    Reads 'current weather' values from a YAML file instead of CLI flags,
    e.g.:
        precipitation: 5.0
        sunshine_hours: 1.5
        temperature_mean_c: 9.0

    This is the file Snakemake points to as an `input:` for this rule --
    edit the file's values to test a different day, no need to touch the
    Snakefile or pass CLI arguments.
    """
    required_keys = ["precipitation", "sunshine_hours", "temperature_mean_c", "is_weekend",
                     "day_of_week", "month", "is_holiday"]

    with open(yaml_path, "r") as f:
        weather = yaml.safe_load(f)

    missing = [k for k in required_keys if k not in weather]
    if missing:
        raise SystemExit(f"ERROR: current_weather YAML missing keys: {missing}")

    return {k: float(weather[k]) for k in required_keys}


# =========================
# PART B: RECOMMENDATION
# =========================
def build_song_index(songs_df, features):
    """
    Fits a MinMaxScaler + NearestNeighbors index on the song catalog.
    Scaling is required because features like tempo (~60-200) and
    valence (0-1) are on very different scales; without scaling, tempo
    would dominate the Euclidean distance just because its numbers are
    bigger, not because it's more important.
    """
    songs_df = songs_df.dropna(subset=features).reset_index(drop=True)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(songs_df[features])

    knn = NearestNeighbors(n_neighbors=10, metric="euclidean")
    knn.fit(X_scaled)

    return songs_df, scaler, knn


def recommend_songs(songs_df, scaler, knn, target_vector, k=5):
    """
    target_vector must be in the SAME feature order used to build the
    index, and gets scaled with the SAME scaler fitted on the catalog.
    """
    target_df = pd.DataFrame([target_vector], columns=scaler.feature_names_in_)
    target_scaled = scaler.transform(target_df)

    # Ask k-NN for more neighbors initially (e.g., 50) so we have enough
    # unique songs left over after dropping duplicates
    distances, indices = knn.kneighbors(target_scaled, n_neighbors=50)

    recs = songs_df.iloc[indices[0]].copy()
    recs["distance"] = distances[0]

    # Drop duplicate songs based on title and artist if those columns exist
    for col_check in [["title", "artist"], ["track_name", "artist_name"], ["title"], ["track_id"]]:
        valid_cols = [c for c in col_check if c in recs.columns]
        if valid_cols:
            recs = recs.drop_duplicates(subset=valid_cols)
            break

            # Check if 'popularity' exists to break ties; otherwise use 'track_name' or 'title'
    tie_breaker = None
    for col in ["popularity", "track_name", "title", "track_id"]:
        if col in recs.columns:
            tie_breaker = col
            break

    if tie_breaker == "popularity":
        # Sort by closest distance first, then highest popularity second
        recs = recs.sort_values(by=["distance", "popularity"], ascending=[True, False])
    elif tie_breaker:
        # Sort alphabetically if popularity isn't available
        recs = recs.sort_values(by=["distance", tie_breaker], ascending=[True, True])
    else:
        # Fallback to absolute index order if no metadata keys match
        recs = recs.sort_values(by="distance", ascending=True)
    # -------------------------------------------------------------

    # Return exactly the top k unique recommendations
    return recs.head(k)


# =========================
# MAIN PIPELINE
# =========================
def main(model_path, songs_csv, rec_path, current_weather_yaml):

    logger = setup_logger()

    songs_df = pd.read_csv(songs_csv)

    song_features = ["valence", "energy", "danceability", "tempo"]

    validate_columns(songs_df, song_features)

    # =========================
    # PART B: RECOMMENDATION (using best model)
    # =========================
    logger.info(f"Reading current weather from {current_weather_yaml}")
    current_weather = load_current_weather(current_weather_yaml)

    print(f"\nUsing current weather from {current_weather_yaml}: {current_weather}")

    sample_weather = pd.DataFrame([{
        "precipitation_mm": current_weather["precipitation"],
        "sunshine_hours": current_weather["sunshine_hours"],
        "temperature_mean_c": current_weather["temperature_mean_c"],
        "is_weekend": current_weather["is_weekend"],
        "day_of_week": current_weather["day_of_week"],
        "month": current_weather["month"],
        "is_holiday": current_weather["is_holiday"]
    }])

    models = joblib.load(model_path)

    target_vector = [
        models["avg_valence"].predict(sample_weather)[0],
        models["avg_energy"].predict(sample_weather)[0],
        models["avg_danceability"].predict(sample_weather)[0],
        models["avg_tempo"].predict(sample_weather)[0],
    ]

    print(f"\nPredicted mood profile: "
          f"valence={target_vector[0]:.3f}, energy={target_vector[1]:.3f}, "
          f"danceability={target_vector[2]:.3f}, tempo={target_vector[3]:.1f}")

    songs_indexed, scaler, knn = build_song_index(songs_df, song_features)
    recommendations = recommend_songs(songs_indexed, scaler, knn, target_vector)

    recommendations.to_csv(rec_path, index=False)

    print("\n=== PART B RESULTS (Top Recommendations) ===")
    display_cols = ["distance"]
    for col in ["title", "track_name", "artist", "artist_name"]:
        if col in recommendations.columns:
            display_cols.append(col)
    display_cols.extend(song_features)
    print(recommendations[display_cols])


# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ7: Weather -> Mood -> Music Recommendation Pipeline"
    )

    parser.add_argument("model", help="Trained model path (from train_model.py)")
    parser.add_argument("songs_csv", help="Song-level dataset (for recommendation)")

    parser.add_argument("-r", "--recs", required=True, help="Output recommendations CSV")

    parser.add_argument("-w", "--weather", required=True,
                        help="YAML file with current-day weather "
                              "(precipitation, sunshine_hours, temperature_mean_c)")
    args = parser.parse_args()

    main(
        args.model,
        args.songs_csv,
        args.recs,
        current_weather_yaml=args.weather,
    )
