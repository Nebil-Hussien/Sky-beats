#!/usr/bin/env python3
"""
weather_music_pipeline.py
-------------------------
RQ7: Can daily weather attributes predict music mood, and can this be
used to recommend songs?

This script contains PART A of the original pipeline:

PART A:
-------
Train a Random Forest model to predict daily music mood (valence,
energy, danceability, tempo) from weather attributes.

(PART B -- recommending songs from the trained model's predicted mood
profile -- now lives in recommend_songs.py.)

Inputs:
    - dataset with weather + music mood (daily aggregated)

Outputs:
    - trained model(s)
    - evaluation metrics (per target)

Usage:
    python train_model.py data.csv \
        -m model.pkl \
        -o metrics.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error


# =========================
# MODEL REGISTRY
# =========================
# Add/remove entries here to control which models can be tried.
# Each value is a zero-arg function returning a fresh, unfitted model.
MODEL_REGISTRY = {
    "random_forest_depth5": lambda: RandomForestRegressor(
        n_estimators=300, max_depth=5, min_samples_split=10, random_state=42
    ),
    "random_forest_depth15": lambda: RandomForestRegressor(
        n_estimators=300, max_depth=15, min_samples_split=2, random_state=42
    ),
    "ridge": lambda: Ridge(alpha=1.0),
    "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(
        max_depth=5, random_state=42
    ),
}


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


# =========================
# PART A: TRAIN MODELS
# =========================
def train_weather_models(df, features, targets, model_name="random_forest_depth5"):
    """
    Trains one RandomForestRegressor PER mood target (valence, energy,
    danceability, tempo), since a single RandomForestRegressor predicts
    only one output column at a time.

    Returns:
        models: dict[target_name] -> fitted model
        metrics: dict[target_name] -> {"r2": ..., "mae": ...}
        importances: dict[target_name] -> {feature_name: importance}
    """

    if model_name not in MODEL_REGISTRY:
        raise SystemExit(
            f"ERROR: unknown model_name '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    X = df[features]

    models = {}
    metrics = {}
    importances = {}

    for target in targets:
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        model = MODEL_REGISTRY[model_name]()

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        metrics[target] = {
            "r2": r2_score(y_test, preds),
            "mae": mean_absolute_error(y_test, preds)
        }

        if hasattr(model, "feature_importances_"):
            importances[target] = dict(zip(features, model.feature_importances_))
        elif hasattr(model, "coef_"):
            importances[target] = dict(zip(features, model.coef_))
        else:
            importances[target] = {}

        models[target] = model

    return models, metrics, importances


def compare_models(df, features, targets, model_names):
    rows = []
    all_results = {}
    for model_name in model_names:
        models, metrics, importances = train_weather_models(
            df, features, targets, model_name=model_name
        )
        all_results[model_name] = (models, metrics, importances)
        for target, m in metrics.items():
            rows.append({"model": model_name, "target": target, "r2": m["r2"], "mae": m["mae"]})
    return pd.DataFrame(rows), all_results


# =========================
# MAIN PIPELINE
# =========================
def main(weather_music_csv, model_path, metrics_path, model_types):

    logger = setup_logger()

    df = pd.read_csv(weather_music_csv, sep=None, engine='python')

    weather_features = ["precipitation_mm", "sunshine_hours", "temperature_mean_c", "is_weekend", "day_of_week", "month", "is_holiday"]
    mood_targets = ["avg_valence", "avg_energy", "avg_danceability", "avg_tempo"]

    validate_columns(df, weather_features + mood_targets)

    # =========================
    # PART A: TRAIN + COMPARE MODELS
    # =========================
    logger.info(f"Training and comparing models: {model_types}")

    comparison_df, all_results = compare_models(df, weather_features, mood_targets, model_types)

    pivot = comparison_df.pivot(index="target", columns="model", values=["r2", "mae"])
    logger.info("Model comparison (R2 / MAE per target):\n%s", pivot)

    comparison_df.to_csv(metrics_path, index=False)
    logger.info(f"Comparison metrics saved to {metrics_path}")

    # Pick the best model overall by mean R2 across targets, for Part B
    mean_r2_by_model = comparison_df.groupby("model")["r2"].mean().sort_values(ascending=False)
    best_model_name = mean_r2_by_model.index[0]
    logger.info("Best model by mean R2 across targets: %s (mean R2=%.4f)",
                best_model_name, mean_r2_by_model.iloc[0])

    models, metrics, importances = all_results[best_model_name]
    joblib.dump(models, model_path)

    logger.info("Feature importances for %s:", best_model_name)
    for target, feats in importances.items():
        # Sort features so the most important one is at the top
        sorted_feats = sorted(feats.items(), key=lambda x: x[1], reverse=True)
        logger.info("  %s: %s", target,
                    ", ".join(f"{f}={i:.4f}" for f, i in sorted_feats))


# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ7: Weather -> Mood -> Music Recommendation Pipeline"
    )

    parser.add_argument("weather_music_csv", help="Merged weather + mood dataset (daily)")

    parser.add_argument("-m", "--model", required=True, help="Output model path")
    parser.add_argument("-o", "--metrics", required=True, help="Output metrics CSV")

    parser.add_argument("--model-type", nargs="+",
                        default=list(MODEL_REGISTRY.keys()),
                        choices=list(MODEL_REGISTRY.keys()),
                        help="One or more model types to train and compare. "
                        f"Default: all of {list(MODEL_REGISTRY.keys())}")
    args = parser.parse_args()

    main(
        args.weather_music_csv,
        args.model,
        args.metrics,
        model_types=args.model_type
    )
