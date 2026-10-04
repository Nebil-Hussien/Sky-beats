#!/usr/bin/env python3
"""
compare_models_w8.py
--------------------
RQ8 analysis (FR-15): does daily weather predict chart mood *beyond* seasonal
effects?

For each estimator, two regressions of the daily mean target (default: valence)
are fit and evaluated on the SAME held-out days:

  baseline : month-of-year only (one-hot encoded)   -> pure seasonal model
  weather  : month one-hot + daily weather columns  -> seasonal + weather model

The split is CHRONOLOGICAL (last `test_fraction` of days held out) so the test
set is genuinely out-of-sample and no calendar leakage occurs; consecutive
chart days are highly autocorrelated, so a random split would be optimistic.

Weather "adds real predictive power" only if the weather model beats the
baseline on the held-out days (delta_r2 > 0).

Output: a CSV with one row per (model, feature_set):
    model, feature_set, n_train, n_test, r2, mae

Usage
-----
  python compare_models_w8.py <daily_csv> -o <output_csv>
      [--target valence] [--weather-features sunshine_hours,precipitation_mm,...]
      [--models ridge,random_forest] [--test-fraction 0.2] [--seed 42]
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

# Each value is a factory returning a fresh, unfitted model (same pattern as
# Workflow 7's MODEL_REGISTRY). `seed` keeps stochastic models reproducible.
MODEL_REGISTRY = {
    "ridge": lambda seed: Ridge(alpha=1.0),
    "random_forest": lambda seed: RandomForestRegressor(
        n_estimators=300, max_depth=5, min_samples_split=10, random_state=seed
    ),
}


def setup_logger():
    """Configure logging to ./logs relative to this script (CWD-independent)."""
    log_path = Path(__file__).resolve().parent / "logs" / "compare_models_w8.log"
    logging.basicConfig(filename=log_path, filemode="w", level=logging.DEBUG)
    return logging.getLogger(__name__)


def build_features(df, weather_features):
    """Build the two design matrices: month-only baseline and month + weather.

    Args:
        df (pd.DataFrame): daily rows with 'month' and the weather columns
        weather_features (list[str]): weather column names to add

    Returns:
        dict[str, pd.DataFrame]: {'baseline': X_base, 'weather': X_weather}
    """
    month_dummies = pd.get_dummies(df["month"].astype(int), prefix="month")
    baseline = month_dummies
    weather = pd.concat([month_dummies, df[weather_features]], axis=1)
    return {"baseline": baseline, "weather": weather}


def chronological_split(n_rows, test_fraction):
    """Return (train_index, test_index) with the last test_fraction held out."""
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    n_test = max(1, int(round(n_rows * test_fraction)))
    n_train = n_rows - n_test
    if n_train < 1:
        raise ValueError(f"not enough rows ({n_rows}) for test_fraction={test_fraction}")
    return range(n_train), range(n_train, n_rows)


def compare_models(df, target, weather_features, models, test_fraction, seed):
    """Fit baseline vs weather feature sets for each model; evaluate held out.

    Args:
        df (pd.DataFrame): daily rows, SORTED BY DATE, with 'month', target
            and weather columns
        target (str): column to predict (e.g. 'valence')
        weather_features (list[str]): weather columns for the weather model
        models (list[str]): keys into MODEL_REGISTRY
        test_fraction (float): fraction of (final) days held out
        seed (int): random seed for stochastic models

    Returns:
        pd.DataFrame: one row per (model, feature_set) with r2 and mae
    """
    unknown = set(models) - set(MODEL_REGISTRY)
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}; "
                         f"available: {sorted(MODEL_REGISTRY)}")

    df = df.dropna(subset=["month", target] + weather_features).reset_index(drop=True)
    train_idx, test_idx = chronological_split(len(df), test_fraction)
    feature_sets = build_features(df, weather_features)
    y = df[target]

    rows = []
    for model_name in models:
        for set_name, X in feature_sets.items():
            model = MODEL_REGISTRY[model_name](seed)
            model.fit(X.iloc[list(train_idx)], y.iloc[list(train_idx)])
            predictions = model.predict(X.iloc[list(test_idx)])
            r2 = r2_score(y.iloc[list(test_idx)], predictions)
            mae = mean_absolute_error(y.iloc[list(test_idx)], predictions)
            LOGGER.debug("%s/%s: r2=%.4f mae=%.4f", model_name, set_name, r2, mae)
            rows.append({"model": model_name, "feature_set": set_name,
                         "n_train": len(train_idx), "n_test": len(test_idx),
                         "r2": r2, "mae": mae})
    return pd.DataFrame(rows)


def main(input_csv, output_csv, target, weather_features, models, test_fraction, seed):
    setup_logger()
    LOGGER.info("Loading %s", input_csv)
    try:
        df = pd.read_csv(input_csv, parse_dates=["date"])
    except FileNotFoundError:
        raise SystemExit(f"ERROR: input file not found: {input_csv}")
    except pd.errors.EmptyDataError:
        raise SystemExit(f"ERROR: input file is empty: {input_csv}")

    required = {"date", "month", target, *weather_features}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: input missing required columns: {sorted(missing)}")

    df = df.sort_values("date").reset_index(drop=True)
    try:
        result = compare_models(df, target, weather_features, models,
                                test_fraction, seed)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}")

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    LOGGER.info("Saved model comparison -> %s", output_csv)

    print(result.to_string(index=False))
    for model_name in models:
        sub = result[result["model"] == model_name].set_index("feature_set")
        delta = sub.loc["weather", "r2"] - sub.loc["baseline", "r2"]
        print(f"{model_name}: delta R2 (weather - baseline) = {delta:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ8: compare weather regression vs season/month-only baseline (FR-15)")
    parser.add_argument("input_csv", help="daily CSV from datapreprocessing_w8.py")
    parser.add_argument("-o", "--output", required=True, help="output metrics CSV")
    parser.add_argument("--target", default="valence",
                        help="daily audio feature to predict (default: valence)")
    parser.add_argument("--weather-features",
                        default="sunshine_hours,precipitation_mm,temperature_mean_c",
                        help="comma-separated weather columns for the weather model")
    parser.add_argument("--models", default="ridge,random_forest",
                        help="comma-separated estimators (available: %s)"
                             % ",".join(sorted(MODEL_REGISTRY)))
    parser.add_argument("--test-fraction", type=float, default=0.2,
                        help="chronological hold-out fraction (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for stochastic models (default: 42)")
    args = parser.parse_args()
    main(args.input_csv, args.output, args.target,
         [f.strip() for f in args.weather_features.split(",") if f.strip()],
         [m.strip() for m in args.models.split(",") if m.strip()],
         args.test_fraction, args.seed)
