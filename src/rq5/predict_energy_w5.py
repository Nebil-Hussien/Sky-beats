#!/usr/bin/env python3
"""
Answers RQ5 directly: "How accurately can a combination of daily
temperature and sunshine duration predict the overall Energy level
of the German Top 100?"

This is a predictive-accuracy question, not an explanatory one — it
asks how well temperature + sunshine alone can forecast energy on a
day the model hasn't seen, not whether their effect is statistically
significant or varies by season (that question belongs to a separate
seasonal-interaction analysis; see regression_energy.py).

Design choices, each addressing a specific way a naive train/test
split would overstate accuracy on daily chart data:

  - TimeSeriesSplit cross-validation (not a random split): daily
    charts are autocorrelated (largely the same songs from one day
    to the next), so a random split would leak near-duplicate days
    into both train and test, inflating the score. TimeSeriesSplit
    only ever trains on the past and tests on the future.

  - Two baselines, evaluated on the exact same folds: a mean baseline
    (predict the training mean every time) and a persistence baseline
    (predict yesterday's energy). Without these, a raw R²/RMSE number
    has no reference point for what "accurate" means here.

  - Linear vs. gradient boosting, same folds: RQ5 asks how accurately
    energy *can* be predicted, not just how accurately a linear model
    predicts it. Comparing to a flexible nonlinear model checks
    whether a linear fit is close to the achievable ceiling or is
    leaving predictable structure on the table.

Only temperature_mean_c and sunshine_hours are used as predictors,
matching RQ5 as written — precipitation and season are intentionally
excluded here (they belong to the separate seasonal-interaction
analysis in regression_energy.py).

Configurable: number of CV folds, model hyperparameters, log file,
log level.
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_HERE = os.path.dirname(os.path.abspath(__file__))

# Default predictors and target
PREDICTORS = ['temperature_mean_c', 'sunshine_hours']
TARGET = 'energy'
REQUIRED_COLUMNS = ['date', TARGET] + PREDICTORS

# Columns added by general preprocessing
GENERAL_PREPROCESSING_COLUMNS = {'year', 'month', 'day', 'weekday',
                                 'is_weekend', 'season'}

MIN_ROWS_PER_FOLD_TEST = 5  # below this, a fold's score is too noisy to trust


def setup_logger(log_file, log_level=logging.DEBUG):
    """Configure a logger with file and console output."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('predict_energy_w5')
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


def validate_file(filename, logger):
    """Validate that the CSV has the general preprocessing columns.

    Args:
        filename (str): path to the input CSV
        logger (logging.Logger): logger instance

    Raises:
        ValueError: if the file is missing required columns
    """
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

    missing_required = [c for c in REQUIRED_COLUMNS if c not in header_cols]
    if missing_required:
        logger.error(f"Missing required columns: {missing_required}")
        raise KeyError(f"Missing required columns: {missing_required}")


def load_and_aggregate_daily(file_path, logger):
    """Load the dataset and collapse to one row per date.

    The general-processed dataset is track-level (one row per charted
    track per day), but temperature/sunshine are day-level values —
    predicting a single day's energy needs a single energy value per
    day, so charted tracks are averaged within each date.

    Args:
        file_path (str): path to the general-processed CSV
        logger (logging.Logger): logger instance

    Raises:
        ValueError: if the CSV is empty/unparseable
        KeyError: if required columns are missing

    Returns:
        pd.DataFrame: one row per date, sorted chronologically, with
            columns date, energy, temperature_mean_c, sunshine_hours
    """
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file is empty: {file_path}")
        raise ValueError(f"CSV file is empty: {file_path}")
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {file_path} ({e})")
        raise ValueError(f"Error parsing CSV file: {file_path}")

    df['date'] = pd.to_datetime(df['date'])

    logger.debug("Aggregating to one row per date "
                 "(mean energy across charted tracks; weather is "
                 "already constant within a day)")
    daily = df.groupby('date', as_index=False).agg({
        TARGET: 'mean',
        'temperature_mean_c': 'first',
        'sunshine_hours': 'first',
    })
    daily = daily.sort_values('date').reset_index(drop=True)
    daily = daily.dropna(subset=REQUIRED_COLUMNS)

    logger.debug(f"Daily dataset: {len(daily)} days, "
                 f"{daily['date'].min()} to {daily['date'].max()}")
    return daily


def add_persistence_feature(daily):
    """Add a lag-1 energy column for the persistence baseline.

    The first row has no prior day and is dropped, so all models/
    baselines are compared on the exact same rows.

    Args:
        daily (pd.DataFrame): one row per date, sorted chronologically

    Returns:
        pd.DataFrame: daily with an added 'energy_lag1' column,
            first row dropped
    """
    daily = daily.copy()
    daily['energy_lag1'] = daily[TARGET].shift(1)
    return daily.dropna(subset=['energy_lag1']).reset_index(drop=True)


def evaluate_fold(y_train, y_test, pred_mean, pred_persistence,
                  pred_linear, pred_gbr, fold_idx):
    """Compute R2/RMSE/MAE for every model+baseline on one fold.

    Args:
        y_train (pd.Series): training target
        y_test (pd.Series): true values for this fold's test rows
        pred_mean (np.ndarray): mean-baseline predictions
        pred_persistence (np.ndarray): persistence-baseline predictions
        pred_linear (np.ndarray): linear regression predictions
        pred_gbr (np.ndarray): gradient boosting predictions
        fold_idx (int): which CV fold this is

    Returns:
        list[dict]: one row per model for this fold
    """
    models = {
        'baseline_mean': pred_mean,
        'baseline_persistence': pred_persistence,
        'linear_regression': pred_linear,
        'gradient_boosting': pred_gbr,
    }
    rows = []
    for name, preds in models.items():
        rows.append({
            'fold': fold_idx,
            'model': name,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'r2': r2_score(y_test, preds),
            'rmse': float(np.sqrt(mean_squared_error(y_test, preds))),
            'mae': mean_absolute_error(y_test, preds),
        })
    return rows


def run_cv(daily, n_splits, gbr_params, logger):
    """Run TimeSeriesSplit CV with all models on identical folds.

    Args:
        daily (pd.DataFrame): output of add_persistence_feature
        n_splits (int): number of TimeSeriesSplit folds
        gbr_params (dict): hyperparameters for GradientBoostingRegressor
        logger (logging.Logger): logger instance

    Raises:
        ValueError: if there isn't enough data for the requested folds

    Returns:
        pd.DataFrame: per-fold, per-model metrics
    """
    min_required = n_splits * MIN_ROWS_PER_FOLD_TEST + MIN_ROWS_PER_FOLD_TEST
    if len(daily) < min_required:
        logger.error(f"Only {len(daily)} days available; need at least "
                     f"{min_required} for {n_splits} folds of "
                     f">= {MIN_ROWS_PER_FOLD_TEST} test rows each")
        raise ValueError(
            f"Insufficient data for {n_splits}-fold time series CV: "
            f"{len(daily)} days available, need at least {min_required}. "
            f"Reduce --n_splits or provide more data.")

    X = daily[PREDICTORS].values
    y = daily[TARGET].values
    y_lag1 = daily['energy_lag1'].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        if len(test_idx) < MIN_ROWS_PER_FOLD_TEST:
            logger.warning(f"Fold {fold_idx} has only {len(test_idx)} "
                           f"test rows (< {MIN_ROWS_PER_FOLD_TEST}) — "
                           "included, but treat this fold's score with caution")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pred_mean = np.full(len(test_idx), y_train.mean())
        pred_persistence = y_lag1[test_idx]

        linear = LinearRegression().fit(X_train, y_train)
        pred_linear = linear.predict(X_test)

        gbr = GradientBoostingRegressor(
            n_estimators=gbr_params.get('n_estimators', 100),
            max_depth=gbr_params.get('max_depth', 3),
            learning_rate=gbr_params.get('learning_rate', 0.05),
            random_state=42
        )
        gbr.fit(X_train, y_train)
        pred_gbr = gbr.predict(X_test)

        fold_rows = evaluate_fold(y_train, y_test, pred_mean,
                                  pred_persistence, pred_linear,
                                  pred_gbr, fold_idx)
        all_rows.extend(fold_rows)
        logger.debug(f"Fold {fold_idx}: n_train={len(train_idx)}, "
                     f"n_test={len(test_idx)}")

    return pd.DataFrame(all_rows)


def summarize_cv(fold_results):
    """Aggregate per-fold results into mean +/- std per model.

    Args:
        fold_results (pd.DataFrame): output of run_cv

    Returns:
        dict: model summaries, best_model, and beats_baselines verdict
    """
    summary = {}
    for model, group in fold_results.groupby('model'):
        summary[model] = {
            'r2_mean': round(float(group['r2'].mean()), 4),
            'r2_std': round(float(group['r2'].std()), 4),
            'rmse_mean': round(float(group['rmse'].mean()), 5),
            'rmse_std': round(float(group['rmse'].std()), 5),
            'mae_mean': round(float(group['mae'].mean()), 5),
            'mae_std': round(float(group['mae'].std()), 5),
            'n_folds': int(group['fold'].nunique()),
        }

    candidate_models = [m for m in summary if not m.startswith('baseline_')]
    best_model = max(candidate_models, key=lambda m: summary[m]['r2_mean'])
    best_r2 = summary[best_model]['r2_mean']
    baseline_r2s = [summary[m]['r2_mean'] for m in summary if m.startswith('baseline_')]
    beats_baselines = all(best_r2 > b for b in baseline_r2s)

    return {
        'models': summary,
        'best_model': best_model,
        'best_model_r2_mean': best_r2,
        'beats_all_baselines': beats_baselines,
        'predictors': PREDICTORS,
        'target': TARGET,
        'n_days_used': None,  # filled in by caller
    }


def process(file_path, n_splits, gbr_params, logger):
    """Full RQ5 pipeline: load, aggregate, add persistence, run CV, summarize.

    Args:
        file_path (str): path to the general-processed CSV
        n_splits (int): number of TimeSeriesSplit folds
        gbr_params (dict): hyperparameters for GradientBoostingRegressor
        logger (logging.Logger): logger instance

    Returns:
        pd.DataFrame, dict: per-fold results, summary dict
    """
    validate_file(file_path, logger)
    daily = load_and_aggregate_daily(file_path, logger)
    daily = add_persistence_feature(daily)

    logger.info(f"Running {n_splits}-fold TimeSeriesSplit CV "
                f"on {len(daily)} days")
    fold_results = run_cv(daily, n_splits, gbr_params, logger)
    summary = summarize_cv(fold_results)
    summary['n_days_used'] = len(daily)
    return fold_results, summary


def main(csvfile, n_splits, out_folds, out_summary,
         gbr_params, log_file, log_level):
    logger = setup_logger(log_file, log_level)
    logger.info('Started RQ5 predictive accuracy analysis')

    # Create output directories
    for out_path in [out_folds, out_summary]:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    try:
        fold_results, summary = process(csvfile, n_splits, gbr_params, logger)

        fold_results.to_csv(out_folds, index=False)
        logger.info(f"Saved per-fold results to {out_folds}")

        with open(out_summary, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {out_summary}")

        # Print summary to console
        print(f"Days used: {summary['n_days_used']} "
              f"({n_splits}-fold TimeSeriesSplit CV)")
        for model, stats in summary['models'].items():
            print(f"  {model:22s} R2 = {stats['r2_mean']:+.4f} "
                  f"(+/-{stats['r2_std']:.4f})   "
                  f"RMSE = {stats['rmse_mean']:.5f}")
        print(f"\nBest model: {summary['best_model']} "
              f"(R2 = {summary['best_model_r2_mean']:.4f})")
        print(f"Beats both baselines: {summary['beats_all_baselines']}")
    except Exception as e:
        logger.error(f'Failed RQ5 predictive analysis: {e}')
        raise
    logger.info('Finished RQ5 predictive accuracy analysis')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Answers RQ5: how accurately can temperature + '
                    'sunshine predict daily Energy, via TimeSeriesSplit '
                    'CV against mean/persistence baselines and a '
                    'linear vs. gradient boosting comparison')
    parser.add_argument('csvfile', type=str,
                        help='Path to the general-processed dataset')
    parser.add_argument('-n', '--n_splits', type=int, default=5,
                        help='Number of TimeSeriesSplit CV folds (default: 5)')
    parser.add_argument('--out_folds', type=str, required=True,
                        help='Output CSV: per-fold, per-model metrics')
    parser.add_argument('--out_summary', type=str, required=True,
                        help='Output JSON: aggregated summary and verdict')
    parser.add_argument('--gbr-estimators', type=int, default=100,
                        help='GradientBoosting n_estimators (default: 100)')
    parser.add_argument('--gbr-depth', type=int, default=3,
                        help='GradientBoosting max_depth (default: 3)')
    parser.add_argument('--gbr-lr', type=float, default=0.05,
                        help='GradientBoosting learning_rate (default: 0.05)')
    parser.add_argument('--log', type=str,
                        default=f"{_HERE}/logs/predict_energy_rq5.log",
                        help="Path to log file")
    parser.add_argument('--log-level', type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='DEBUG', help="Logging level (default: DEBUG)")

    args = parser.parse_args()

    # Validate arguments
    if args.n_splits < 2:
        parser.error("--n_splits must be at least 2")
    if args.gbr_estimators < 1:
        parser.error("--gbr-estimators must be at least 1")
    if args.gbr_depth < 1:
        parser.error("--gbr-depth must be at least 1")
    if not (0 < args.gbr_lr <= 1):
        parser.error("--gbr-lr must be between 0 and 1")

    gbr_params = {
        'n_estimators': args.gbr_estimators,
        'max_depth': args.gbr_depth,
        'learning_rate': args.gbr_lr,
    }

    main(args.csvfile, args.n_splits, args.out_folds, args.out_summary,
         gbr_params, args.log, args.log_level)
