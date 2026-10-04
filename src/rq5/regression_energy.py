#!/usr/bin/env python3
"""
This script runs the regression analysis for RQ-Energy.

It builds two nested OLS models:
  Model 1 : energy ~ temp + precip + sunshine + season
  Model 2 : Model 1 + (temp:season) + (precip:season) + (sunshine:season)

and compares them with an F-test to determine whether the predictors'
effect on energy genuinely varies by season. Where Model 2 is interactive
with the season, it extracts the effective slope of each predictor within
each season, and saves it to a CSV.

Before fitting, it checks for multicollinearity among the three
weather predictors via Variance Inflation Factor. If any
predictor's VIF exceeds the threshold, it is iteratively dropped
(highest VIF first) and VIF is recomputed, until all remaining
predictors are below threshold.

Configurable: VIF threshold, test size, random seed, significance level,
log file, log level.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor

_HERE = os.path.dirname(os.path.abspath(__file__))

# Default predictors and target
CANDIDATE_PREDICTORS = ['temperature_mean_c', 'precipitation_mm', 'sunshine_hours']
TARGET = 'energy'

# Defaults
DEFAULT_VIF_THRESHOLD = 5.0
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_SEED = 42
DEFAULT_ALPHA = 0.05

# Columns added by general preprocessing
GENERAL_PREPROCESSING_COLUMNS = {'year', 'month', 'day', 'weekday',
                                 'is_weekend', 'season'}

# Required columns for this script
REQUIRED_COLUMNS = ['date', 'energy', 'season'] + CANDIDATE_PREDICTORS


def setup_logger(log_file, log_level=logging.DEBUG):
    """Configure a logger with file and console output."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('regression_energy')
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
    """
    Validate that the CSV has the general preprocessing columns and required columns.

    Args:
        filename (str): path to the input CSV
        logger (logging.Logger): logger instance

    Raises:
        ValueError: if file is missing required columns
        KeyError: if required columns are missing
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


def load_data(file_path, logger):
    """
    Load the CSV and return the dataframe.

    Args:
        file_path (str): path to the CSV
        logger (logging.Logger): logger instance

    Raises:
        ValueError: if file is empty or unreadable

    Returns:
        pd.DataFrame: loaded data
    """
    validate_file(file_path, logger)

    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file is empty: {file_path}")
        raise ValueError(f"CSV file is empty: {file_path}")
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {file_path} ({e})")
        raise ValueError(f"Error parsing CSV file: {file_path}")

    logger.debug(f"Loaded data: {df.shape}")

    # Drop rows with missing values in essential columns
    essential = REQUIRED_COLUMNS
    before = len(df)
    df = df.dropna(subset=essential)
    dropped = before - len(df)
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with missing essential values")

    if len(df) == 0:
        logger.error("No data remaining after cleaning")
        raise ValueError("No valid data for regression.")

    return df


def compute_vif(df, predictors):
    """
    Compute the Variance Inflation Factor for each predictor.

    Args:
        df (pd.DataFrame): dataframe containing the predictor columns
        predictors (list): names of predictor columns to check

    Returns:
        pd.DataFrame: columns ['predictor', 'vif']
    """
    X = df[predictors].assign(const=1.0)
    vifs = []
    for i, col in enumerate(predictors):
        vif = variance_inflation_factor(X.values, i)
        vifs.append({'predictor': col, 'vif': round(vif, 3)})
    return pd.DataFrame(vifs).sort_values('vif', ascending=False)


def drop_high_vif_iteratively(df, predictors, threshold, logger):
    """
    Iteratively drop the predictor with the highest VIF until all
    remaining predictors are below the threshold.

    Args:
        df (pd.DataFrame): dataframe containing the predictor columns
        predictors (list): candidate predictor column names
        threshold (float): VIF cutoff above which a predictor is dropped
        logger (logging.Logger): logger instance

    Returns:
        tuple: (retained_predictors (list), vif_history (pd.DataFrame),
                dropped_log (list of str))
    """
    retained = list(predictors)
    vif_history = []
    dropped_log = []

    while len(retained) > 1:
        vif_table = compute_vif(df, retained)
        max_vif_row = vif_table.iloc[0]
        if max_vif_row['vif'] <= threshold:
            vif_history.append(vif_table.assign(
                step=len(dropped_log), retained_at_step=','.join(retained)))
            break
        vif_history.append(vif_table.assign(
            step=len(dropped_log), retained_at_step=','.join(retained)))
        dropped = max_vif_row['predictor']
        msg = (f"Dropping '{dropped}' (VIF={max_vif_row['vif']}) "
               f"exceeds threshold {threshold}")
        logger.warning(msg)
        dropped_log.append(msg)
        retained.remove(dropped)

    if not vif_history:
        vif_history.append(compute_vif(df, retained).assign(
            step=0, retained_at_step=','.join(retained)))

    return retained, pd.concat(vif_history, ignore_index=True), dropped_log


def fit_nested_models(df, retained_predictors, logger):
    """
    Fit the no-interaction and with-interaction OLS models.

    Args:
        df (pd.DataFrame): data containing 'energy', 'season', and
            the retained predictor columns
        retained_predictors (list): predictor column names that
            survived the VIF check
        logger (logging.Logger): logger instance

    Returns:
        tuple: (model1_results, model2_results, formula1, formula2)
    """
    main_terms = " + ".join(retained_predictors)
    formula1 = f"energy ~ {main_terms} + C(season, Treatment(reference='Winter'))"

    interaction_terms = " + ".join(
        f"{p}:C(season, Treatment(reference='Winter'))" for p in retained_predictors)
    formula2 = f"{formula1} + {interaction_terms}"

    logger.info(f"Model 1 formula: {formula1}")
    logger.info(f"Model 2 formula: {formula2}")

    model1 = smf.ols(formula1, data=df).fit()
    model2 = smf.ols(formula2, data=df).fit()
    return model1, model2, formula1, formula2


def nested_model_comparison(model1, model2, logger):
    """
    Compare the no-interaction vs with-interaction model via an F-test.

    Args:
        model1: fitted statsmodels results, no interactions (nested)
        model2: fitted statsmodels results, with interactions
        logger (logging.Logger): logger instance

    Returns:
        pd.DataFrame: ANOVA comparison table
    """
    comparison = anova_lm(model1, model2)
    return comparison


def extract_seasonal_slopes(model2, retained_predictors, logger,
                            seasons=('Winter', 'Spring', 'Summer', 'Autumn')):
    """
    Derive the effective slope of each predictor within each season.

    Args:
        model2: fitted statsmodels results, with interactions
        retained_predictors (list): predictor column names
        logger (logging.Logger): logger instance
        seasons (tuple): season labels, reference season first

    Returns:
        pd.DataFrame: columns ['predictor', 'season', 'effective_slope']
    """
    params = model2.params
    reference = seasons[0]
    rows = []
    for predictor in retained_predictors:
        base_slope = params.get(predictor, np.nan)
        rows.append({'predictor': predictor, 'season': reference,
                    'effective_slope': round(base_slope, 5)})
        for season in seasons[1:]:
            interaction_key = (
                f"{predictor}:C(season, Treatment(reference='{reference}'))"
                f"[T.{season}]")
            interaction_term = params.get(interaction_key, 0.0)
            rows.append({
                'predictor': predictor, 'season': season,
                'effective_slope': round(base_slope + interaction_term, 5),
            })
    return pd.DataFrame(rows)


def evaluate_predictive_model(df, predictors, target='energy', test_size=0.2,
                              random_state=42, logger=None):
    """
    Train a simple linear regression and report predictive metrics.

    Args:
        df (pd.DataFrame): input data
        predictors (list): predictor column names
        target (str): target column name
        test_size (float): proportion of data for testing
        random_state (int): random seed
        logger (logging.Logger, optional): logger instance

    Raises:
        KeyError: if required columns are missing

    Returns:
        pd.DataFrame: predictive metrics
    """
    required = predictors + [target]
    missing = [col for col in required if col not in df.columns]
    if missing:
        if logger:
            logger.error(f"Missing columns: {missing}")
        raise KeyError(f"Missing columns: {missing}")

    X = df[predictors]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = pd.DataFrame([{
        'model': 'linear_regression',
        'r2': round(r2_score(y_test, preds), 5),
        'rmse': round(float(np.sqrt(mean_squared_error(y_test, preds))), 5),
        'mae': round(float(mean_absolute_error(y_test, preds)), 5),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'predictors': ','.join(predictors),
        'target': target,
    }])
    return metrics


def main(csvfile, out_coefficients, out_vif, out_comparison,
         out_seasonal_slopes, out_significance, out_predictive_metrics,
         vif_threshold, test_size, random_seed, alpha,
         log_file, log_level):
    logger = setup_logger(log_file, log_level)
    logger.info('Started RQ-Energy regression analysis')

    # Create output directories
    output_paths = [out_coefficients, out_vif, out_comparison,
                    out_seasonal_slopes, out_significance,
                    out_predictive_metrics]
    for path in output_paths:
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    try:
        # Load data
        df = load_data(csvfile, logger)

        # Check multicollinearity
        logger.info('Checking multicollinearity (VIF)')
        retained_predictors, vif_history, dropped_log = drop_high_vif_iteratively(
            df, CANDIDATE_PREDICTORS, threshold=vif_threshold, logger=logger)
        vif_history.to_csv(out_vif, index=False)
        logger.info(f"Retained predictors after VIF check: {retained_predictors}")
        if dropped_log:
            logger.warning(f"Predictors dropped due to high VIF: {len(dropped_log)}")
            for line in dropped_log:
                logger.warning(f"  - {line}")

        # Fit nested models
        logger.info('Fitting nested models')
        model1, model2, formula1, formula2 = fit_nested_models(
            df, retained_predictors, logger)

        # Save model 2 coefficients
        logger.info('Saving model 2 (full) coefficients')
        coef_table = model2.summary2().tables[1].reset_index()
        coef_table.columns = ['term', 'coef', 'std_err', 't', 'p_value',
                              'ci_lower', 'ci_upper']
        coef_table.to_csv(out_coefficients, index=False)

        # Run nested F-test
        logger.info('Running nested F-test comparison (Model 1 vs Model 2)')
        comparison = nested_model_comparison(model1, model2, logger)
        comparison.to_csv(out_comparison)
        interaction_p_value = comparison['Pr(>F)'].iloc[-1]
        logger.info(f"Nested F-test p-value for seasonal interactions: {interaction_p_value:.4f}")

        # Extract seasonal slopes
        logger.info('Extracting effective seasonal slopes')
        seasonal_slopes = extract_seasonal_slopes(model2, retained_predictors, logger)
        seasonal_slopes.to_csv(out_seasonal_slopes, index=False)

        # Evaluate predictive performance
        logger.info('Evaluating predictive performance')
        predictive_predictors = ['temperature_mean_c', 'sunshine_hours']
        predictive_metrics = evaluate_predictive_model(
            df,
            predictors=[p for p in predictive_predictors if p in df.columns],
            target='energy',
            test_size=test_size,
            random_state=random_seed,
            logger=logger
        )
        predictive_metrics.to_csv(out_predictive_metrics, index=False)

        # Write significance flag
        significant = interaction_p_value < alpha
        with open(out_significance, 'w') as f:
            f.write("True" if significant else "False")

        # Log summary
        logger.info('Finished RQ-Energy regression analysis')
        logger.info(f"Retained predictors: {retained_predictors}")
        if dropped_log:
            logger.info("Dropped due to high VIF:")
            for line in dropped_log:
                logger.info(f"  - {line}")
        logger.info(f"Model 1 R-squared: {model1.rsquared:.4f}")
        logger.info(f"Model 2 R-squared: {model2.rsquared:.4f}")
        logger.info(f"Nested F-test (interactions add value?) p-value: {interaction_p_value:.4f}")
        logger.info(f"Seasonal interactions significant at α={alpha}: {significant}")
        logger.info("Effective seasonal slopes:\n" + seasonal_slopes.to_string(index=False))

        # Print summary to console
        print(f"Retained predictors: {retained_predictors}")
        print(f"Model 1 R²: {model1.rsquared:.4f}, Model 2 R²: {model2.rsquared:.4f}")
        print(f"Seasonal interactions significant at α={alpha}: {significant} (p={interaction_p_value:.4f})")

    except Exception as e:
        logger.error(f"Regression analysis failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Executes a regression analysis: VIF check, '
                    'nested model comparison, and seasonal slope extraction'
    )
    parser.add_argument('csvfile', type=str,
                        help='Path of the RQ-Energy processed dataset')
    parser.add_argument('-c', '--out_coefficients', type=str, required=True,
                        help='Output CSV for model 2 (full) coefficients')
    parser.add_argument('--out_vif', type=str, required=True,
                        help='Output CSV for VIF history')
    parser.add_argument('--out_comparison', type=str, required=True,
                        help='Output CSV for nested model F-test comparison')
    parser.add_argument('--out_seasonal_slopes', type=str, required=True,
                        help='Output CSV for effective per-season slopes')
    parser.add_argument('-o', '--out_significance', type=str, required=True,
                        help='Output TXT: True/False, seasonal interactions significant')
    parser.add_argument('--out_predictive_metrics', type=str, required=True,
                        help='Output CSV for predictive metrics')
    parser.add_argument('--vif-threshold', type=float, default=DEFAULT_VIF_THRESHOLD,
                        help=f'VIF threshold for dropping predictors (default: {DEFAULT_VIF_THRESHOLD})')
    parser.add_argument('--test-size', type=float, default=DEFAULT_TEST_SIZE,
                        help=f'Test size for predictive evaluation (default: {DEFAULT_TEST_SIZE})')
    parser.add_argument('--random-seed', type=int, default=DEFAULT_RANDOM_SEED,
                        help=f'Random seed for train/test split (default: {DEFAULT_RANDOM_SEED})')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                        help=f'Significance level (default: {DEFAULT_ALPHA})')
    parser.add_argument('--log', type=str,
                        default=f"{_HERE}/logs/regression_energy.log",
                        help='Path to log file')
    parser.add_argument('--log-level', type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='DEBUG', help='Logging level (default: DEBUG)')

    args = parser.parse_args()

    # Validate arguments
    if args.vif_threshold <= 1.0:
        parser.error("--vif-threshold must be > 1.0")
    if not (0 < args.test_size < 1):
        parser.error("--test-size must be between 0 and 1")
    if not (0 < args.alpha < 1):
        parser.error("--alpha must be between 0 and 1")

    main(args.csvfile, args.out_coefficients, args.out_vif,
         args.out_comparison, args.out_seasonal_slopes,
         args.out_significance, args.out_predictive_metrics,
         args.vif_threshold, args.test_size, args.random_seed,
         args.alpha, args.log, args.log_level)
