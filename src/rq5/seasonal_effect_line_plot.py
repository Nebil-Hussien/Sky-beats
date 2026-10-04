#!/usr/bin/env python3
"""
The script produces a plot of predicted Energy vs. one weather
predictor, with one line per season (holding the other predictors
at their mean).

It refits the interaction model (Model 2) internally and generates
predictions across the range of the focus predictor.

Configurable: log file, log level, season order.
"""

import os
import sys
import argparse
import logging
import yaml
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

_HERE = os.path.dirname(os.path.abspath(__file__))

# Default season order (meteorological seasons, Winter first for reference)
DEFAULT_SEASON_ORDER = ['Winter', 'Spring', 'Summer', 'Autumn']


def setup_logger(log_file, log_level=logging.DEBUG):
    """Configure a logger with file and console output."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('seasonal_effect_plot')
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


def load_config(config_file, logger):
    """
    Load configuration parameters from a YAML file.

    Args:
        config_file (str): path of the configuration file
        logger (logging.Logger): logger instance

    Raises:
        ValueError: if config_file is not a .yaml file or cannot be read

    Returns:
        dict: configuration parameters
    """
    if not config_file.endswith(('.yaml', '.yml')):
        logger.error(f"Config file must be .yaml or .yml: {config_file}")
        raise ValueError("Please provide a .yaml or .yml file")

    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_file}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {config_file} ({e})")
        raise ValueError(f"Invalid YAML in {config_file}")

    return config


def set_plot_params(config_file, logger):
    """
    Set matplotlib parameters according to the configuration file.

    Args:
        config_file (str): path of the configuration file
        logger (logging.Logger): logger instance

    Returns:
        None
    """
    plot_params = load_config(config_file, logger)
    for param, value in plot_params.items():
        if param == 'style':
            plt.style.use(value)
        else:
            plt.rcParams[param] = value
    logger.debug(f"Applied plot parameters from {config_file}")


def draw_seasonal_effect_plot(csv_file_path, predictors, focus_predictor,
                              output, season_order, logger):
    """
    Refit Model 2 (with seasonal interactions) on the data, then
    plot predicted Energy across the range of focus_predictor,
    one line per season, holding all other predictors at their mean.

    Args:
        csv_file_path (str): path to the processed dataset
        predictors (list): the retained predictor column names
        focus_predictor (str): which predictor to vary on the x-axis
        output (str): output filename for the plot
        season_order (list): ordered list of season labels
        logger (logging.Logger): logger instance

    Raises:
        ValueError: if CSV is empty or unreadable, or insufficient data
        KeyError: if required columns are missing
    """
    logger.debug(f"Loading data from {csv_file_path}")

    # Load CSV
    try:
        df = pd.read_csv(csv_file_path)
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file is empty: {csv_file_path}")
        raise ValueError(f"CSV file is empty: {csv_file_path}")
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {csv_file_path} ({e})")
        raise ValueError(f"Error parsing CSV file: {csv_file_path}")

    # Check required columns
    required_cols = predictors + [focus_predictor, 'season', 'energy']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"Missing columns in CSV file: {missing}")
        raise KeyError(f"Missing columns in CSV file: {missing}")

    # Check that focus_predictor is in predictors
    if focus_predictor not in predictors:
        logger.error(f"Focus predictor '{focus_predictor}' not in predictors: {predictors}")
        raise ValueError(f"Focus predictor '{focus_predictor}' must be in predictors list")

    # Drop rows with missing values in required columns
    before = len(df)
    df = df.dropna(subset=required_cols)
    dropped = before - len(df)
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with missing values")

    if len(df) == 0:
        logger.error("No data remaining after cleaning")
        raise ValueError("No valid data for plotting.")

    # Check that we have data for each season
    available_seasons = df['season'].unique()
    missing_seasons = [s for s in season_order if s not in available_seasons]
    if missing_seasons:
        logger.warning(f"Missing seasons in data: {missing_seasons}")

    # Refit Model 2 with seasonal interactions
    logger.debug("Refitting Model 2 (with seasonal interactions)")
    main_terms = " + ".join(predictors)
    interaction_terms = " + ".join(
        f"{p}:C(season, Treatment(reference='{season_order[0]}'))" for p in predictors)
    formula = (f"energy ~ {main_terms} "
               f"+ C(season, Treatment(reference='{season_order[0]}')) + {interaction_terms}")

    try:
        model = smf.ols(formula, data=df).fit()
    except Exception as e:
        logger.error(f"Model fitting failed: {e}")
        raise ValueError(f"Could not fit model: {e}")

    logger.debug(f"Model R²: {model.rsquared:.4f}")

    # Build prediction grid
    logger.debug(f"Building prediction grid varying {focus_predictor}")
    x_range = np.linspace(df[focus_predictor].min(), df[focus_predictor].max(), 50)

    other_predictors = [p for p in predictors if p != focus_predictor]
    other_means = {p: df[p].mean() for p in other_predictors}

    # Create the plot
    fig, ax = plt.subplots(figsize=(9, 6))

    # Filter to seasons that actually exist in the data
    valid_seasons = [s for s in season_order if s in available_seasons]
    if not valid_seasons:
        logger.error("No valid seasons found in data")
        raise ValueError("No valid seasons found in data")

    for season in valid_seasons:
        grid = pd.DataFrame({focus_predictor: x_range, 'season': season})
        for p, mean_val in other_means.items():
            grid[p] = mean_val
        grid['season'] = pd.Categorical(grid['season'], categories=season_order)
        predicted = model.predict(grid)
        ax.plot(x_range, predicted, label=season)

    ax.set_xlabel(focus_predictor)
    ax.set_ylabel('Predicted Energy')
    ax.set_title(f'Predicted Energy vs {focus_predictor}, by Season')
    ax.legend(title='Season')
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()

    # Save the plot
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot to {output}")


def main(csvfile, predictors, focus_predictor, output, season_order,
         log_file, log_level):
    logger = setup_logger(log_file, log_level)
    logger.info('Drawing seasonal effect plot')

    # Set plot parameters from config
    config_path = os.path.join(_HERE, "..", "common", "configuration_plots.yaml")
    try:
        set_plot_params(config_path, logger)
    except Exception as e:
        logger.warning(f"Could not load plot config: {e}. Using defaults.")

    draw_seasonal_effect_plot(csvfile, predictors, focus_predictor,
                              output, season_order, logger)

    logger.info('Plot finished')
    print(f"Plot saved to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plots predicted Energy vs one weather predictor, '
                    'one line per season'
    )
    parser.add_argument('csvfile', type=str,
                        help='Path of the RQ-Energy processed dataset')
    parser.add_argument('-p', '--predictors', type=str, required=True,
                        help="Comma-separated retained predictor columns, "
                             "e.g. 'temperature_mean_c,sunshine_hours'")
    parser.add_argument('-f', '--focus_predictor', type=str, required=True,
                        help='Which predictor to vary on the x-axis')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='The output name for the plot')
    parser.add_argument('--season-order', type=str,
                        default=",".join(DEFAULT_SEASON_ORDER),
                        help=f"Comma-separated season order (default: {','.join(DEFAULT_SEASON_ORDER)})")
    parser.add_argument('--log', type=str,
                        default=None, help='Path to log file')
    parser.add_argument('--log-level', type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='DEBUG', help='Logging level (default: DEBUG)')

    args = parser.parse_args()

    # Parse predictors list
    predictors_list = [p.strip() for p in args.predictors.split(',') if p.strip()]
    if not predictors_list:
        parser.error("--predictors must contain at least one predictor")

    # Parse season order
    season_order = [s.strip() for s in args.season_order.split(',') if s.strip()]
    if not season_order:
        season_order = DEFAULT_SEASON_ORDER

    # Validate focus_predictor is in predictors
    if args.focus_predictor not in predictors_list:
        parser.error(f"Focus predictor '{args.focus_predictor}' not in predictors list")

    # Set log default if not provided
    if args.log is None:
        args.log = f"{_HERE}/logs/seasonal_effect_{args.focus_predictor}.log"

    main(args.csvfile, predictors_list, args.focus_predictor, args.output,
         season_order, args.log, args.log_level)
