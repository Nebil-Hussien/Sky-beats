#!/usr/bin/env python3
"""
Generates a scatter plot and Markdown report for the absolute deviation
magnitude regression (RQ4). The report includes OLS regression statistics
and an interpretation.

Configurable: log file, log level, significance threshold.
"""

import os
import sys
import argparse
import logging
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from common.utils import set_plot_params

_HERE = os.path.dirname(os.path.abspath(__file__))

REQUIRED_KEYS = ["y_variable", "slope", "slope_p_value", "r_squared",
                 "n", "effect_label", "intercept", "significant_at_0.05"]


def setup_logger(log_file, log_level=logging.DEBUG):
    """Configure a logger with file and console output."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('magnitude_regression_report')
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


def load_result(json_path, logger):
    """
    Load and validate the regression result JSON.

    Args:
        json_path (str): Path to the JSON file.
        logger (logging.Logger): Logger instance.

    Raises:
        FileNotFoundError: If the JSON file is missing.
        ValueError: If the JSON is invalid or missing required keys.

    Returns:
        dict: Parsed result.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
    except FileNotFoundError:
        logger.error(f"Result JSON file not found: {json_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {json_path}: {e}")
        raise ValueError(f"Invalid JSON format in {json_path}")

    missing = [k for k in REQUIRED_KEYS if k not in result]
    if missing:
        logger.error(f"Result JSON missing expected keys: {missing}")
        raise ValueError(
            f"Result JSON missing expected keys: {missing}. "
            "Was this produced by the regression script (magnitude_regression.py)?"
        )
    return result


def load_points(csv_path, y_var, logger):
    """
    Load the CSV and compute absolute deviation, returning x and y series.

    Args:
        csv_path (str): Path to the CSV file.
        y_var (str): Name of the audio feature column.
        logger (logging.Logger): Logger instance.

    Raises:
        FileNotFoundError: If CSV is missing.
        ValueError: If CSV is empty or unreadable.
        KeyError: If required columns are missing.

    Returns:
        pd.Series, pd.Series: x (absolute temperature deviation), y (audio feature).
    """
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file is empty: {csv_path}")
        raise ValueError(f"CSV file is empty: {csv_path}")
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {csv_path} ({e})")
        raise ValueError(f"Error parsing CSV file: {csv_path}")

    required_cols = ['temperature_deviation', y_var]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"CSV missing required columns: {missing_cols}")
        raise KeyError(f"CSV missing required columns: {missing_cols}. "
                       "Make sure this is the RQ4-processed dataset.")

    sub = df[['temperature_deviation', y_var]].dropna()
    if len(sub) == 0:
        logger.error("No valid rows after dropping NaNs.")
        raise ValueError("No valid data for regression plotting.")

    x = sub['temperature_deviation'].abs()
    y = sub[y_var]
    logger.debug(f"Loaded {len(x)} valid points for {y_var}")
    return x, y


def _fmt_p(p):
    """Format p‑value: show '<0.001' for very small values."""
    return "<0.001" if p < 0.001 else f"{p:.4f}"


def plot_scatter(x, y, result, output_path, logger):
    """
    Draw a scatter plot with OLS regression line and statistics annotation.

    Args:
        x (pd.Series): Absolute temperature deviation.
        y (pd.Series): Audio feature values.
        result (dict): Parsed regression result.
        output_path (str): Path to save the PNG.
        logger (logging.Logger): Logger instance.
    """
    fig, ax = plt.subplots()

    ax.scatter(x, y, alpha=0.4, s=15, color="#2a78d6", edgecolors="none",
               label="Observation")

    # OLS trend line (using the slope and intercept from the regression)
    # We use the stored coefficients for consistency with the test.
    slope = result['slope']
    intercept = result['intercept']
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color="#e34948", linewidth=2, label="OLS trend")

    ax.set_xlabel("Absolute temperature deviation (°C)")
    ax.set_ylabel(result["y_variable"])
    ax.set_title(f"{result['y_variable']} vs. |temperature deviation|",
                 loc="left", fontsize=13, fontweight="medium")
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best", fontsize=9)

    # Stats annotation
    stat_text = (f"slope = {slope:.4f}   "
                 f"p = {_fmt_p(result['slope_p_value'])}   "
                 f"R² = {result['r_squared']:.3f}   "
                 f"n = {result['n']}   "
                 f"effect = {result['effect_label']}")
    fig.text(0.01, -0.02, stat_text, fontsize=9, color="#52514e")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot saved to {output_path}")


def write_summary(result, output_path, alpha, logger):
    """
    Write a Markdown summary of the regression result.

    Args:
        result (dict): Parsed regression result.
        output_path (str): Path to save the Markdown file.
        alpha (float): Significance threshold.
        logger (logging.Logger): Logger instance.
    """
    sig = result['significant_at_0.05']
    verdict = "statistically significant" if sig else "not statistically significant"
    direction = "positive" if result['slope'] > 0 else "negative"

    md = f"""# Temperature deviation magnitude regression report

**Variable:** `{result['y_variable']}` vs. absolute temperature deviation

| Statistic | Value |
|---|---|
| Slope | {result['slope']:.4f} |
| Intercept | {result['intercept']:.4f} |
| p-value (slope) | {_fmt_p(result['slope_p_value'])} |
| R² | {result['r_squared']:.4f} |
| n | {result['n']} |
| Effect size | {result['effect_label']} |
| Direction | {direction} |
| Significant at α = {alpha} | {"Yes" if sig else "No"} |

## Interpretation

A 1°C increase in absolute temperature deviation from the monthly norm is associated with a **{direction}** change of **{result['slope']:.4f}** in `{result['y_variable']}` (p = {_fmt_p(result['slope_p_value'])}).
The model explains **{result['r_squared'] * 100:.1f}%** of the variance (R²), indicating a **{result['effect_label']}** effect size.
This result is **{verdict}** at the {alpha} significance level.

![scatter plot]({os.path.basename(output_path).replace('.md', '.png')})
"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info(f"Summary saved to {output_path}")


def main(json_path, csv_path, output_png, output_md, alpha, log_file, log_level):
    # Convert log level
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
    }
    log_level = level_map.get(log_level.upper(), logging.DEBUG)

    logger = setup_logger(log_file, log_level)
    logger.info("Generating magnitude regression report")

    # Set plotting parameters from config
    config_path = os.path.join(_HERE, "..", "common", "configuration_plots.yaml")
    set_plot_params(config_path)

    # Load result and data
    result = load_result(json_path, logger)
    x, y = load_points(csv_path, result["y_variable"], logger)

    # Generate plot and summary
    plot_scatter(x, y, result, output_png, logger)
    write_summary(result, output_md, alpha, logger)

    # Print short summary to console
    print(f"slope={result['slope']:.3f}, p={_fmt_p(result['slope_p_value'])}, R²={result['r_squared']:.3f}")
    print(f"Plot: {output_png}")
    print(f"Summary: {output_md}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a scatter‑plot report from the magnitude regression result."
    )
    parser.add_argument("-j", "--json", required=True,
                        help="Path to the regression result JSON")
    parser.add_argument("-c", "--csv", required=True,
                        help="Path to the original CSV (RQ4-processed) for plotting")
    parser.add_argument("-o", "--output_png", default="magnitude_regression_plot.png",
                        help="Output path for the plot PNG")
    parser.add_argument("-m", "--markdown", default="magnitude_regression_report.md",
                        help="Output path for the markdown summary")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level to report")
    parser.add_argument("--log", type=str, default=f"{_HERE}/logs/deviation_magnitude_regression_report_w4.log",
                        help="Path to log file")
    parser.add_argument("--log-level", type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='DEBUG', help="Logging level (default: DEBUG)")

    args = parser.parse_args()

    # Validate alpha
    if not (0 < args.alpha < 1):
        parser.error("--alpha must be between 0 and 1")

    main(args.json, args.csv, args.output_png, args.markdown,
         args.alpha, args.log, args.log_level)
