#!/usr/bin/env python3
"""
Generates a bar chart and scatter plot with a linear trend line and a Markdown summary
from the Spearman correlation result JSON and the original dataset CSV.

Configurable: log file, log level, output paths, and alpha threshold for
significance reporting.
"""

import os
import sys
import argparse
import logging
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from common.utils import set_plot_params

_HERE = os.path.dirname(os.path.abspath(__file__))

REQUIRED_KEYS = ["x_variable", "y_variable", "rho", "p_value", "n"]
DEFAULT_BUCKET_ORDER = ["No rain", "Low", "Moderate", "High"]


def setup_logger(log_file, log_level=logging.DEBUG):
    """Configure a logger with file and console output."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('spearman_report')
    logger.setLevel(log_level)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(ch)

    return logger


def load_result(json_path, logger):
    """Load and validate the Spearman result JSON."""
    try:
        with open(json_path, 'r') as f:
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
            "Was this produced by spearman_correlation.py (JSON output)?"
        )
    return result


def load_points(csv_path, x_var, y_var, logger):
    """Load the two relevant columns from the source CSV.

    Raises:
        KeyError: if columns are missing
        ValueError: if CSV is empty or unreadable

    Returns:
        pd.Series, pd.Series: cleaned x and y series"""
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file is empty: {csv_path}")
        raise ValueError(f"CSV file is empty: {csv_path}")
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {csv_path} ({e})")
        raise ValueError(f"Error parsing CSV file: {csv_path}")

    missing = [c for c in (x_var, y_var) if c not in df.columns]
    if missing:
        logger.error(f"Source CSV missing columns: {missing}")
        raise KeyError(f"Source CSV missing columns: {missing}. "
                       "Make sure this is the same file used for the correlation test.")

    sub = df[[x_var, y_var]].apply(pd.to_numeric, errors='coerce').dropna()
    logger.debug(f"Loaded {len(sub)} valid points from {csv_path}")
    if len(sub) < 2:
        logger.warning(f"Only {len(sub)} valid points; trend line may be unreliable.")
    return sub[x_var], sub[y_var]


def _fmt_p(p):
    """Format p‑value: show '<0.001' for very small values."""
    return "<0.001" if p < 0.001 else f"{p:.4f}"


def plot_scatter(x, y, result, output_path, logger):
    """
    Draw a scatter plot with linear trend line and statistics annotation.

    Args:
        x (pd.Series): x values
        y (pd.Series): y values
        result (dict): parsed Spearman result
        output_path (str): path to save the PNG
        logger (logging.Logger): logger instance
    """
    fig, ax = plt.subplots()

    ax.scatter(x, y, alpha=0.55, s=28, color="#2a78d6", edgecolors="none",
               label="Observation")

    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.array([x.min(), x.max()])
    ax.plot(x_line, slope * x_line + intercept, color="#e34948",
            linewidth=2, label="Linear trend")

    ax.set_xlabel(result["x_variable"])
    ax.set_ylabel(result["y_variable"])
    ax.set_title(f"{result['x_variable']} vs. {result['y_variable']}",
                 loc="left", fontsize=13, fontweight="medium")
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best", fontsize=9)

    ci = result.get("ci_95", [None, None])
    ci_text = (f"95% CI [{ci[0]}, {ci[1]}]" if ci and ci[0] is not None
               else "95% CI unavailable")
    stat_text = (f"rho = {result['rho']:.3f}   "
                 f"p = {_fmt_p(result['p_value'])}   "
                 f"n = {result['n']}   "
                 f"{ci_text}")

    fig.text(0.01, -0.02, stat_text, fontsize=9, color="#52514e")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved scatter plot to {output_path}")


def plot_bar(csv_path, group_col, y_var, result, bucket_order, output_path, logger):
    """Draw a bar chart of mean y_var across precipitation buckets, annotated with Spearman result."""
    logger.debug(f"Drawing bar plot from file: {csv_path}")

    # Load CSV and check columns
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file is empty: {csv_path}")
        raise ValueError(f"CSV file is empty: {csv_path}")
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {csv_path} ({e})")
        raise ValueError(f"Error parsing CSV file: {csv_path}")

    missing_cols = [c for c in [y_var, group_col] if c not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns in CSV for bar plot: {missing_cols}")
        raise KeyError(f"Missing columns in CSV file: {missing_cols}")

    # Compute means and counts, reindex to bucket_order
    means = df.groupby(group_col)[y_var].mean().reindex(bucket_order)
    counts = df.groupby(group_col)[y_var].count().reindex(bucket_order)

    valid_buckets = means.dropna().index.tolist()
    if not valid_buckets:
        logger.error("No valid data for any of the specified buckets.")
        raise ValueError("No data found for the given bucket order.")

    means = means[valid_buckets]
    counts = counts[valid_buckets]

    # Create bar plot
    fig, ax = plt.subplots()
    bars = ax.bar(means.index, means.values, color="#4C72B0")

    # Add sample size labels
    for bar, n in zip(bars, counts.values):
        if pd.notna(n):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"n={int(n)}", ha="center", va="bottom", fontsize=9,
                    color="#52514e")

    ax.set_xlabel("Precipitation category")
    ax.set_ylabel(f'Average {y_var}')
    ax.set_title(f'Average {y_var} by Precipitation Category',
                 loc="left", fontweight="medium")

    # Build annotation from Spearman result
    ci = result.get("ci_95", [None, None])
    ci_text = (f"95% CI [{ci[0]}, {ci[1]}]" if ci and ci[0] is not None else "95% CI unavailable")
    sig_text = "significant" if result["significant_at_0.05"] else "not significant"
    stat_line = (f"Spearman vs. {result.get('x_variable', 'weather variable')}: "
                 f"\u03C1 = {result['rho']:.3f}  ({result.get('effect_size', '')}, "
                 f"{result.get('direction', '')})   "
                 f"p = {_fmt_p(result['p_value'])} ({sig_text})   "
                 f"n = {result['n']}   {ci_text}")

    fig.text(0.01, -0.02, stat_line, fontsize=9, color="#52514e")
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved bar plot to {output_path}")


def write_summary(result, output_path, alpha, bar_output, logger):
    """Write a Markdown summary including both scatter and bar plot if bar_output is provided."""
    direction = result.get("direction", "no clear")
    strength = result.get("effect_size", "unspecified")
    significant = result.get("significant_at_0.05", result["p_value"] < alpha)
    verdict = "statistically significant" if significant else "not statistically significant"
    ci = result.get("ci_95", [None, None])
    ci_text = f"[{ci[0]}, {ci[1]}]" if ci and ci[0] is not None else "unavailable"
    n_dropped = result.get("n_dropped", 0)

    md = f"""# Spearman correlation report

**Variables:** `{result['x_variable']}` vs. `{result['y_variable']}`

| Statistic | Value |
|---|---|
| rho | {result['rho']:.4f} |
| p-value | {_fmt_p(result['p_value'])} |
| n | {result['n']}{f" ({n_dropped} rows dropped)" if n_dropped else ""} |
| 95% CI | {ci_text} |
| Effect size | {strength} |
| Direction | {direction} |
| Significant at α = {alpha} | {"Yes" if significant else "No"} |

## Interpretation

There is a **{strength}**, **{direction}** monotonic relationship between
`{result['x_variable']}` and `{result['y_variable']}`
(rho = {result['rho']:.3f}, n = {result['n']}). This result is
**{verdict}** at the {alpha} significance level (p = {_fmt_p(result['p_value'])}).

![scatter plot]({os.path.basename(output_path).replace('.md', '.png')})
"""
    if bar_output:
        md += f"\n![bar plot]({os.path.basename(bar_output)})\n"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding='utf-8') as f:
        f.write(md)
    logger.info(f"Saved summary to {output_path}")


def main(json_path, csv_path, plot_output, summary_output, alpha,
         log_file, log_level, group_col, bucket_order, bar_output):
    # Convert log level
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
    }
    log_level = level_map.get(log_level.upper(), logging.DEBUG)

    logger = setup_logger(log_file, log_level)
    logger.info("Generating Spearman correlation report")

    # Load result and data
    result = load_result(json_path, logger)
    x, y = load_points(csv_path, result["x_variable"], result["y_variable"], logger)

    # Generate scatter plot
    plot_scatter(x, y, result, plot_output, logger)

    # Generate bar
    if group_col and bar_output:
        plot_bar(csv_path, group_col, result["y_variable"], result,
                 bucket_order, bar_output, logger)

    # Write Markdown summary
    write_summary(result, summary_output, alpha, bar_output, logger)

    # Print short summary to console
    print(f"rho={result['rho']:.3f}, p={_fmt_p(result['p_value'])}, "
          f"n={result['n']}, effect_size={result.get('effect_size')}")
    print(f"Scatter plot saved to: {plot_output}")
    if bar_output:
        print(f"Bar plot saved to: {bar_output}")
    print(f"Summary saved to: {summary_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a scatter-plot report from the Spearman correlation result. "
                    "Optionally also generate a bar chart of means across precipitation categories."
    )
    parser.add_argument("-j", "--json", required=True,
                        help="Path to the result JSON from spearman_correlation.py")
    parser.add_argument("-c", "--csv", required=True,
                        help="Path to the original dataset CSV used for the test")
    parser.add_argument("-o", "--output", default="spearman_report.png",
                        help="Output path for the scatter plot PNG (default: spearman_report.png)")
    parser.add_argument("-m", "--markdown", default="spearman_report.md",
                        help="Output path for the markdown summary (default: spearman_report.md)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level to report in the summary (default: 0.05)")
    parser.add_argument("--log", type=str,
                        default=f"{_HERE}/logs/spearman_report.log",
                        help="Path to log file (default: logs/spearman_report.log)")
    parser.add_argument("--log-level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='DEBUG', help="Logging level (default: DEBUG)")

    # New arguments for bar plot
    parser.add_argument("--group", type=str, default=None,
                        help="Precipitation category column (e.g., precipitation_mm_cat) – required for bar plot")
    parser.add_argument("--bucket-order", type=str,
                        default=",".join(DEFAULT_BUCKET_ORDER),
                        help=f"Comma‑separated bucket order (default: {','.join(DEFAULT_BUCKET_ORDER)})")
    parser.add_argument("--bar-output", type=str, default=None,
                        help="Output path for the bar plot PNG (if --group is given)")

    args = parser.parse_args()

    # Validate arguments for bar plot
    if args.group and not args.bar_output:
        parser.error("--bar-output is required when --group is provided")
    if args.bar_output and not args.group:
        parser.error("--group is required when --bar-output is provided")

    # Parse bucket order
    bucket_order = [b.strip() for b in args.bucket_order.split(',') if b.strip()]
    if not bucket_order:
        bucket_order = DEFAULT_BUCKET_ORDER

    # Set plot parameters from shared config
    config_path = os.path.join(_HERE, "..", "common", "configuration_plots.yaml")
    set_plot_params(config_path)

    main(args.json, args.csv, args.output, args.markdown,
         args.alpha, args.log, args.log_level,
         args.group, bucket_order, args.bar_output)
