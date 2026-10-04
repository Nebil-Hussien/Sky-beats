#!/usr/bin/env python3
"""
Generates a report for predict_energy_w5.py's output: a bar chart
comparing mean cross-validated R2 (with std-dev error bars) across
the mean baseline, persistence baseline, linear regression, and
gradient boosting, plus a markdown summary answering RQ5 directly.

Configurable: log file, log level, alpha threshold for reporting.
"""

import os
import sys
import argparse
import logging
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))

# Default model order and styling
MODEL_ORDER = ['baseline_mean', 'baseline_persistence',
               'linear_regression', 'gradient_boosting']
MODEL_LABELS = {
    'baseline_mean': 'Mean\nbaseline',
    'baseline_persistence': 'Persistence\nbaseline',
    'linear_regression': 'Linear\nregression',
    'gradient_boosting': 'Gradient\nboosting',
}
MODEL_COLORS = {
    'baseline_mean': '#B0AFA8',
    'baseline_persistence': '#8C8B85',
    'linear_regression': '#4C72B0',
    'gradient_boosting': '#185FA5',
}

REQUIRED_KEYS = ['models', 'best_model', 'best_model_r2_mean',
                 'beats_all_baselines', 'predictors', 'target', 'n_days_used']


def setup_logger(log_file, log_level=logging.DEBUG):
    """Configure a logger with file and console output."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('predict_energy_report')
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


def load_summary(json_path, logger):
    """
    Load and validate the predict_energy_w5.py summary JSON.

    Args:
        json_path (str): path to the summary JSON
        logger (logging.Logger): logger instance

    Raises:
        FileNotFoundError: If the JSON file is missing.
        ValueError: If the JSON is invalid or missing required keys.

    Returns:
        dict: the parsed summary
    """
    try:
        with open(json_path, 'r') as f:
            summary = json.load(f)
    except FileNotFoundError:
        logger.error(f"Summary JSON file not found: {json_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {json_path}: {e}")
        raise ValueError(f"Invalid JSON format in {json_path}")

    missing = [k for k in REQUIRED_KEYS if k not in summary]
    if missing:
        logger.error(f"Summary JSON missing expected keys: {missing}")
        raise ValueError(
            f"Summary JSON missing expected keys: {missing}. "
            "Was this produced by predict_energy_w5.py?"
        )

    # Validate that models contains the expected entries
    expected_models = ['baseline_mean', 'baseline_persistence',
                       'linear_regression', 'gradient_boosting']
    missing_models = [m for m in expected_models if m not in summary['models']]
    if missing_models:
        logger.warning(f"Missing some expected models: {missing_models}")

    return summary


def plot_cv_comparison(summary, output_path, logger):
    """
    Draw the R2-by-model bar chart with std-dev error bars.

    Args:
        summary (dict): parsed summary JSON
        output_path (str): path to save the PNG
        logger (logging.Logger): logger instance
    """
    # Get models in the defined order, falling back to what's available
    models = [m for m in MODEL_ORDER if m in summary['models']]
    if not models:
        logger.error("No models found in summary")
        raise ValueError("No models found in summary")

    r2_means = [summary['models'][m]['r2_mean'] for m in models]
    r2_stds = [summary['models'][m]['r2_std'] for m in models]
    colors = [MODEL_COLORS.get(m, '#808080') for m in models]
    labels = [MODEL_LABELS.get(m, m.replace('_', ' ').title()) for m in models]

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    bars = ax.bar(labels, r2_means, yerr=r2_stds, capsize=5,
                  color=colors, edgecolor="none")

    for bar, mean_val in zip(bars, r2_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{mean_val:+.3f}", ha="center",
                va="bottom" if mean_val >= 0 else "top", fontsize=10)

    ax.axhline(0, color="#898781", linewidth=0.8)
    ax.set_ylabel("Cross-validated R\u00b2 (mean \u00b1 std across folds)")
    ax.set_title(f"Predicting {summary['target']} from "
                 f"{' + '.join(summary['predictors'])}",
                 loc="left", fontsize=13, fontweight="medium")
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    n_folds = next(iter(summary['models'].values()))['n_folds']
    fig.text(0.01, -0.03,
             f"{summary['n_days_used']} days, {n_folds}-fold TimeSeriesSplit CV. "
             f"Best model: {summary['best_model']} "
             f"(beats both baselines: {summary['beats_all_baselines']})",
             fontsize=9, color="#52514e")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot to {output_path}")


def write_summary(summary, output_path, alpha, logger):
    """
    Write a markdown summary answering RQ5 from the results.

    Args:
        summary (dict): parsed summary JSON
        output_path (str): path to save the markdown file
        alpha (float): significance threshold for reporting
        logger (logging.Logger): logger instance
    """
    rows = "\n".join(
        f"| {m} | {s['r2_mean']:+.4f} | {s['r2_std']:.4f} | "
        f"{s['rmse_mean']:.5f} | {s['mae_mean']:.5f} | {s['n_folds']} |"
        for m, s in summary['models'].items()
    )

    best = summary['best_model']
    best_stats = summary['models'][best]
    verdict = ("clearly outperforms" if summary['beats_all_baselines']
               else "does not consistently outperform")

    # Use the actual model label if available, otherwise the raw name
    best_display = MODEL_LABELS.get(best, best.replace('_', ' ').title())

    md = f"""# RQ5 report: predicting {summary['target']} from {', '.join(summary['predictors'])}

**Data:** {summary['n_days_used']} days, evaluated via TimeSeriesSplit
cross-validation (chronological folds — training only ever precedes
testing, avoiding leakage from autocorrelated daily charts).

**Significance threshold:** α = {alpha}

## Cross-validated performance by model

| Model | R2 (mean) | R2 (std) | RMSE (mean) | MAE (mean) | Folds |
|---|---|---|---|---|---|
{rows}

## Answer to RQ5

The best-performing model is **{best_display}**, with a mean cross-validated
R2 of **{best_stats['r2_mean']:+.4f}** (RMSE = {best_stats['rmse_mean']:.5f}).
This {verdict} both the mean and persistence baselines, meaning
temperature and sunshine duration {"provide real, non-trivial predictive signal for" if summary['beats_all_baselines'] else "do not reliably improve prediction of"}
daily Energy beyond what a naive baseline already achieves.

![CV comparison]({os.path.basename(output_path).replace('.md', '.png')})
"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info(f"Saved summary to {output_path}")


def main(json_path, plot_output, summary_output, alpha, log_file, log_level):
    # Convert log level
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
    }
    log_level = level_map.get(log_level.upper(), logging.DEBUG)

    logger = setup_logger(log_file, log_level)
    logger.info("Generating RQ5 predictive accuracy report")

    # Load summary
    summary = load_summary(json_path, logger)

    # Generate plot and summary
    plot_cv_comparison(summary, plot_output, logger)
    write_summary(summary, summary_output, alpha, logger)

    # Print summary to console
    print(f"Best model: {summary['best_model']} "
          f"(R2={summary['best_model_r2_mean']:.4f}, "
          f"beats baselines={summary['beats_all_baselines']})")
    print(f"Plot saved to: {plot_output}")
    print(f"Summary saved to: {summary_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a CV-comparison report from "
                    "predict_energy_w5.py's JSON summary output"
    )
    parser.add_argument("-j", "--json", required=True,
                        help="Path to the summary JSON from predict_energy_w5.py")
    parser.add_argument("-o", "--output", default="predict_energy_report.png",
                        help="Output path for the plot PNG")
    parser.add_argument("-m", "--markdown", default="predict_energy_report.md",
                        help="Output path for the markdown summary")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance threshold for reporting (default: 0.05)")
    parser.add_argument("--log", type=str,
                        default=f"{_HERE}/logs/predict_energy_report.log",
                        help="Path to log file")
    parser.add_argument("--log-level", type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='DEBUG', help="Logging level (default: DEBUG)")

    args = parser.parse_args()

    # Validate alpha
    if not (0 < args.alpha < 1):
        parser.error("--alpha must be between 0 and 1")

    main(args.json, args.output, args.markdown,
         args.alpha, args.log, args.log_level)
