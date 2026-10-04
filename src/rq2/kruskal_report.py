#!/usr/bin/env python3
"""
Generates a report from the Kruskal-Wallis result JSON:
- a box plot across precipitation buckets
- a Markdown summary including post‑hoc pairwise results (if any)

The script is configurable: output paths, log file, log level, and bucket order.
"""

import os
import sys
import argparse
import logging
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from common.utils import set_plot_params

_HERE = os.path.dirname(os.path.abspath(__file__))

# Expected keys in the Kruskal-Wallis result JSON
REQUIRED_KEYS = ["y_variable", "h_statistic", "p_value", "group_summary"]

# Default bucket order (can be overridden via command line)
DEFAULT_BUCKET_ORDER = ["No rain", "Low", "Moderate", "High"]


def setup_logger(log_file, log_level=logging.DEBUG):
    """Configure a logger with file and console output."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('kruskal_report')
    logger.setLevel(log_level)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(fh)

    # Console handler for immediate feedback
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(ch)

    return logger


def load_result(json_path, logger):
    """Load and validate the Kruskal-Wallis result JSON.

    Args:
        json_path (str): path to the JSON file
        logger (logging.Logger): logger instance

    Raises:
        ValueError: if required keys are missing
        FileNotFoundError: if the JSON file does not exist

    Returns:
        dict: parsed result
    """
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
            "Was this produced by kruskal_wallis.py (JSON output)?"
        )
    return result


def _fmt_p(p):
    """Format p‑value for display (show <0.001 for very small values)."""
    return "<0.001" if p < 0.001 else f"{p:.4f}"


def plot_boxplot(result, output_path, bucket_order, logger):
    """Draw a box plot from the pre‑computed group_summary quartiles.

    Args:
        result (dict): parsed Kruskal-Wallis result
        output_path (str): path to save the PNG
        bucket_order (list): ordered list of bucket labels
        logger (logging.Logger): logger instance
    """
    summary = result["group_summary"]
    # Keep only buckets that appear in the summary, in the given order
    buckets = [b for b in bucket_order if b in summary]
    if not buckets:
        logger.error("No recognized buckets found in group_summary")
        raise ValueError("group_summary has no recognized bucket labels")

    stats = []
    for b in buckets:
        s = summary[b]
        stats.append({
            "label": f"{b}\n(n={s['n']})",
            "med": s["median"],
            "q1": s["q1"],
            "q3": s["q3"],
            "whislo": s["q1"],
            "whishi": s["q3"],
            "fliers": [],
        })

    fig, ax = plt.subplots()
    bp = ax.bxp(stats, patch_artist=True, widths=0.5, showfliers=False)

    # Customise box plot colours
    for box in bp["boxes"]:
        box.set(facecolor="#B5D4F4", edgecolor="#185FA5", linewidth=1.3)
    for median in bp["medians"]:
        median.set(color="#042C53", linewidth=2)
    for whisker in bp["whiskers"]:
        whisker.set(color="#378ADD", linewidth=1.2)
    for cap in bp["caps"]:
        cap.set(color="#378ADD", linewidth=1.2)

    ax.set_ylabel(result["y_variable"])
    ax.set_title(
        f"{result['y_variable']} across {result.get('cat_column', 'bucket')}",
        loc="left", fontsize=13, fontweight="medium"
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    strength = result.get("effect_size", {}).get("label", "unspecified")
    stat_text = (
        f"H = {result['h_statistic']:.2f}   "
        f"p = {_fmt_p(result['p_value'])}   "
        f"effect size = {strength}"
    )
    fig.text(0.01, -0.02, stat_text, fontsize=9, color="#52514e")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot to {output_path}")


def write_summary(result, output_path, bucket_order, logger):
    """Write a Markdown summary of the result, including post‑hoc comparisons.

    Args:
        result (dict): parsed Kruskal-Wallis result
        output_path (str): path to save the Markdown file
        bucket_order (list): ordered list of bucket labels
        logger (logging.Logger): logger instance
    """
    significant = result.get("significant_at_0.05", result["p_value"] < 0.05)
    verdict = "statistically significant" if significant else "not statistically significant"
    strength = result.get("effect_size", {})
    eps_sq = strength.get("epsilon_squared", "—")
    label = strength.get("label", "unspecified")
    buckets_used = result.get("buckets_used", list(result["group_summary"].keys()))

    summary_rows = "\n".join(
        f"| {b} | {s['n']} | {s['median']:.4f} | {s['q1']:.4f} | {s['q3']:.4f} |"
        for b, s in result["group_summary"].items()
    )

    posthoc = result.get("posthoc_pairwise_p", {})
    if posthoc:
        posthoc_rows = "\n".join(
            f"| {k.replace('_vs_', ' vs. ')} | {_fmt_p(v)} | "
            f"{'Significant' if v < 0.05 else 'Not significant'} |"
            for k, v in posthoc.items()
        )
        posthoc_section = f"""## Pairwise comparisons (Dunn's test, Bonferroni-corrected)

| Comparison | Adjusted p-value | Result |
|---|---|---|
{posthoc_rows}
"""
    else:
        posthoc_section = ("## Pairwise comparisons\n\nNot run — the omnibus "
                           "test was not significant.\n")

    md = f"""# Precipitation bucket comparison report

**Variable:** `{result['y_variable']}` across `{result.get('cat_column', 'precipitation bucket')}`

**Buckets used:** {', '.join(buckets_used)}

## Group summary

| Bucket | n | Median | Q1 | Q3 |
|---|---|---|---|---|
{summary_rows}

## Kruskal-Wallis test

| Statistic | Value |
|---|---|
| H | {result['h_statistic']:.4f} |
| p-value | {_fmt_p(result['p_value'])} |
| Effect size (epsilon-squared) | {eps_sq} ({label}) |
| Significant at 0.05 | {"Yes" if significant else "No"} |

{posthoc_section}
## Interpretation

Across the buckets **{', '.join(buckets_used)}**, `{result['y_variable']}`
shows a **{label}** group difference (H = {result['h_statistic']:.2f}).
This result is **{verdict}** at the 0.05 level (p = {_fmt_p(result['p_value'])}).

![box plot]({os.path.basename(output_path).replace('.md', '.png')})
"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(md)
    logger.info(f"Saved summary to {output_path}")


def main(json_path, plot_output, summary_output, bucket_order, log_file, log_level):
    """Main entry point: load JSON, generate plot and markdown."""
    # Convert log level string to logging constant
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
    }
    log_level = level_map.get(log_level.upper(), logging.DEBUG)

    logger = setup_logger(log_file, log_level)
    logger.info("Generating precipitation bucket comparison report")

    # Load and validate result
    result = load_result(json_path, logger)

    # Plot
    plot_boxplot(result, plot_output, bucket_order, logger)

    # Markdown summary
    write_summary(result, summary_output, bucket_order, logger)

    # Print short summary to console
    print(f"H={result['h_statistic']:.2f}, p={_fmt_p(result['p_value'])}, "
          f"effect_size={result.get('effect_size', {}).get('label')}")
    print(f"Plot saved to: {plot_output}")
    print(f"Summary saved to: {summary_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a box‑plot report from the Kruskal-Wallis result JSON."
    )
    parser.add_argument("-j", "--json", required=True,
                        help="Path to the result JSON from kruskal_wallis.py")
    parser.add_argument("-o", "--output", default="precip_bucket_report.png",
                        help="Output path for the plot PNG (default: precip_bucket_report.png)")
    parser.add_argument("-m", "--markdown", default="precip_bucket_report.md",
                        help="Output path for the markdown summary (default: precip_bucket_report.md)")
    parser.add_argument("--bucket-order", type=str,
                        default=",".join(DEFAULT_BUCKET_ORDER),
                        help=f"Comma‑separated bucket order (default: {','.join(DEFAULT_BUCKET_ORDER)})")
    parser.add_argument("--log", type=str,
                        default=f"{_HERE}/logs/precip_bucket_report.log",
                        help="Path to log file (default: logs/precip_bucket_report.log)")
    parser.add_argument("--log-level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='DEBUG', help="Logging level (default: DEBUG)")

    args = parser.parse_args()

    # Parse bucket order
    bucket_order = [b.strip() for b in args.bucket_order.split(',') if b.strip()]
    if not bucket_order:
        bucket_order = DEFAULT_BUCKET_ORDER

    # Set plot parameters from shared config
    config_path = os.path.join(_HERE, "..", "common", "configuration_plots.yaml")
    set_plot_params(config_path)

    main(args.json, args.output, args.markdown, bucket_order,
         args.log, args.log_level)
