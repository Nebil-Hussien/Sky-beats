#!/usr/bin/env python3
"""
Generates a report for a Kruskal-Wallis result for RQ4 (extreme temperature groups):
- a box plot across the three groups (Cold extreme / Normal / Hot extreme)
- a markdown summary including post-hoc pairwise results

Configurable: log file, log level, significance threshold, group order.
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

REQUIRED_KEYS = ["y_variable", "h_statistic", "p_value", "group_summary"]
DEFAULT_GROUP_ORDER = ["Cold extreme", "Normal", "Hot extreme"]


def setup_logger(log_file, log_level=logging.DEBUG):
    """Configure a logger with file and console output."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('kruskal_report_w4')
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
    Load and validate the Kruskal-Wallis result JSON.

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
            "Was this produced by kruskal_extremes_w4.py (JSON output)?"
        )

    # Validate that group_summary has the expected structure
    if not isinstance(result.get("group_summary"), dict):
        logger.error("group_summary is not a dictionary")
        raise ValueError("group_summary must be a dictionary")

    return result


def _fmt_p(p):
    """Format p‑value: show '<0.001' for very small values."""
    return "<0.001" if p < 0.001 else f"{p:.4f}"


def plot_boxplot(result, output_path, group_order, logger):
    """
    Draw a box plot from the precomputed group_summary quartiles.

    Args:
        result (dict): Parsed Kruskal-Wallis result.
        output_path (str): Path to save the PNG.
        group_order (list): Ordered list of group labels.
        logger (logging.Logger): Logger instance.
    """
    summary = result["group_summary"]

    # Only plot groups that exist in the summary and follow the order
    groups = [g for g in group_order if g in summary]
    if not groups:
        logger.error("No recognized groups found in group_summary")
        raise ValueError("group_summary has no recognized group labels")

    stats = []
    for g in groups:
        s = summary[g]
        stats.append({
            "label": f"{g}\n(n={s['n']})",
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
        f"{result['y_variable']} across {result.get('cat_column', 'temperature group')}",
        loc="left", fontsize=13, fontweight="medium"
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    strength = result.get("effect_size", {}).get("label", "unspecified")
    stat_text = (f"H = {result['h_statistic']:.2f}   "
                 f"p = {_fmt_p(result['p_value'])}   "
                 f"effect size = {strength}")

    fig.text(0.01, -0.02, stat_text, fontsize=9, color="#52514e")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot to {output_path}")


def write_summary(result, output_path, alpha, group_order, logger):
    """
    Write a Markdown summary of the Kruskal-Wallis result.

    Args:
        result (dict): Parsed Kruskal-Wallis result.
        output_path (str): Path to save the Markdown file.
        alpha (float): Significance threshold.
        group_order (list): Ordered list of group labels.
        logger (logging.Logger): Logger instance.
    """
    significant = result.get("significant_at_0.05", result["p_value"] < alpha)
    verdict = "statistically significant" if significant else "not statistically significant"
    strength = result.get("effect_size", {})
    eps_sq = strength.get("epsilon_squared", "—")
    label = strength.get("label", "unspecified")
    groups_used = result.get("buckets_used", list(result["group_summary"].keys()))

    summary_rows = "\n".join(
        f"| {b} | {s['n']} | {s['median']:.4f} | {s['q1']:.4f} | {s['q3']:.4f} |"
        for b, s in result["group_summary"].items()
    )

    posthoc = result.get("posthoc_pairwise_p", {})
    if posthoc:
        posthoc_rows = "\n".join(
            f"| {k.replace('_vs_', ' vs. ')} | {_fmt_p(v)} | "
            f"{'Significant' if v < alpha else 'Not significant'} |"
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

    md = f"""# Extreme temperature group comparison report (RQ4)

**Variable:** `{result['y_variable']}` across `{result.get('cat_column', 'temperature group')}`

**Groups used:** {', '.join(groups_used)}

## Group summary

| Group | n | Median | Q1 | Q3 |
|---|---|---|---|---|
{summary_rows}

## Kruskal-Wallis test

| Statistic | Value |
|---|---|
| H | {result['h_statistic']:.4f} |
| p-value | {_fmt_p(result['p_value'])} |
| Effect size (epsilon-squared) | {eps_sq} ({label}) |
| Significant at α = {alpha} | {"Yes" if significant else "No"} |

{posthoc_section}
## Interpretation

Across the groups **{', '.join(groups_used)}**, `{result['y_variable']}`
shows a **{label}** group difference (H = {result['h_statistic']:.2f}).
This result is **{verdict}** at the {alpha} significance level (p = {_fmt_p(result['p_value'])}).

![box plot]({os.path.basename(output_path).replace('.md', '.png')})
"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info(f"Saved summary to {output_path}")


def main(json_path, plot_output, summary_output, group_order, alpha, log_file, log_level):
    # Convert log level
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
    }
    log_level = level_map.get(log_level.upper(), logging.DEBUG)

    logger = setup_logger(log_file, log_level)
    logger.info("Generating RQ4 extreme-temperature comparison report")

    # Set plotting parameters from config
    config_path = os.path.join(_HERE, "..", "common", "configuration_plots.yaml")
    set_plot_params(config_path)

    # Load result
    result = load_result(json_path, logger)

    # Generate plot and summary
    plot_boxplot(result, plot_output, group_order, logger)
    write_summary(result, summary_output, alpha, group_order, logger)

    # Print short summary to console
    print(f"H={result['h_statistic']:.2f}, p={_fmt_p(result['p_value'])}, "
          f"effect_size={result.get('effect_size', {}).get('label')}")
    print(f"Plot saved to: {plot_output}")
    print(f"Summary saved to: {summary_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a box-plot report from kruskal_extremes_w4.py's JSON output"
    )
    parser.add_argument("-j", "--json", required=True,
                        help="Path to the result JSON from kruskal_extremes_w4.py")
    parser.add_argument("-o", "--output", default="kruskal_report_w4.png",
                        help="Output path for the plot PNG")
    parser.add_argument("-m", "--markdown", default="kruskal_report_w4.md",
                        help="Output path for the markdown summary")
    parser.add_argument("--group-order", type=str,
                        default=",".join(DEFAULT_GROUP_ORDER),
                        help=f"Comma-separated group order (default: {','.join(DEFAULT_GROUP_ORDER)})")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level to report (default: 0.05)")
    parser.add_argument("--log", type=str, default=f"{_HERE}/logs/kruskal_report_w4.log",
                        help="Path to log file")
    parser.add_argument("--log-level", type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='DEBUG', help="Logging level (default: DEBUG)")

    args = parser.parse_args()

    # Validate alpha
    if not (0 < args.alpha < 1):
        parser.error("--alpha must be between 0 and 1")

    # Parse group order
    group_order = [g.strip() for g in args.group_order.split(',') if g.strip()]
    if not group_order:
        group_order = DEFAULT_GROUP_ORDER

    main(args.json, args.output, args.markdown, group_order,
         args.alpha, args.log, args.log_level)
