#!/usr/bin/env python3
"""
plot_sunshine_valence.py
------------------------
RQ1 plot: 2x2 grid of scatter plots (one per meteorological season) of daily
sunshine duration vs mean chart valence, each with a least-squares trend line
and annotated with that season's Spearman rho / p-value from the correlation
table produced by sunshine_valence_correlation.py.

Usage
-----
  python plot_sunshine_valence.py <daily_csv> -c <correlation_csv> -o <output_png>
      [-x sunshine_hours] [-y valence]
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, so it works headless / under Snakemake
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Make the shared src/utils.py importable when this is run as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from utils import set_plot_params  # noqa: E402

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

# Shared plot styling config lives at src/configuration_plots.yaml.
PLOT_CONFIG = Path(__file__).resolve().parent.parent / "common" / "configuration_plots.yaml"

SEASON_ORDER = ["winter", "spring", "summer", "autumn"]
SEASON_COLORS = {
    "winter": "#1f77b4",
    "spring": "#2ca02c",
    "summer": "#ff7f0e",
    "autumn": "#8c564b",
}


def setup_logger():
    """Configure logging to ./logs relative to this script (CWD-independent)."""
    log_path = Path(__file__).resolve().parent / "logs" / "plot_sunshine_valence.log"
    logging.basicConfig(filename=log_path, filemode="w", level=logging.DEBUG)
    return logging.getLogger(__name__)


def plot_seasonal_scatter(daily, correlations, x_variable, y_variable, out_png):
    """2x2 seasonal scatter of x vs y with trend lines and rho annotations."""
    stats = correlations.set_index("group")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)

    for ax, season in zip(axes.flat, SEASON_ORDER):
        group = daily.loc[daily["season"] == season, [x_variable, y_variable]].dropna()
        ax.scatter(group[x_variable], group[y_variable],
                   s=12, alpha=0.5, color=SEASON_COLORS[season])

        if len(group) >= 2:
            slope, intercept = np.polyfit(group[x_variable], group[y_variable], 1)
            xs = np.linspace(group[x_variable].min(), group[x_variable].max(), 100)
            ax.plot(xs, slope * xs + intercept, color="black", linewidth=1.5)

        if season in stats.index:
            rho = stats.loc[season, "rho"]
            p_value = stats.loc[season, "p_value"]
            ax.set_title(f"{season} (rho = {rho:.2f}, p = {p_value:.3g})")
        else:
            ax.set_title(season)

    fig.supxlabel("Daily sunshine duration (hours)")
    fig.supylabel("Mean valence of German Top 200")
    overall = stats.loc["all", "rho"] if "all" in stats.index else float("nan")
    fig.suptitle(f"RQ1: Sunshine vs chart valence by season (overall rho = {overall:.2f})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(daily_csv, correlation_csv, out_png, x_variable, y_variable):
    setup_logger()
    LOGGER.info("Loading daily data %s and correlations %s", daily_csv, correlation_csv)
    try:
        daily = pd.read_csv(daily_csv)
        correlations = pd.read_csv(correlation_csv)
    except FileNotFoundError as exc:
        raise SystemExit(f"ERROR: input file not found: {exc.filename}")
    except pd.errors.EmptyDataError:
        raise SystemExit("ERROR: an input file is empty")

    missing = {x_variable, y_variable, "season"} - set(daily.columns)
    if missing:
        raise SystemExit(f"ERROR: daily CSV missing required columns: {sorted(missing)}")
    if not {"group", "rho", "p_value"}.issubset(correlations.columns):
        raise SystemExit("ERROR: correlation CSV missing 'group'/'rho'/'p_value' columns")

    set_plot_params(str(PLOT_CONFIG))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plot_seasonal_scatter(daily, correlations, x_variable, y_variable, out_png)
    LOGGER.info("Saved chart -> %s", out_png)
    print(f"Saved chart -> {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ1: seasonal scatter plots of sunshine vs valence")
    parser.add_argument("daily_csv", help="daily CSV from datapreprocessing_w1.py")
    parser.add_argument("-c", "--correlations", required=True,
                        help="correlation CSV from sunshine_valence_correlation.py")
    parser.add_argument("-o", "--output", required=True, help="output PNG path")
    parser.add_argument("-x", "--x_variable", default="sunshine_hours",
                        help="weather column (default: sunshine_hours)")
    parser.add_argument("-y", "--y_variable", default="valence",
                        help="audio feature column (default: valence)")
    args = parser.parse_args()
    main(args.daily_csv, args.correlations, args.output, args.x_variable, args.y_variable)
