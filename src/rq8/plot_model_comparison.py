#!/usr/bin/env python3
"""
plot_model_comparison.py
------------------------
RQ8 plot: grouped bar charts comparing the season/month-only baseline against
the weather-augmented model on the held-out days — test R² (left panel) and
test MAE (right panel), one bar group per estimator, annotated with the
R² delta (weather − baseline).

Reads the metrics table produced by compare_models_w8.py and writes a PNG.

Usage
-----
  python plot_model_comparison.py <metrics_csv> -o <output_png>
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

SET_COLORS = {"baseline": "#9e9e9e", "weather": "#1f77b4"}
SET_LABELS = {"baseline": "Season baseline (month only)",
              "weather": "Month + weather features"}


def setup_logger():
    """Configure logging to ./logs relative to this script (CWD-independent)."""
    log_path = Path(__file__).resolve().parent / "logs" / "plot_model_comparison.log"
    logging.basicConfig(filename=log_path, filemode="w", level=logging.DEBUG)
    return logging.getLogger(__name__)


def plot_comparison(metrics, out_png):
    """Grouped bars of test R² and MAE, baseline vs weather, per estimator."""
    models = list(metrics["model"].unique())
    x = np.arange(len(models))
    width = 0.35

    fig, (ax_r2, ax_mae) = plt.subplots(1, 2, figsize=(13, 6))
    pivot = metrics.set_index(["model", "feature_set"])

    for offset, feature_set in ((-width / 2, "baseline"), (width / 2, "weather")):
        r2 = [pivot.loc[(m, feature_set), "r2"] for m in models]
        mae = [pivot.loc[(m, feature_set), "mae"] for m in models]
        ax_r2.bar(x + offset, r2, width, label=SET_LABELS[feature_set],
                  color=SET_COLORS[feature_set])
        ax_mae.bar(x + offset, mae, width, label=SET_LABELS[feature_set],
                   color=SET_COLORS[feature_set])

    for i, model in enumerate(models):
        delta = pivot.loc[(model, "weather"), "r2"] - pivot.loc[(model, "baseline"), "r2"]
        # anchor above the taller bar (not y=0) so it stays readable when R2 < 0
        top = max(pivot.loc[(model, "weather"), "r2"],
                  pivot.loc[(model, "baseline"), "r2"])
        ax_r2.annotate(f"ΔR² = {delta:+.3f}", (x[i], top),
                       textcoords="offset points", xytext=(0, 8),
                       ha="center", fontsize=11, fontweight="bold")

    ax_r2.axhline(0, color="black", linewidth=0.8)
    ax_r2.set_xticks(x, models)
    ax_r2.set_ylabel("Test R² (higher is better)")
    ax_r2.set_title("Explained variance on held-out days")
    ax_r2.legend()

    ax_mae.set_xticks(x, models)
    ax_mae.set_ylabel("Test MAE (lower is better)")
    ax_mae.set_title("Mean absolute error on held-out days")
    ax_mae.legend()

    n_test = int(metrics["n_test"].iloc[0])
    fig.suptitle(f"RQ8: Does weather predict valence beyond season? "
                 f"(chronological hold-out, {n_test} test days)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(metrics_csv, out_png):
    setup_logger()
    LOGGER.info("Loading metrics %s", metrics_csv)
    try:
        metrics = pd.read_csv(metrics_csv)
    except FileNotFoundError:
        raise SystemExit(f"ERROR: metrics file not found: {metrics_csv}")
    except pd.errors.EmptyDataError:
        raise SystemExit(f"ERROR: metrics file is empty: {metrics_csv}")

    required = {"model", "feature_set", "r2", "mae", "n_test"}
    missing = required - set(metrics.columns)
    if missing:
        raise SystemExit(f"ERROR: metrics CSV missing columns: {sorted(missing)}")
    if not {"baseline", "weather"}.issubset(set(metrics["feature_set"])):
        raise SystemExit("ERROR: metrics CSV must contain both 'baseline' and "
                         "'weather' feature sets")

    set_plot_params(str(PLOT_CONFIG))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(metrics, out_png)
    LOGGER.info("Saved chart -> %s", out_png)
    print(f"Saved chart -> {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ8: bar chart comparing weather model vs season baseline")
    parser.add_argument("metrics_csv", help="metrics CSV from compare_models_w8.py")
    parser.add_argument("-o", "--output", required=True, help="output PNG path")
    args = parser.parse_args()
    main(args.metrics_csv, args.output)
