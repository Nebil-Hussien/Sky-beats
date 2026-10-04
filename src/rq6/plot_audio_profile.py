#!/usr/bin/env python3
"""
plot_audio_profile.py
---------------------
RQ6 plot: radar (spider) chart of the mean audio profile on gloomy vs clear days.

Each feature is min-max normalized across all days so that features on different
scales (e.g. tempo ~117 vs valence ~0.5) are comparable on the same radar axes.
Reads the daily labelled CSV from datapreprocessing_w6.py.

Usage
-----
  python plot_audio_profile.py <daily_labeled_csv> -o <output_png>
      [--features valence,energy,danceability,acousticness,tempo]
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

# Make the shared src/utils.py importable when run as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from utils import set_plot_params  # noqa: E402

DEFAULT_FEATURES = ["valence", "energy", "danceability", "acousticness", "tempo"]
PLOT_CONFIG = Path(__file__).resolve().parent.parent / "common" / "configuration_plots.yaml"
COLORS = {"gloomy": "#4C72B0", "clear": "#DD8452"}

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


def setup_logger():
    """Configure logging to ./logs relative to this script (CWD-independent)."""
    log_path = Path(__file__).resolve().parent / "logs" / "plot_audio_profile.log"
    logging.basicConfig(filename=log_path, filemode="w", level=logging.DEBUG)
    return logging.getLogger(__name__)


def normalized_group_means(df, features):
    """Min-max normalize each feature across all days, then return the mean
    normalized profile for the 'gloomy' and 'clear' groups (DataFrame indexed
    by day_type)."""
    ranges = (df[features].max() - df[features].min()).replace(0, 1)
    norm = (df[features] - df[features].min()) / ranges
    norm["day_type"] = df["day_type"].values
    return norm[norm["day_type"].isin(["gloomy", "clear"])].groupby("day_type")[features].mean()


def plot_radar(means, features, out_png):
    """Radar chart with one closed polygon per day_type."""
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for day_type, row in means.iterrows():
        vals = row[features].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, label=day_type, color=COLORS.get(day_type), linewidth=2)
        ax.fill(angles, vals, alpha=0.15, color=COLORS.get(day_type))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.set_title("RQ6: mean audio profile — gloomy vs clear days\n(features min-max normalized)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main(input_csv, out_png, features):
    setup_logger()
    LOGGER.info("Loading %s", input_csv)
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        raise SystemExit(f"ERROR: input file not found: {input_csv}")
    except pd.errors.EmptyDataError:
        raise SystemExit(f"ERROR: input file is empty: {input_csv}")

    if "day_type" not in df.columns:
        raise SystemExit("ERROR: input missing 'day_type' column (run datapreprocessing_w6 first).")
    for feat in features:
        if feat not in df.columns:
            raise SystemExit(f"ERROR: feature column '{feat}' not in input.")

    means = normalized_group_means(df, features)
    if len(means) < 2:
        raise SystemExit("ERROR: need both 'gloomy' and 'clear' groups to plot a comparison.")

    set_plot_params(str(PLOT_CONFIG))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plot_radar(means, features, out_png)
    LOGGER.info("Saved radar chart -> %s", out_png)
    print(f"Saved radar chart -> {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ6: radar chart of mean audio profile, gloomy vs clear days")
    parser.add_argument("input_csv", help="daily labelled CSV from datapreprocessing_w6.py")
    parser.add_argument("-o", "--output", required=True, help="output PNG path")
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES),
                        help="comma-separated audio feature columns to plot")
    args = parser.parse_args()
    feats = [f.strip() for f in args.features.split(",") if f.strip()]
    main(args.input_csv, args.output, feats)
