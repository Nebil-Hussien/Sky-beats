#!/usr/bin/env python3
"""
plot_genre_shift.py
-------------------
RQ3 plot: horizontal bar chart of the genres with the largest absolute
stream-share shift between the rainiest and driest months. Reads the per-genre
shift table produced by genre_shift.py and writes a PNG.

Usage
-----
  python plot_genre_shift.py <shift_csv> -o <output_png> [--top-genres 15]
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, so it works headless / under Snakemake
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Make the shared src/utils.py importable when this is run as a standalone script
# (scripts are invoked by Snakemake from their own workflow directory, not as a package).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from utils import set_plot_params  # noqa: E402

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

# Shared plot styling config lives at src/configuration_plots.yaml.
PLOT_CONFIG = Path(__file__).resolve().parent.parent / "common" / "configuration_plots.yaml"


def setup_logger():
    """Configure logging to ./logs relative to this script (CWD-independent)."""
    log_path = Path(__file__).resolve().parent / "logs" / "plot_genre_shift.log"
    logging.basicConfig(filename=log_path, filemode="w", level=logging.DEBUG)
    return logging.getLogger(__name__)


def plot_shift(shift, top_genres, out_png):
    """Bar chart of the genres with the largest absolute stream-share shift."""
    top = shift.reindex(shift["stream_shift"].abs().sort_values(ascending=False).index)
    top = top.head(top_genres).sort_values("stream_shift")

    colors = ["#d62728" if v < 0 else "#2ca02c" for v in top["stream_shift"]]
    plt.figure(figsize=(9, max(4, 0.4 * len(top))))
    plt.barh(top.index, top["stream_shift"] * 100, color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Stream-share shift, rainy − dry months (percentage points)")
    plt.ylabel("Genre")
    plt.title("RQ3: Genres more popular in rainy (green) vs dry (red) months")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main(shift_csv, out_png, top_genres):
    setup_logger()
    LOGGER.info("Loading shift table %s", shift_csv)
    try:
        shift = pd.read_csv(shift_csv, index_col=0)
    except FileNotFoundError:
        raise SystemExit(f"ERROR: shift table not found: {shift_csv}")
    except pd.errors.EmptyDataError:
        raise SystemExit(f"ERROR: shift table is empty: {shift_csv}")

    if "stream_shift" not in shift.columns:
        raise SystemExit("ERROR: input is missing the 'stream_shift' column.")

    set_plot_params(str(PLOT_CONFIG))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plot_shift(shift, top_genres, out_png)
    LOGGER.info("Saved chart -> %s", out_png)
    print(f"Saved chart -> {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ3: bar chart of largest genre stream-share shifts (rainy vs dry)")
    parser.add_argument("shift_csv", help="per-genre shift CSV from genre_shift.py")
    parser.add_argument("-o", "--output", required=True, help="output PNG path")
    parser.add_argument("--top-genres", type=int, default=15,
                        help="number of genres (by largest absolute shift) to plot (default 15)")
    args = parser.parse_args()
    main(args.shift_csv, args.output, args.top_genres)
