#!/usr/bin/env python3
"""
plot_classifier.py
------------------
RQ9 plot: a two-panel report of the gloomy-vs-clear classifier.

  Left  : confusion matrix on the held-out test set (how often each true class
          is predicted correctly vs confused with the other).
  Right : per-feature importance of the best model (signed coefficient for
          logistic regression, Gini importance for random forest).

Reads the two tidy CSVs produced by train_classifier.py (predictions and
feature importances), so plotting stays decoupled from model training.

Usage
-----
  python plot_classifier.py <predictions_csv> <importance_csv> -o <output_png>
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, so it works headless / under Snakemake
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Make the shared src/utils.py importable when run as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from utils import set_plot_params  # noqa: E402

CLASSES = ["gloomy", "clear"]  # fixed axis order, matches train_classifier.py
PLOT_CONFIG = Path(__file__).resolve().parent.parent / "common" / "configuration_plots.yaml"

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


def setup_logger():
    """Configure logging to ./logs relative to this script (CWD-independent)."""
    log_path = Path(__file__).resolve().parent / "logs" / "plot_classifier.log"
    logging.basicConfig(filename=log_path, filemode="w", level=logging.DEBUG)
    return logging.getLogger(__name__)


def confusion_counts(preds):
    """Confusion matrix as a DataFrame (rows = true, cols = predicted), with a
    fixed gloomy/clear order and zeros for any absent cell."""
    matrix = pd.crosstab(preds["y_true"], preds["y_pred"])
    return matrix.reindex(index=CLASSES, columns=CLASSES, fill_value=0)


def plot_report(matrix, importance, out_png):
    """Draw the confusion matrix (left) and feature importances (right)."""
    fig, (ax_cm, ax_imp) = plt.subplots(1, 2, figsize=(14, 6))

    # Confusion matrix
    ax_cm.grid(False)
    im = ax_cm.imshow(matrix.values, cmap="Blues")
    ax_cm.set_xticks(range(len(CLASSES)), labels=CLASSES)
    ax_cm.set_yticks(range(len(CLASSES)), labels=CLASSES)
    ax_cm.set_xlabel("Predicted day type")
    ax_cm.set_ylabel("True day type")
    ax_cm.set_title("Confusion matrix (held-out test days)")
    # Annotate each cell; pick a readable text colour against the cell shade.
    threshold = matrix.values.max() / 2 if matrix.values.max() else 0
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            count = matrix.values[i, j]
            ax_cm.text(j, i, str(count), ha="center", va="center",
                       color="white" if count > threshold else "black")
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    # Feature importance
    kind = importance["kind"].iloc[0] if "kind" in importance.columns else "importance"
    imp = importance.sort_values("importance")
    ax_imp.barh(imp["feature"], imp["importance"], color="#4C72B0")
    ax_imp.set_xlabel(kind.replace("_", " "))
    ax_imp.set_title(f"Feature importance ({kind.replace('_', ' ')})")
    if (imp["importance"] < 0).any():
        ax_imp.axvline(0, color="grey", linewidth=0.8)  # reference for signed coefficients

    fig.suptitle("RQ9: predicting the day's weather (gloomy vs clear) from the audio profile")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main(preds_csv, importance_csv, out_png):
    setup_logger()
    LOGGER.info("Loading %s and %s", preds_csv, importance_csv)
    try:
        preds = pd.read_csv(preds_csv)
        importance = pd.read_csv(importance_csv)
    except FileNotFoundError as exc:
        raise SystemExit(f"ERROR: input file not found: {exc.filename}")
    except pd.errors.EmptyDataError:
        raise SystemExit("ERROR: an input file is empty.")

    for col in ("y_true", "y_pred"):
        if col not in preds.columns:
            raise SystemExit(f"ERROR: predictions CSV missing '{col}' column.")
    if "importance" not in importance.columns or "feature" not in importance.columns:
        raise SystemExit("ERROR: importance CSV missing 'feature'/'importance' columns.")

    matrix = confusion_counts(preds)

    set_plot_params(str(PLOT_CONFIG))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plot_report(matrix, importance, out_png)
    LOGGER.info("Saved classifier report -> %s", out_png)
    print(f"Saved classifier report -> {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ9: plot confusion matrix and feature importance of the classifier")
    parser.add_argument("predictions_csv", help="held-out predictions CSV from train_classifier.py")
    parser.add_argument("importance_csv", help="feature-importance CSV from train_classifier.py")
    parser.add_argument("-o", "--output", required=True, help="output PNG path")
    args = parser.parse_args()
    main(args.predictions_csv, args.importance_csv, args.output)
