#!/usr/bin/env python3
"""
plot_results.py
---------------
Generates publication-ready feature importance visualizations for RQ7.
Loads the trained models via joblib and saves a clean 2x2 grid chart.
"""

import argparse
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive: write files, no GUI window
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.inspection import permutation_importance

# Real dataframe column names, in the same order train_model.py trains on.
# Used only for the permutation-importance fallback below.
ACTUAL_FEATURE_COLUMNS = [
    "precipitation_mm", "sunshine_hours", "temperature_mean_c",
    "is_weekend", "day_of_week", "month", "is_holiday"
]


def main(model_path, output_png, data_csv=None):
    # 1. Load the dictionary of trained target models
    if not Path(model_path).exists():
        raise SystemExit(f"ERROR: Model file not found at {model_path}")

    models_dict = joblib.load(model_path)

    # 2. Define our targets and features exactly as they match the model structures
    targets = ["avg_valence", "avg_energy", "avg_danceability", "avg_tempo"]
    features = [
        "precipitation", "sunshine_hours", "temperature_mean_c",
        "is_weekend", "day_of_week", "month", "is_holiday"
    ]

    # 3. Set up a clean 2x2 grid of subplots
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # 4. Extract importances and plot for each target
    data_df = None  # loaded lazily, only if some model needs permutation importance
    for i, target in enumerate(targets):
        model = models_dict[target]

        # Get importances: mean decrease in impurity for tree models, coefficients for Ridge
        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
            value_label = "Predictive Weight (Mean Decrease in Impurity)"
            xlim = (0, 1.0)  # non-negative and sum to 1
        elif hasattr(model, "coef_"):
            values = model.coef_
            value_label = "Regression Coefficient"
            xlim = None  # coefficients can be negative/unbounded, no fixed scale
        else:
            # Models like HistGradientBoostingRegressor expose neither
            # attribute, so fall back to permutation importance instead.
            if data_csv is None:
                raise SystemExit(
                    f"ERROR: model for '{target}' ({type(model).__name__}) has neither "
                    "feature_importances_ nor coef_; pass --data <daily_processed.csv> "
                    "to compute permutation importance instead."
                )
            if data_df is None:
                data_df = pd.read_csv(data_csv)
            result = permutation_importance(
                model, data_df[ACTUAL_FEATURE_COLUMNS], data_df[target],
                n_repeats=10, random_state=42, scoring="r2"
            )
            values = result.importances_mean
            value_label = "Permutation Importance (mean R² drop)"
            xlim = None  # can be negative (feature actively hurts the model)

        # Build clean dataframe for Seaborn, ranked by magnitude (handles signed coefficients too)
        df_plot = pd.DataFrame({
            'Feature': features,
            'Importance': values
        }).sort_values(by='Importance', key=lambda s: s.abs(), ascending=False)

        # Human-friendly clean labels for the y-axis
        label_map = {
            "precipitation": "Precipitation",
            "sunshine_hours": "Sunshine Hours",
            "temperature_mean_c": "Mean Temperature",
            "month": "Month (Season)",
            "day_of_week": "Day of Week",
            "is_weekend": "Is Weekend",
            "is_holiday": "Is Holiday"
        }
        df_plot['Feature'] = df_plot['Feature'].map(label_map)

        # Draw horizontal bars
        sns.barplot(
            x='Importance',
            y='Feature',
            data=df_plot,
            ax=axes[i],
            palette='viridis',
            hue='Feature',
            legend=False
        )

        # Styling configurations
        clean_title = target.replace("avg_", "").upper()
        axes[i].set_title(f"Drivers for {clean_title}", fontsize=13, fontweight='bold', pad=10)
        axes[i].set_xlabel(value_label, fontsize=10)
        axes[i].set_ylabel('')
        if xlim is not None:
            axes[i].set_xlim(*xlim)  # Fixed scale makes visual comparison easy

    plt.tight_layout()

    # 5. Save chart output
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"Success: Feature importance plot saved to {output_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Feature Importance Charts")
    parser.add_argument("model_path", help="Path to your trained .pkl models dictionary")
    parser.add_argument("output_png", help="Path where the output plot PNG should be saved")
    parser.add_argument(
        "--data", dest="data_csv", default=None,
        help="Daily processed CSV (needed only as a fallback for models "
             "without feature_importances_/coef_, e.g. HistGradientBoostingRegressor)"
    )
    args = parser.parse_args()

    main(args.model_path, args.output_png, args.data_csv)
