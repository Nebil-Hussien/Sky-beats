#!/usr/bin/env python3
"""
train_classifier.py
-------------------
RQ9: Can the day's weather category be inferred from the music alone?

Trains classifiers to predict a day's weather label (``gloomy`` vs ``clear``)
from its aggregated audio profile ONLY (valence, energy, danceability,
acousticness, tempo). This is the reverse direction of RQ6.

Two models are trained and compared (see MODEL_REGISTRY):
  - logistic_regression : linear, interpretable coefficients (feature-scaled)
  - random_forest       : non-linear, gives impurity-based feature importances

Every model is judged against a majority-class BASELINE (DummyClassifier). A
result only "means something" if it beats that baseline, so the baseline is
reported in the same metrics table. Model selection uses stratified
cross-validated accuracy on the training split; the winner is then evaluated
once on a held-out test set and saved.

Outputs (all paths are CLI arguments):
  - metrics CSV        : one row per model + baseline (CV accuracy, held-out
                         accuracy / precision / recall / F1)
  - model file (.pkl)  : the best fitted pipeline, reusable to classify new days
  - predictions CSV    : held-out test rows with true and predicted labels
                         (consumed by plot_classifier.py for the confusion matrix)
  - importance CSV     : per-feature importance of the best model
                         (signed coefficient for logreg, Gini importance for RF)

Usage
-----
  python train_classifier.py <two_class_csv> \
      -o metrics.csv -m model.pkl --predictions preds.csv --importance importance.csv \
      [--features valence,energy,danceability,acousticness,tempo] \
      [--models logistic_regression,random_forest] \
      [--test-size 0.25] [--cv-folds 5] [--random-seed 42]
"""

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COL = "day_type"
CLASSES = ["gloomy", "clear"]  # fixed label order for consistent reporting
DEFAULT_FEATURES = ["valence", "energy", "danceability", "acousticness", "tempo"]

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


# A pipeline per model so pre-processing travels with the estimator. Scaling is
# needed for logistic regression (tempo ~117 vs other features in [0, 1]) and is
# harmless for the tree model, so both are wrapped uniformly.
def _build_model(name, seed):
    if name == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=seed)
    elif name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2, random_state=seed)
    else:
        raise SystemExit(
            f"ERROR: unknown model '{name}'. Available: logistic_regression, random_forest")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def setup_logger():
    """Configure logging to ./logs relative to this script (CWD-independent)."""
    log_path = Path(__file__).resolve().parent / "logs" / "train_classifier.log"
    logging.basicConfig(filename=log_path, filemode="w", level=logging.DEBUG)
    return logging.getLogger(__name__)


def _test_metrics(y_true, y_pred):
    """Held-out metrics. Macro averaging is robust to the mild class imbalance;
    precision/recall use zero_division=0 so an unpredicted class scores 0, not NaN.
    """
    return {
        "test_accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }


def evaluate_models(df, features, model_names, test_size, cv_folds, seed):
    """Train/compare models + baseline. Returns (metrics_df, best_name,
    best_fitted_pipeline, test_predictions_df, importance_df).
    """
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: feature columns missing from input: {missing}")

    X = df[features]
    y = df[TARGET_COL]

    present = set(y.unique())
    if not {"gloomy", "clear"} <= present:
        raise SystemExit(
            f"ERROR: need both 'gloomy' and 'clear' in '{TARGET_COL}'; got {sorted(present)}.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    rows = []
    fitted = {}
    # Baseline: always predict the majority class. The bar every model must clear.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline_cv = cross_val_score(baseline, X_train, y_train, cv=cv, scoring="accuracy")
    baseline.fit(X_train, y_train)
    rows.append({
        "model": "baseline_majority",
        "cv_accuracy_mean": round(baseline_cv.mean(), 4),
        "cv_accuracy_std": round(baseline_cv.std(), 4),
        **_test_metrics(y_test, baseline.predict(X_test)),
    })

    for name in model_names:
        pipe = _build_model(name, seed)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        rows.append({
            "model": name,
            "cv_accuracy_mean": round(cv_scores.mean(), 4),
            "cv_accuracy_std": round(cv_scores.std(), 4),
            **_test_metrics(y_test, pipe.predict(X_test)),
        })
        LOGGER.info("%s: CV acc %.4f (+/- %.4f)", name, cv_scores.mean(), cv_scores.std())

    metrics_df = pd.DataFrame(rows)

    # Best trained model = highest CV accuracy among the real models (not baseline).
    trained = metrics_df[metrics_df["model"] != "baseline_majority"]
    best_name = trained.sort_values("cv_accuracy_mean", ascending=False).iloc[0]["model"]
    best = fitted[best_name]

    # Held-out predictions of the winner, for the confusion matrix.
    preds_df = pd.DataFrame({"y_true": y_test.values, "y_pred": best.predict(X_test)})

    importance_df = _feature_importance(best, features)

    return metrics_df, best_name, best, preds_df, importance_df


def _feature_importance(pipe, features):
    """Signed coefficient (logistic regression) or Gini importance (random forest)."""
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
        kind = "gini_importance"
    elif hasattr(clf, "coef_"):
        values = clf.coef_[0]
        kind = "coefficient"
    else:
        values = [float("nan")] * len(features)
        kind = "unavailable"
    return pd.DataFrame({"feature": features, "importance": values, "kind": kind})


def main(input_csv, metrics_path, model_path, preds_path, importance_path,
         features, model_names, test_size, cv_folds, seed):
    setup_logger()
    LOGGER.info("Loading %s", input_csv)
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        raise SystemExit(f"ERROR: input file not found: {input_csv}")
    except pd.errors.EmptyDataError:
        raise SystemExit(f"ERROR: input file is empty: {input_csv}")

    if TARGET_COL not in df.columns:
        raise SystemExit(
            f"ERROR: input missing '{TARGET_COL}' column (run datapreprocessing_w9 first).")

    metrics_df, best_name, best_model, preds_df, importance_df = evaluate_models(
        df, features, model_names, test_size, cv_folds, seed)

    for path in (metrics_path, model_path, preds_path, importance_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(metrics_path, index=False)
    joblib.dump(best_model, model_path)
    preds_df.to_csv(preds_path, index=False)
    importance_df.to_csv(importance_path, index=False)

    LOGGER.info("Best model: %s -> %s", best_name, model_path)
    print("\n=== RQ9 model comparison (gloomy vs clear from audio profile) ===")
    print(metrics_df.to_string(index=False))
    best_acc = metrics_df.loc[metrics_df["model"] == best_name, "test_accuracy"].iloc[0]
    base_acc = metrics_df.loc[metrics_df["model"] == "baseline_majority", "test_accuracy"].iloc[0]
    print(f"\nBest model: {best_name} (held-out accuracy {best_acc}) "
          f"vs majority baseline {base_acc}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ9: classify gloomy vs clear days from the daily audio profile")
    parser.add_argument("input_csv", help="two-class labelled CSV from datapreprocessing_w9.py")
    parser.add_argument("-o", "--metrics", required=True, help="output metrics CSV")
    parser.add_argument("-m", "--model", required=True, help="output best-model .pkl")
    parser.add_argument("--predictions", required=True, help="output held-out predictions CSV")
    parser.add_argument("--importance", required=True, help="output feature-importance CSV")
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES),
                        help="comma-separated audio feature columns to use as predictors")
    parser.add_argument("--models", default="logistic_regression,random_forest",
                        help="comma-separated model names to train and compare")
    parser.add_argument("--test-size", type=float, default=0.25,
                        help="fraction of days held out for testing (default 0.25)")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="stratified cross-validation folds (default 5)")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="random seed for split and models (default 42)")
    args = parser.parse_args()
    feats = [f.strip() for f in args.features.split(",") if f.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    main(args.input_csv, args.metrics, args.model, args.predictions, args.importance,
         feats, models, args.test_size, args.cv_folds, args.random_seed)
