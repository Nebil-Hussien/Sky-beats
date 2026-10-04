"""
Tests for RQ9 (Workflow 9):
  - datapreprocessing_w9.build_classification_table  (label + keep gloomy/clear only)
  - train_classifier.evaluate_models                 (model comparison vs baseline)

Run with:  pytest src/tests/test_classifier.py
"""

import pandas as pd
import pytest

# from src/ (see tests/conftest.py)
import datapreprocessing_w9 as prep
import train_classifier as train


# ── datapreprocessing_w9.build_classification_table ─────────────────────────────

def _daily():
    """8 days ordered darkest+wettest -> sunniest+driest (as in the RQ6 tests)."""
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=8),
        "sunshine_hours": [0, 1, 2, 3, 6, 8, 10, 12],
        "precipitation_mm": [10, 8, 6, 4, 2, 1, 0.5, 0],
    })


def test_table_keeps_only_gloomy_and_clear():
    """Output contains only the two target classes; 'other' days are dropped and counted."""
    table, n_dropped = prep.build_classification_table(
        _daily(), sun_low_pct=25, sun_high_pct=75, precip_low_pct=25, precip_high_pct=75)
    assert set(table["day_type"]) <= {"gloomy", "clear"}
    assert {"gloomy", "clear"} <= set(table["day_type"])       # both present
    assert n_dropped == 8 - len(table)                         # dropped == the 'other' middle
    assert n_dropped > 0


def test_table_percentiles_use_full_distribution():
    """Filtering must not change the cut-points: the darkest+wettest day stays gloomy,
    the sunniest+driest day stays clear."""
    table, _ = prep.build_classification_table(
        _daily(), sun_low_pct=25, sun_high_pct=75, precip_low_pct=25, precip_high_pct=75)
    assert table.iloc[0]["day_type"] == "gloomy"   # first kept row is the extreme gloomy day
    assert table.iloc[-1]["day_type"] == "clear"   # last kept row is the extreme clear day


# ── train_classifier.evaluate_models ────────────────────────────────────────────

def _separable(n=40):
    """Cleanly separable two-class data: gloomy days have low energy/valence,
    clear days have high — so a good classifier should easily beat the baseline."""
    rows = []
    for i in range(n):
        rows.append({"day_type": "gloomy", "energy": 0.20 + 0.002 * i,
                     "valence": 0.25 + 0.002 * i, "tempo": 100 + 0.1 * i})
        rows.append({"day_type": "clear", "energy": 0.80 - 0.002 * i,
                     "valence": 0.75 - 0.002 * i, "tempo": 130 + 0.1 * i})
    return pd.DataFrame(rows)


FEATURES = ["energy", "valence", "tempo"]
MODELS = ["logistic_regression", "random_forest"]


def test_models_beat_baseline_on_separable_data():
    """On separable data every trained model should out-accuracy the majority baseline."""
    metrics, best_name, model, preds, importance = train.evaluate_models(
        _separable(), FEATURES, MODELS, test_size=0.25, cv_folds=3, seed=42)

    assert "baseline_majority" in set(metrics["model"])
    assert set(MODELS) <= set(metrics["model"])
    assert best_name in MODELS

    baseline_acc = metrics.loc[metrics["model"] == "baseline_majority", "test_accuracy"].iloc[0]
    for name in MODELS:
        model_acc = metrics.loc[metrics["model"] == name, "test_accuracy"].iloc[0]
        assert model_acc > baseline_acc

    # Returned side-artifacts have the expected shape for the plot step.
    assert list(preds.columns) == ["y_true", "y_pred"]
    assert set(importance.columns) >= {"feature", "importance"}
    assert len(importance) == len(FEATURES)


def test_evaluate_raises_on_missing_feature():
    """Requesting a predictor column that isn't present should fail loudly."""
    with pytest.raises(SystemExit):
        train.evaluate_models(_separable(), ["energy", "nonexistent"], MODELS,
                              test_size=0.25, cv_folds=3, seed=42)


def test_evaluate_raises_without_both_classes():
    """Only one class present -> cannot train a gloomy-vs-clear classifier."""
    one_class = _separable()
    one_class = one_class[one_class["day_type"] == "gloomy"]
    with pytest.raises(SystemExit):
        train.evaluate_models(one_class, FEATURES, MODELS,
                              test_size=0.25, cv_folds=3, seed=42)
