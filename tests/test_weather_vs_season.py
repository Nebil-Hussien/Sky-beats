"""
Tests for RQ8: datapreprocessing_w8.py and compare_models_w8.py.

Run with:  pytest tests/test_weather_vs_season.py
"""

import numpy as np
import pandas as pd
import pytest

# from src/ (see tests/conftest.py)
import datapreprocessing_w8 as preprocessing
import compare_models_w8 as compare


# ── preprocessing ────────────────────────────────────────────────────────────

def test_build_daily_features_adds_month_and_season():
    df = pd.DataFrame({
        "date": ["2020-01-15", "2020-01-15", "2020-07-15"],
        "valence": [0.2, 0.4, 0.8],
        "sunshine_hours": [1.0, 1.0, 10.0],
        "precipitation_mm": [5.0, 5.0, 0.0],
        "temperature_mean_c": [2.0, 2.0, 25.0],
    })
    daily = preprocessing.build_daily_features(df)

    assert len(daily) == 2
    assert list(daily["month"]) == [1, 7]
    assert list(daily["season"]) == ["winter", "summer"]
    assert daily["valence"].iloc[0] == pytest.approx(0.3)
    # sorted by date
    assert daily["date"].is_monotonic_increasing


# ── model comparison ─────────────────────────────────────────────────────────

def _synthetic_daily(weather_effect, n_days=730, seed=0):
    """Daily data where valence = month effect + weather_effect * sunshine + noise.

    With weather_effect > 0 the weather model has real extra signal;
    with weather_effect == 0 valence is purely seasonal.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n_days, freq="D")
    month = dates.month.to_numpy()
    sunshine = rng.uniform(0, 12, n_days)
    valence = (
        0.4 + 0.02 * np.sin(2 * np.pi * (month - 1) / 12)  # seasonal cycle
        + weather_effect * sunshine
        + rng.normal(0, 0.005, n_days)
    )
    return pd.DataFrame({
        "date": dates,
        "month": month,
        "valence": valence,
        "sunshine_hours": sunshine,
    })


def test_weather_model_wins_when_weather_drives_valence():
    df = _synthetic_daily(weather_effect=0.02)
    result = compare.compare_models(
        df, "valence", ["sunshine_hours"], ["ridge"],
        test_fraction=0.2, seed=42).set_index("feature_set")

    assert result.loc["weather", "r2"] > result.loc["baseline", "r2"] + 0.1
    assert result.loc["weather", "mae"] < result.loc["baseline", "mae"]


def test_no_weather_gain_when_valence_is_purely_seasonal():
    df = _synthetic_daily(weather_effect=0.0)
    result = compare.compare_models(
        df, "valence", ["sunshine_hours"], ["ridge"],
        test_fraction=0.2, seed=42).set_index("feature_set")

    # weather features carry no signal -> no meaningful R2 improvement
    delta = result.loc["weather", "r2"] - result.loc["baseline", "r2"]
    assert delta < 0.05


def test_compare_models_outputs_one_row_per_model_and_set():
    df = _synthetic_daily(weather_effect=0.01)
    result = compare.compare_models(
        df, "valence", ["sunshine_hours"], ["ridge", "random_forest"],
        test_fraction=0.2, seed=42)
    assert len(result) == 4
    assert set(zip(result["model"], result["feature_set"])) == {
        ("ridge", "baseline"), ("ridge", "weather"),
        ("random_forest", "baseline"), ("random_forest", "weather"),
    }


def test_chronological_split_holds_out_the_end():
    train_idx, test_idx = compare.chronological_split(10, 0.2)
    assert list(train_idx) == list(range(8))
    assert list(test_idx) == [8, 9]


def test_chronological_split_rejects_bad_fraction():
    with pytest.raises(ValueError):
        compare.chronological_split(10, 1.5)


def test_compare_models_rejects_unknown_model():
    df = _synthetic_daily(weather_effect=0.01)
    with pytest.raises(ValueError):
        compare.compare_models(df, "valence", ["sunshine_hours"],
                               ["not_a_model"], 0.2, 42)


def test_main_missing_input_file_exits(tmp_path):
    with pytest.raises(SystemExit):
        preprocessing.main(str(tmp_path / "nope.csv"), str(tmp_path / "out.csv"))


def test_main_missing_columns_exits(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2020-01-01"], "valence": [0.5]}).to_csv(bad, index=False)
    with pytest.raises(SystemExit):
        preprocessing.main(str(bad), str(tmp_path / "out.csv"))
