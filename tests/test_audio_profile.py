"""
Tests for RQ6 (Workflow 6):
  - utils.label_gloomy_clear   (percentile-based gloomy/clear/other labelling)
  - audio_profile.compare_profiles  (per-feature means + Mann-Whitney U)

Run with:  pytest src/tests/test_audio_profile.py
"""

import pandas as pd
import pytest

# from src/ (see tests/conftest.py)
import utils
import audio_profile


# ── label_gloomy_clear ─────────────────────────────────────────────────────────

def _daily():
    """8 days ordered darkest+wettest -> sunniest+driest."""
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=8),
        "sunshine_hours": [0, 1, 2, 3, 6, 8, 10, 12],
        "precipitation_mm": [10, 8, 6, 4, 2, 1, 0.5, 0],
    })


def test_label_extremes_are_gloomy_and_clear():
    """The darkest+wettest day is gloomy; the sunniest+driest day is clear."""
    labeled = utils.label_gloomy_clear(_daily())  # default 25/75 percentiles
    assert labeled.iloc[0]["day_type"] == "gloomy"
    assert labeled.iloc[-1]["day_type"] == "clear"
    # both groups present, and 'other' is used for the middle
    assert {"gloomy", "clear", "other"} <= set(labeled["day_type"])


def test_label_requires_both_conditions():
    """A day that is dark but also dry must NOT be gloomy (needs high precip too)."""
    df = pd.DataFrame({
        "sunshine_hours": [0, 0, 12, 12],
        "precipitation_mm": [0, 0, 10, 10],  # dark days are dry; sunny days are wet
    })
    labeled = utils.label_gloomy_clear(df)
    assert (labeled["day_type"] == "other").all()


# ── compare_profiles ───────────────────────────────────────────────────────────

def test_compare_profiles_means_and_diff():
    """Means and gloomy-minus-clear diff are computed correctly per feature."""
    df = pd.DataFrame({
        "day_type": ["gloomy", "gloomy", "clear", "clear"],
        "valence": [0.2, 0.4, 0.6, 0.8],   # gloomy mean 0.3, clear mean 0.7
        "energy": [0.5, 0.5, 0.5, 0.5],
    })
    res = audio_profile.compare_profiles(df, ["valence", "energy"]).set_index("feature")
    assert res.loc["valence", "gloomy_mean"] == pytest.approx(0.3)
    assert res.loc["valence", "clear_mean"] == pytest.approx(0.7)
    assert res.loc["valence", "diff_gloomy_minus_clear"] == pytest.approx(-0.4)
    assert res.loc["energy", "diff_gloomy_minus_clear"] == pytest.approx(0.0)


def test_compare_profiles_raises_without_both_groups():
    """Missing one of the two groups should fail loudly."""
    df = pd.DataFrame({"day_type": ["gloomy", "gloomy"], "valence": [0.2, 0.4]})
    with pytest.raises(SystemExit):
        audio_profile.compare_profiles(df, ["valence"])


def test_compare_profiles_raises_on_missing_feature():
    """Requesting a feature column that isn't present should fail loudly."""
    df = pd.DataFrame({
        "day_type": ["gloomy", "clear"],
        "valence": [0.2, 0.8],
    })
    with pytest.raises(SystemExit):
        audio_profile.compare_profiles(df, ["nonexistent"])
