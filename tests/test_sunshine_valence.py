"""
Tests for RQ1: datapreprocessing_w1.py and sunshine_valence_correlation.py.

Run with:  pytest tests/test_sunshine_valence.py
"""

import pandas as pd
import pytest

# from src/ (see tests/conftest.py)
import datapreprocessing_w1 as preprocessing
import sunshine_valence_correlation as correlation


def _song_level_df():
    """Two songs per day across two days in different seasons."""
    rows = [
        # date,        valence, sunshine_hours, precipitation_mm, temperature_mean_c
        ("2020-01-15", 0.2, 1.0, 5.0, 2.0),
        ("2020-01-15", 0.4, 1.0, 5.0, 2.0),
        ("2020-07-15", 0.6, 10.0, 0.0, 25.0),
        ("2020-07-15", 0.8, 10.0, 0.0, 25.0),
    ]
    return pd.DataFrame(
        rows, columns=["date", "valence", "sunshine_hours",
                       "precipitation_mm", "temperature_mean_c"])


def test_build_daily_profile_aggregates_and_labels_seasons():
    daily = preprocessing.build_daily_profile(_song_level_df())

    assert len(daily) == 2  # one row per date
    daily = daily.sort_values("date").reset_index(drop=True)
    assert daily.loc[0, "season"] == "winter"
    assert daily.loc[1, "season"] == "summer"
    # valence is averaged per day, weather taken as-is
    assert daily.loc[0, "valence"] == pytest.approx(0.3)
    assert daily.loc[1, "valence"] == pytest.approx(0.7)
    assert daily.loc[0, "sunshine_hours"] == pytest.approx(1.0)


def test_season_mapping_covers_all_months():
    assert set(preprocessing.SEASONS) == set(range(1, 13))
    assert set(preprocessing.SEASONS.values()) == {"winter", "spring", "summer", "autumn"}


def _daily_df():
    """Daily rows where valence increases perfectly with sunshine in summer
    and decreases perfectly in winter -> Spearman rho +1 / -1."""
    rows = []
    for i in range(10):
        rows.append((f"2020-07-{i + 1:02d}", "summer", float(i), 0.1 * i))
        rows.append((f"2020-01-{i + 1:02d}", "winter", float(i), 1.0 - 0.1 * i))
    return pd.DataFrame(rows, columns=["date", "season", "sunshine_hours", "valence"])


def test_correlate_by_season_perfect_monotonic():
    result = correlation.correlate_by_season(
        _daily_df(), "sunshine_hours", "valence").set_index("group")

    assert result.loc["summer", "rho"] == pytest.approx(1.0)
    assert result.loc["winter", "rho"] == pytest.approx(-1.0)
    assert bool(result.loc["summer", "significant"]) is True
    assert result.loc["summer", "n_days"] == 10
    # opposite seasonal trends cancel out overall
    assert abs(result.loc["all", "rho"]) < 0.5


def test_correlate_by_season_returns_all_groups():
    result = correlation.correlate_by_season(_daily_df(), "sunshine_hours", "valence")
    assert list(result["group"]) == ["all", "winter", "spring", "summer", "autumn"]


def test_correlate_by_season_too_few_rows_yields_nan():
    df = _daily_df().head(2)  # only 2 summer/winter rows in total
    result = correlation.correlate_by_season(
        df, "sunshine_hours", "valence").set_index("group")
    assert pd.isna(result.loc["spring", "rho"])
    assert bool(result.loc["spring", "significant"]) is False


def test_main_missing_input_file_exits(tmp_path):
    with pytest.raises(SystemExit):
        preprocessing.main(str(tmp_path / "nope.csv"), str(tmp_path / "out.csv"))


def test_main_missing_columns_exits(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2020-01-01"], "valence": [0.5]}).to_csv(bad, index=False)
    with pytest.raises(SystemExit):
        preprocessing.main(str(bad), str(tmp_path / "out.csv"))
