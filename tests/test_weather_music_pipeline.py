#!/usr/bin/env python3
"""
test_weather_music_pipeline.py
-------------------------------
Unit tests for the RQ7 weather -> mood -> music pipeline.

Covers:
    - datapreprocessing_w7.add_calendar_features
    - train_model.validate_columns
    - train_model.load_current_weather
    - train_model.train_weather_models / compare_models
    - train_model.build_song_index / recommend_songs

Place this file in the same directory as `train_model.py` and
`datapreprocessing_w7.py` (i.e. the workflow_7 folder), then run:

    pytest test_weather_music_pipeline.py -v

No CSVs from `data/` or `results/` are needed -- all inputs are
synthetic and generated inside the tests.
"""

import numpy as np
import pandas as pd
import pytest

import train_model as tm
import recommend_song as rec
import datapreprocessing_w7 as dp


# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture
def weather_features():
    return ["precipitation_mm", "sunshine_hours", "temperature_mean_c",
            "is_weekend", "day_of_week", "month", "is_holiday"]


@pytest.fixture
def mood_targets():
    return ["avg_valence", "avg_energy", "avg_danceability", "avg_tempo"]


@pytest.fixture
def weather_music_df(weather_features, mood_targets):
    """Synthetic daily weather+mood dataset, big enough for an 80/20 split."""
    rng = np.random.default_rng(42)
    n = 40
    df = pd.DataFrame({
        "precipitation_mm": rng.uniform(0, 20, n),
        "sunshine_hours": rng.uniform(0, 12, n),
        "temperature_mean_c": rng.uniform(-5, 30, n),
        "is_weekend": rng.integers(0, 2, n),
        "day_of_week": rng.integers(0, 7, n),
        "month": rng.integers(1, 13, n),
        "is_holiday": rng.integers(0, 2, n),
        "avg_valence": rng.uniform(0, 1, n),
        "avg_energy": rng.uniform(0, 1, n),
        "avg_danceability": rng.uniform(0, 1, n),
        "avg_tempo": rng.uniform(60, 180, n),
    })
    return df


@pytest.fixture
def songs_df():
    """Synthetic song catalog for the recommender.

    NOTE: `recommend_songs` hardcodes `n_neighbors=50` when it calls
    `knn.kneighbors(...)`. That raises a ValueError if the catalog has
    fewer than 50 rows (see test_recommend_songs_catalog_smaller_than_50
    below), so the catalog here is kept >= 50 rows to exercise the
    "normal" path.
    """
    rng = np.random.default_rng(7)
    n = 60
    df = pd.DataFrame({
        "track_name": [f"Song {i}" for i in range(n)],
        "artist_name": [f"Artist {i % 5}" for i in range(n)],
        "popularity": rng.integers(0, 100, n),
        "valence": rng.uniform(0, 1, n),
        "energy": rng.uniform(0, 1, n),
        "danceability": rng.uniform(0, 1, n),
        "tempo": rng.uniform(60, 180, n),
    })
    return df


@pytest.fixture
def current_weather_yaml(tmp_path):
    content = (
        "precipitation: 8.0\n"
        "sunshine_hours: 0.0\n"
        "temperature_mean_c: 4.0\n"
        "is_weekend: 0\n"
        "day_of_week: 4\n"
        "month: 7\n"
        "is_holiday: 0\n"
    )
    path = tmp_path / "current_weather.yaml"
    path.write_text(content)
    return path


# =========================================================
# datapreprocessing_w7.add_calendar_features
# =========================================================

class TestAddCalendarFeatures:

    def test_adds_expected_columns(self):
        df = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"]})
        out = dp.add_calendar_features(df, date_col="date")
        for col in ["day_of_week", "is_weekend", "month", "is_holiday"]:
            assert col in out.columns

    def test_weekend_flagging(self):
        # 2024-01-06 = Saturday, 2024-01-08 = Monday
        df = pd.DataFrame({"date": ["2024-01-06", "2024-01-08"]})
        out = dp.add_calendar_features(df, date_col="date")
        assert out.loc[out["date"] == "2024-01-06", "is_weekend"].iloc[0] == 1
        assert out.loc[out["date"] == "2024-01-08", "is_weekend"].iloc[0] == 0

    def test_known_german_holiday_is_flagged(self):
        # 2024-01-01 (New Year's Day) is a public holiday in all German states
        df = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"]})
        out = dp.add_calendar_features(df, date_col="date")
        assert out.loc[out["date"] == "2024-01-01", "is_holiday"].iloc[0] == 1
        assert out.loc[out["date"] == "2024-01-02", "is_holiday"].iloc[0] == 0

    def test_month_extraction(self):
        df = pd.DataFrame({"date": ["2024-03-15"]})
        out = dp.add_calendar_features(df, date_col="date")
        assert out["month"].iloc[0] == 3

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"date": ["2024-01-01"]})
        dp.add_calendar_features(df, date_col="date")
        # original df should still lack the derived columns
        assert "is_holiday" not in df.columns


# =========================================================
# train_model.validate_columns
# =========================================================

class TestValidateColumns:

    def test_passes_when_all_columns_present(self, weather_music_df, weather_features, mood_targets):
        # should not raise
        tm.validate_columns(weather_music_df, weather_features + mood_targets)

    def test_raises_systemexit_on_missing_columns(self, weather_music_df):
        with pytest.raises(SystemExit):
            tm.validate_columns(weather_music_df, ["nonexistent_column"])


# =========================================================
# train_model.load_current_weather
# =========================================================

class TestLoadCurrentWeather:

    def test_loads_all_required_keys_as_floats(self, current_weather_yaml):
        weather = rec.load_current_weather(current_weather_yaml)
        expected_keys = {"precipitation", "sunshine_hours", "temperature_mean_c",
                         "is_weekend", "day_of_week", "month", "is_holiday"}
        assert set(weather.keys()) == expected_keys
        assert all(isinstance(v, float) for v in weather.values())
        assert weather["temperature_mean_c"] == 4.0

    def test_raises_systemexit_on_missing_key(self, tmp_path):
        path = tmp_path / "bad_weather.yaml"
        path.write_text("precipitation: 1.0\nsunshine_hours: 2.0\n")
        with pytest.raises(SystemExit):
            rec.load_current_weather(path)


# =========================================================
# train_model.train_weather_models / compare_models
# =========================================================

class TestTrainWeatherModels:

    def test_unknown_model_name_raises(self, weather_music_df, weather_features, mood_targets):
        with pytest.raises(SystemExit):
            tm.train_weather_models(weather_music_df, weather_features, mood_targets,
                                    model_name="not_a_real_model")

    def test_returns_one_model_per_target(self, weather_music_df, weather_features, mood_targets):
        models, metrics, importances = tm.train_weather_models(
            weather_music_df, weather_features, mood_targets, model_name="ridge"
        )
        assert set(models.keys()) == set(mood_targets)
        assert set(metrics.keys()) == set(mood_targets)
        for target in mood_targets:
            assert "r2" in metrics[target]
            assert "mae" in metrics[target]

    def test_tree_model_has_feature_importances(self, weather_music_df, weather_features, mood_targets):
        _, _, importances = tm.train_weather_models(
            weather_music_df, weather_features, mood_targets,
            model_name="random_forest_depth5"
        )
        for target in mood_targets:
            assert set(importances[target].keys()) == set(weather_features)

    def test_linear_model_uses_coefficients(self, weather_music_df, weather_features, mood_targets):
        _, _, importances = tm.train_weather_models(
            weather_music_df, weather_features, mood_targets, model_name="ridge"
        )
        for target in mood_targets:
            assert set(importances[target].keys()) == set(weather_features)


class TestCompareModels:

    def test_comparison_dataframe_shape(self, weather_music_df, weather_features, mood_targets):
        model_names = ["ridge", "random_forest_depth5"]
        comparison_df, all_results = tm.compare_models(
            weather_music_df, weather_features, mood_targets, model_names
        )
        assert set(comparison_df["model"].unique()) == set(model_names)
        assert set(comparison_df["target"].unique()) == set(mood_targets)
        assert len(comparison_df) == len(model_names) * len(mood_targets)
        assert set(all_results.keys()) == set(model_names)


# =========================================================
# train_model.build_song_index / recommend_songs
# =========================================================

class TestRecommendation:

    def test_build_song_index_scales_and_fits(self, songs_df):
        song_features = ["valence", "energy", "danceability", "tempo"]
        songs_indexed, scaler, knn = rec.build_song_index(songs_df, song_features)
        assert len(songs_indexed) == len(songs_df)
        assert list(scaler.feature_names_in_) == song_features

    def test_recommend_songs_returns_k_unique_rows(self, songs_df):
        song_features = ["valence", "energy", "danceability", "tempo"]
        songs_indexed, scaler, knn = rec.build_song_index(songs_df, song_features)
        target_vector = [0.5, 0.5, 0.5, 120.0]
        recs = rec.recommend_songs(songs_indexed, scaler, knn, target_vector, k=5)
        assert len(recs) == 5
        # distances should be sorted ascending
        assert list(recs["distance"]) == sorted(recs["distance"])
        # no duplicate (track_name, artist_name) pairs
        assert not recs.duplicated(subset=["track_name", "artist_name"]).any()

    def test_recommend_songs_catalog_smaller_than_50(self, songs_df):
        """KNOWN BUG: `recommend_songs` hardcodes `n_neighbors=50` in its
        call to `knn.kneighbors(...)`. Any catalog with fewer than 50
        rows raises a ValueError instead of gracefully returning what's
        available. This test documents the current (buggy) behavior so
        it starts failing loudly -- in a good way -- once it's fixed.
        If you fix the code to clamp n_neighbors to len(songs_df), update
        this test to assert on the graceful behavior instead.
        """
        song_features = ["valence", "energy", "danceability", "tempo"]
        small_songs = songs_df.head(3)
        songs_indexed, scaler, knn = rec.build_song_index(small_songs, song_features)
        target_vector = [0.5, 0.5, 0.5, 120.0]
        with pytest.raises(ValueError, match="n_neighbors"):
            rec.recommend_songs(songs_indexed, scaler, knn, target_vector, k=10)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
