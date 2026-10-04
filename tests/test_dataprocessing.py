import pandas as pd
import pytest

import dataprocessing  # from src/ (see tests/conftest.py)


def test_preprocess_dataframe_creates_time_and_weather_features():
    df = pd.DataFrame(
        {
            "date": ["2017-01-01", "2017-01-02"],
            "title": ["Song A", "Song B"],
            "artist": ["Artist A", "Artist B"],
            "streams": [10, None],
            "energy": [0.5, None],
            "danceability": [0.8, 0.9],
            "tempo": [120.0, 121.0],
            "valence": [0.6, 0.7],
            "acousticness": [0.2, 0.3],
            "popularity": [80, 85],
            "temperature_mean_c": [1.5, -2.0],
            "precipitation_mm": [0.0, 3.5],
            "sunshine_hours": [3.0, 1.0],
            "track_id": ["x", "y"],
        }
    )

    result = dataprocessing.preprocess_dataframe(df)

    assert pd.api.types.is_datetime64_any_dtype(result["date"])
    assert result.loc[0, "year"] == 2017
    assert result.loc[0, "month"] == 1
    assert "season" in result.columns
    assert "is_weekend" in result.columns
    assert result["streams"].isna().sum() == 0
    assert result["energy"].isna().sum() == 0


def test_process_csvfile_raises_for_missing_file(tmp_path):
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        dataprocessing.process_csvfile(str(missing_path))


def test_process_csvfile_raises_for_empty_file(tmp_path):
    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        dataprocessing.process_csvfile(str(empty_path))


def test_preprocess_dataframe_raises_for_missing_columns():
    df = pd.DataFrame({"date": ["2017-01-01"], "title": ["Song A"]})

    with pytest.raises(ValueError, match="missing required columns"):
        dataprocessing.preprocess_dataframe(df)


def test_preprocess_dataframe_raises_for_invalid_dates():
    df = pd.DataFrame(
        {
            "date": ["2017-01-01", "not-a-date"],
            "streams": [10, 20],
            "track_id": ["x", "y"],
            "danceability": [0.8, 0.9],
            "energy": [0.5, 0.6],
            "tempo": [120.0, 121.0],
            "valence": [0.6, 0.7],
            "acousticness": [0.2, 0.3],
            "popularity": [80, 85],
            "temperature_mean_c": [1.5, -2.0],
            "precipitation_mm": [0.0, 3.5],
            "sunshine_hours": [3.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="invalid date"):
        dataprocessing.preprocess_dataframe(df)


def test_preprocess_dataframe_raises_for_non_contiguous_dates():
    df = pd.DataFrame(
        {
            "date": ["2017-01-01", "2017-01-03"],
            "streams": [10, 20],
            "track_id": ["x", "y"],
            "danceability": [0.8, 0.9],
            "energy": [0.5, 0.6],
            "tempo": [120.0, 121.0],
            "valence": [0.6, 0.7],
            "acousticness": [0.2, 0.3],
            "popularity": [80, 85],
            "temperature_mean_c": [1.5, -2.0],
            "precipitation_mm": [0.0, 3.5],
            "sunshine_hours": [3.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="not contiguous"):
        dataprocessing.preprocess_dataframe(df)
