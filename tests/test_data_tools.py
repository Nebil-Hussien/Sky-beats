"""Tests for the acquisition and integration command-line components."""


import pandas as pd
import pytest
import yaml

import load_spotify
import load_weather
import merge_spotify
import merge_weather_spotify


def write_config(tmp_path):
    config = {
        "dates": {"start": "2020-01-01", "end": "2020-01-02"},
        "paths": {"raw_dir": "raw", "output_dir": "data"},
        "dwd": {
            "base_url": "https://example.invalid/",
            "columns": ["MESS_DATUM", "RSK", "SDK", "TMK"],
            "output_file": "weather.csv",
        },
        "spotify": {
            "region": "Germany",
            "charts_file": "charts.csv",
            "features_file": "features.csv",
            "output_file": "spotify.csv",
            "audio_features": ["track_id", "track_genre", "danceability",
                               "energy", "tempo", "valence", "acousticness",
                               "popularity"],
        },
        "merge": {"output_merged": "songs.csv", "output_daily": "daily.csv"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_merge_spotify_happy_path(tmp_path):
    config = write_config(tmp_path)
    charts = tmp_path / "charts.csv"
    features = tmp_path / "features.csv"
    output = tmp_path / "merged.csv"
    pd.DataFrame({
        "date": ["2020-01-01"], "title": ["Song"], "artist": ["Artist"],
        "rank": [1], "streams": [100], "trend": ["SAME"],
        "url": ["https://open.spotify.com/track/abc"], "region": ["Germany"],
    }).to_csv(charts, index=False)
    pd.DataFrame({
        "track_id": ["abc"], "track_genre": ["pop"], "danceability": [0.5],
        "energy": [0.6], "tempo": [120], "valence": [0.7],
        "acousticness": [0.1], "popularity": [80],
    }).to_csv(features, index=False)

    merge_spotify.main(config, charts, features, output)

    result = pd.read_csv(output)
    assert result.loc[0, "track_id"] == "abc"
    assert result.loc[0, "energy"] == pytest.approx(0.6)


def test_merge_spotify_rejects_bad_schema(tmp_path):
    config = write_config(tmp_path)
    charts = tmp_path / "charts.csv"
    features = tmp_path / "features.csv"
    pd.DataFrame({"date": ["2020-01-01"]}).to_csv(charts, index=False)
    pd.DataFrame({"track_id": ["abc"]}).to_csv(features, index=False)
    with pytest.raises(ValueError, match="charts input is missing columns"):
        merge_spotify.main(config, charts, features, tmp_path / "out.csv")


def test_merge_weather_spotify_outputs_both_grains(tmp_path):
    config = write_config(tmp_path)
    spotify = tmp_path / "spotify.csv"
    weather = tmp_path / "weather.csv"
    song_output = tmp_path / "songs.csv"
    daily_output = tmp_path / "daily.csv"
    pd.DataFrame({
        "date": ["2020-01-01", "2020-01-01"], "track_id": ["a", "b"],
        "streams": [10, 20], "valence": [0.2, 0.6], "energy": [0.3, 0.7],
        "danceability": [0.4, 0.8], "tempo": [100, 120],
        "acousticness": [0.1, 0.3],
    }).to_csv(spotify, index=False)
    pd.DataFrame({
        "date": ["2020-01-01"], "precipitation_mm": [1.0],
        "sunshine_hours": [2.0], "temperature_mean_c": [10.0],
    }).to_csv(weather, index=False)

    merge_weather_spotify.main(config, spotify, weather, song_output, daily_output)

    assert len(pd.read_csv(song_output)) == 2
    daily = pd.read_csv(daily_output)
    assert daily.loc[0, "avg_valence"] == pytest.approx(0.4)
    assert daily.loc[0, "total_streams"] == 30


def test_merge_weather_rejects_bad_schema(tmp_path):
    config = write_config(tmp_path)
    spotify = tmp_path / "spotify.csv"
    weather = tmp_path / "weather.csv"
    pd.DataFrame({"date": ["2020-01-01"]}).to_csv(spotify, index=False)
    pd.DataFrame({"date": ["2020-01-01"]}).to_csv(weather, index=False)
    with pytest.raises(ValueError, match="Spotify input is missing columns"):
        merge_weather_spotify.main(
            config, spotify, weather, tmp_path / "songs.csv", tmp_path / "daily.csv"
        )


def test_download_decision_respects_existing_files(tmp_path):
    target = tmp_path / "download.csv"
    assert load_spotify.should_download_file(target)
    target.write_text("data", encoding="utf-8")
    assert not load_spotify.should_download_file(target)
    assert load_spotify.should_download_file(target, force=True)


def test_load_weather_fails_when_station_listing_is_empty(tmp_path, monkeypatch):
    config = write_config(tmp_path)

    class Response:
        text = "<html>No station archives</html>"

        @staticmethod
        def raise_for_status():
            return None

    monkeypatch.setattr(load_weather.requests, "get", lambda *args, **kwargs: Response())
    with pytest.raises(SystemExit, match="no station data downloaded"):
        load_weather.main(config, tmp_path / "weather.csv")


def test_all_data_tools_expose_help_without_running():
    for module in (load_spotify, load_weather, merge_spotify, merge_weather_spotify):
        assert callable(module.parse_args)
        assert callable(module.main)
