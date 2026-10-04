# Data

This directory holds all input data for the Sky Beats workflow. Most of its
contents are **git-ignored** (large raw downloads and regenerable intermediate
files); three small example input files are committed so the analysis
workflows can be run directly after cloning.

This is a deliberate repository-size and reproducibility decision: the raw
Spotify charts download is approximately 3 GB, whereas the committed inputs
contain the data needed to reproduce every example result. The acquisition and
preparation commands below rebuild those inputs from the original sources.

## Committed example inputs

| File | Description | Columns |
| ---- | ----------- | ------- |
| `spotify_weather_merged.csv` | Song-level German Top-200 chart entries (2017–2021) with audio features and the day's weather joined on `date`. **Main workflow input** — every `rqN` workflow starts from this file. | date, title, artist, rank, streams, trend, track_id, track_genre, danceability, energy, tempo, valence, acousticness, popularity, precipitation_mm, sunshine_hours, temperature_mean_c |
| `spotify_germany.csv` | Song-level German charts with audio features, *before* the weather join. | as above, without the three weather columns |
| `spotify_weather_daily_merged.csv` | One row per day: chart-wide averages of the audio features plus daily weather. | date, avg_valence, avg_energy, avg_danceability, avg_tempo, avg_acousticness, total_streams, unique_songs, precipitation_mm, sunshine_hours, temperature_mean_c |

Everything else that appears here at runtime (`data/raw/`,
`dwd_daily_germany.csv`, `spotify_weather_merged_processed*.csv`, …) is
downloaded or derived and stays untracked.

### Provenance and integrity of the committed snapshot

The original raw-download date was not recorded when the source datasets were
obtained. To avoid presenting an unverifiable date, the table below reports the
earliest verifiable date for this repository snapshot: **2026-07-15**, when the
three prepared example files were first committed. The files cover the analysis
period 2017-01-01 through 2021-12-31. A future raw-data rebuild should record the
actual retrieval date and source-dataset version at download time.

| File | Snapshot/preparation date | SHA-256 |
| ---- | ------------------------- | ------- |
| `spotify_germany.csv` | 2026-07-15 | `4c242bd059083fd5a43a5cf6d6ab0afae1b9a08a163803402634655823adbef9` |
| `spotify_weather_merged.csv` | 2026-07-15 | `997bd0710b8f51609f0aa9e267046b1fcb80d89e30ec95cb55bcbef5a98e330c` |
| `spotify_weather_daily_merged.csv` | 2026-07-15 | `39fe60f5903d113d842ae0e1d3f6263381030dbea96e650931229e5c9751e44e` |

Verify the committed files from the repository root with:

```bash
shasum -a 256 data/spotify_germany.csv \
  data/spotify_weather_merged.csv \
  data/spotify_weather_daily_merged.csv
```

## Sources and licenses

| Dataset | Source | License |
| ------- | ------ | ------- |
| Weather observations for Germany (daily precipitation, sunshine duration, mean temperature) | [DWD Climate Data Center (CDC)](https://opendata.dwd.de/climate_environment/CDC/) | [CC-BY-4.0](https://opendata.dwd.de/climate_environment/CDC/Terms_of_use.pdf) — © Deutscher Wetterdienst (DWD), used with attribution |
| Spotify Charts (daily Top-200 streams by country) | [Kaggle: dhruvildave/spotify-charts](https://www.kaggle.com/datasets/dhruvildave/spotify-charts) | [ODbL v1.0](https://opendatacommons.org/licenses/odbl/1-0/) |
| Spotify Tracks (audio features per track) | [Kaggle: maharshipandya/-spotify-tracks-dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) | [ODbL v1.0](https://opendatacommons.org/licenses/odbl/1-0/) |

The committed CSVs are **derivative databases** of the sources above:

- `spotify_germany.csv` derives from the two Spotify datasets and is therefore
  shared under **ODbL v1.0**.
- `spotify_weather_merged.csv` and `spotify_weather_daily_merged.csv`
  additionally incorporate DWD weather data and are shared under
  **ODbL v1.0** with attribution to the Deutscher Wetterdienst (DWD)
  as required by CC-BY-4.0.

## Rebuilding from raw data

The committed files can be regenerated from the original sources with the
scripts in `src/data/` (run from the project root, in this order):

```bash
# 1. Download DWD daily weather        -> data/dwd_daily_germany.csv
python src/data/load_weather.py

# 2. Download the two Spotify datasets -> data/raw/
#    (interactive: choose Kaggle or the team's public Box.UP share;
#     Box.UP needs no login, Kaggle needs an API token — see the
#     script docstring)
python src/data/load_spotify.py

# 3. Merge charts with audio features,
#    filter to Germany                 -> data/spotify_germany.csv
python src/data/merge_spotify.py

# 4. Join daily weather onto the songs -> data/spotify_weather_merged.csv
#                                         data/spotify_weather_daily_merged.csv
python src/data/merge_weather_spotify.py
```

Paths, date range and download options are configured in
`config/config.yaml`. The raw Spotify charts download is ~3 GB — the committed
example inputs exist precisely so you can skip this step and run
`snakemake --cores 1` right away.
