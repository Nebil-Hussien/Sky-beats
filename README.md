# Sky Beats

A data-analysis workflow investigating the relationship between **weather/climate** and
the **mood and characteristics of music people listen to**.

## Project idea

Does the weather influence what and how people stream music? *Sky Beats* combines German weather observations with Spotify listening and audio-feature data to
explore questions such as:

1. How does the daily duration of sunshine (hours) correlate with the average 'Valence' of the German Spotify Top 200, and does this relationship vary by season?
2. In what ways does high daily precipitation influence the preference for 'Acoustic' vs. 'High-Energy' tracks in the German charts?
3. Which music genres become more or less popular on the rainiest months in Germany compared to the driest months?
4. To what extent do extreme temperature deviations (very hot or very cold days) impact the average 'Danceability' and 'Tempo' of top-streamed songs?
5. How accurately can a combination of daily temperature and sunshine duration predict the overall 'Energy' level of the German Top 100?
6. What is the characteristic 'Audio Profile' (a composite of all audio features) of the Spotify Charts on 'Gloomy Days' (defined as 0 sunshine hours + >2mm rain) compared to 'Clear Days'?
7. Can daily weather attributes (e.g. precipitation, sunshine duration, temperature) predict whether Germany streams predominantly sad or happy music on a given day, and which songs would be recommended based on current weather conditions?
8. Does daily weather genuinely predict music mood *beyond* seasonal effects? (We compare a regression using weather features against a season/month-only baseline to test whether weather adds real predictive power, rather than just tracking the seasons.)
9. Can the day's weather be inferred from the music alone? (We train a classifier to predict whether a day is 'Gloomy' or 'Clear' from its aggregated audio profile — a reverse-prediction of weather from listening behaviour.)


## Datasets

| Dataset | Source | License |
| ------- | ------ | ------- |
| Weather/climate observations (Germany) | [DWD Climate Data Center (CDC)](https://opendata.dwd.de/climate_environment/CDC/) | [CC-BY-4.0](https://opendata.dwd.de/climate_environment/CDC/Terms_of_use.pdf) |
| Spotify Charts (daily streams by country) | [Kaggle: dhruvildave/spotify-charts](https://www.kaggle.com/datasets/dhruvildave/spotify-charts) | [ODbL v1.0](https://opendatacommons.org/licenses/odbl/1-0/) |
| Spotify Tracks (audio features) | [Kaggle: maharshipandya/-spotify-tracks-dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) | [ODbL v1.0](https://opendatacommons.org/licenses/odbl/1-0/) |

> DWD data is licensed CC-BY-4.0 and is used here with appropriate attribution to the
> Deutscher Wetterdienst (DWD).

## Project structure

The project follows the standard Snakemake layout: the workflow is orchestrated
from `workflow/` and launched from the project root.

```
.
├── config/               # configuration (config.yaml + per-workflow settings)
├── src/                  # workflow components (command-line tools)
│   ├── common/           # shared helpers (utils) + plot styling
│   ├── data/             # data acquisition + general preprocessing
│   └── rq1/ … rq9/       # per-research-question scripts
├── workflow/
│   ├── Snakefile         # master workflow (runs all 9 research questions)
│   └── rules/            # common.smk + one rq{1..9}.smk per research question
├── data/                 # committed example inputs; raw/derived data ignored
├── results/              # generated figures & tables (git-ignored, per workflow_N/)
├── tests/                # automated tests
├── docs/                 # requirements.md + activity diagram
├── notebooks/            # per-research-question reporting notebooks
├── requirements.txt
├── README.md  LICENSE  CONDUCT.md
└── .gitignore
```

## Setup

The repository includes three example input files, so the complete workflow
can run immediately after cloning. The original Spotify charts download is
approximately 3 GB and is therefore not committed; acquisition scripts and
rebuilding instructions are provided in [`data/README.md`](data/README.md).

Python 3.11 or newer is required. The runtime, test, linting, and notebook
dependencies are all declared in `requirements.txt`.

### Platform support

The workflow has been tested on **macOS 14 (Apple Silicon/ARM64)** with Python
3.14. Linux and Windows through **WSL2** are expected to work because the
workflow uses platform-independent Python components, but those platforms have
not yet been tested by the project team. Native Windows execution is not
currently claimed or supported; Windows users should run the following commands
inside WSL2.

### Clone and install

Clone the repository and enter its root directory:

```bash
git clone https://gitup.uni-potsdam.de/mammadov/sky-beats.git
cd sky-beats
```

The Git.UP project is private until submission, so cloning requires an account
that has been granted project access.

Create and activate a virtual environment, then install the dependencies:

macOS, Linux, or WSL2:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If using PowerShell for development tasks outside the supported WSL2 workflow,
the equivalent environment activation command is:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

All subsequent commands must be run from the repository root with the virtual
environment activated. The default analysis uses the committed example inputs,
so it does not require Kaggle credentials, network access, or the 3 GB raw
Spotify download.

## Running the workflow

The entire analysis is orchestrated by [Snakemake](https://snakemake.readthedocs.io/)
from the **project root**. To run every workflow (all nine research questions) end
to end:

```bash
snakemake --cores 1
```

This regenerates all outputs under `results/workflow_1/ … results/workflow_9/`. Add
`-F` to force a clean re-run of everything.

The default workflow starts from the three committed example inputs in `data/`;
raw acquisition is an optional preparation procedure documented in
[`data/README.md`](data/README.md). Reporting notebooks under `notebooks/` are
supplementary interpretations of the generated outputs and are opened or run
manually after Snakemake completes.

To run a single research question, ask for its output(s), for example RQ3:

```bash
snakemake --cores 1 results/workflow_3/genre_shift.png
```

All parameters (date range, thresholds, model settings, per-question options) live in
`config/config.yaml` — edit them there rather than in the scripts.

### Expected outputs

Successful execution creates the following result directories:

| Directory | Research output |
|-----------|-----------------|
| `results/workflow_1/` | Sunshine–valence correlation and seasonal plot |
| `results/workflow_2/` | Precipitation-group and continuous-correlation analyses |
| `results/workflow_3/` | Genre shifts in rainy versus dry months |
| `results/workflow_4/` | Audio-feature associations with extreme-temperature deviations |
| `results/workflow_5/` | Weather-to-energy prediction and seasonal effects |
| `results/workflow_6/` | Gloomy-versus-clear audio-profile comparison |
| `results/workflow_7/` | Weather-to-mood models and song recommendations |
| `results/workflow_8/` | Weather models versus seasonal baselines |
| `results/workflow_9/` | Weather-category prediction from audio features |

The workflow uses `--cores 1` by default for predictable memory use. Runtime
depends on the machine and installed Python stack and has not yet been formally
benchmarked. Allow several gigabytes of free disk space for the virtual
environment and generated intermediate/results files. The optional raw-data
rebuild requires substantially more space because the Spotify source download
alone is approximately 3 GB.

## Verifying the installation

Run the automated tests and validate the workflow graph before starting the
analysis:

```bash
python -m pytest tests/
snakemake --cores 1 --dry-run
```

Then execute the complete workflow and immediately check its reproducibility:

```bash
snakemake --cores 1
snakemake --cores 1
```

The first invocation generates the results. If no inputs, code, or
configuration changed, the second should report that there is nothing to be
done.

## Viewing the reporting notebooks

The notebooks interpret the outputs generated by Snakemake. After the workflow
has completed, start Jupyter from the repository root:

```bash
jupyter notebook
```

Open the corresponding `notebooks/rq*.ipynb` file. The notebooks are
supplementary reports and are intentionally not executed by the default
Snakemake target.

## Troubleshooting

- **`snakemake: command not found`** — activate `.venv` and rerun
  `pip install -r requirements.txt`.
- **Python reports a missing package** — confirm that `python` and `pip` both
  resolve inside `.venv` (`which python` and `which pip` on macOS/Linux/WSL2).
- **PowerShell blocks `Activate.ps1`** — use WSL2, which is the supported
  Windows route, rather than changing machine-wide execution policy.
- **Matplotlib cannot write its cache** — set a writable cache for the current
  shell, for example `export MPLCONFIGDIR=/tmp/sky-beats-matplotlib`.
- **Snakemake cannot write its cache on macOS** — run with a writable cache,
  for example `XDG_CACHE_HOME=/tmp/sky-beats-cache snakemake --cores 1`.
- **A workflow rule fails** — inspect the final command/error printed by
  Snakemake, correct the reported input or environment problem, and rerun the
  same command. Snakemake will resume from completed outputs.
- **Raw-data download fails** — raw rebuilding is optional. Restore the three
  committed example CSV files in `data/` and run the default workflow instead.

## Limitations and potential confounders

The workflow identifies associations and predictive patterns; it does **not**
establish that weather causes changes in music listening. Results should be
interpreted with the following limitations in mind:

- **Seasonality and cultural events:** weather, music releases, holidays, and
  events such as Christmas, Oktoberfest, and Fasching vary with the calendar.
  Seasonal adjustment reduces but cannot eliminate this confounding.
- **COVID-19 period:** the 2017–2021 window includes pandemic restrictions in
  2020–2021, when mobility, routines, entertainment, and streaming behaviour
  changed independently of weather.
- **Chart-selection bias:** Spotify Top-200 charts represent popular tracks and
  exclude listening outside the chart. The results therefore do not describe
  all German Spotify users or the general population.
- **Dataset matching and genre labels:** tracks without a matching audio-feature
  record are excluded, and a track's assigned genre may be incomplete or
  ambiguous. Both can affect the resulting sample.
- **Temporal and platform scope:** conclusions apply to the available German
  Spotify data from 2017–2021 and may not transfer to other periods, countries,
  music platforms, or definitions of gloomy, clear, and extreme weather.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CONDUCT.md).

## Contributing

Contributions are welcome! Please read the
[contribution guidelines](CONTRIBUTING.md) for how to report bugs, propose
changes, and submit merge requests.

## Citing this project

Citation metadata is provided in [CITATION.cff](CITATION.cff). If you use this
software, please cite it using the information in that file (GitLab/GitHub can
render it as BibTeX or APA automatically).

## License

This project is licensed under the MIT License: see [LICENSE](LICENSE).
