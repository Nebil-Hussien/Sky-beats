import yaml
import matplotlib.pyplot as plt
import pandas as pd


AUDIO_FEATURES = [
    "acousticness",
    "energy",
    "tempo",
    "valence",
    "danceability"
]

# Per-day weather columns (identical for every chart row on a given date).
WEATHER_COLS = [
    "temperature_mean_c",
    "sunshine_hours",
    "precipitation_mm",
]


def aggregate_daily(df):
    """
    Aggregate a track-level dataframe to one row per date.

    If the dataframe is already daily, it is returned unchanged.

    Args:
        df (pd.DataFrame)

    Returns:
        pd.DataFrame
    """

    if not df["date"].duplicated().any():
        return df

    aggregation = {col: "first" for col in WEATHER_COLS if col in df.columns}

    for feature in AUDIO_FEATURES:
        if feature in df.columns:
            aggregation[feature] = "mean"

    return (
        df.groupby("date", as_index=False)
        .agg(aggregation)
    )


def label_gloomy_clear(df, sun_low_pct=25, sun_high_pct=75,
                       precip_low_pct=25, precip_high_pct=75,
                       sunshine_col="sunshine_hours", precip_col="precipitation_mm"):
    """
    Label each daily row as 'gloomy', 'clear', or 'other' using PERCENTILE
    cut-points of the daily sunshine and precipitation distributions.

        gloomy : sunshine <= P(sun_low_pct)  AND precip >= P(precip_high_pct)
        clear  : sunshine >= P(sun_high_pct) AND precip <= P(precip_low_pct)
        everything else is 'other'.

    Percentiles are used instead of the README's literal "0 sunshine / >2 mm"
    definition because our sunshine is a NATIONAL AVERAGE across DWD stations, so it
    is almost never exactly 0 — the fixed definition yields ~no gloomy days.
    Percentiles give balanced, data-driven groups. Shared by RQ6 (audio-profile
    comparison) and RQ9 (reverse classification) so the definition never diverges.

    Note: percentiles are computed from the passed dataframe, so pass the DAILY
    weather series (one row per day), not the song-level table.

    Args:
        df (pd.DataFrame): daily rows with sunshine + precipitation columns
        sun_low_pct, sun_high_pct, precip_low_pct, precip_high_pct (float):
            percentile cut-points in [0, 100]
        sunshine_col, precip_col (str): column names to read

    Returns:
        pd.DataFrame: copy of df with an added 'day_type' column
    """
    df = df.copy()
    sun_low = df[sunshine_col].quantile(sun_low_pct / 100)
    sun_high = df[sunshine_col].quantile(sun_high_pct / 100)
    precip_low = df[precip_col].quantile(precip_low_pct / 100)
    precip_high = df[precip_col].quantile(precip_high_pct / 100)

    is_gloomy = (df[sunshine_col] <= sun_low) & (df[precip_col] >= precip_high)
    is_clear = (df[sunshine_col] >= sun_high) & (df[precip_col] <= precip_low)
    df["day_type"] = "other"
    df.loc[is_gloomy, "day_type"] = "gloomy"
    df.loc[is_clear, "day_type"] = "clear"
    return df


def create_temperature_deviation(
        df,
        temperature_column="temperature_mean_c"):
    """
    Compute monthly temperature deviation.

    deviation =
        daily temperature -
        monthly climatological mean.

    Args:
        df (pd.DataFrame)

    Returns:
        pd.DataFrame
    """

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    monthly_mean = (
        df.groupby(df["date"].dt.month)[temperature_column]
        .transform("mean")
    )

    df["temperature_deviation"] = (
        df[temperature_column] -
        monthly_mean
    )

    return df


def load_config(config_file):
    """The function loads configuration parameters
    stored in config_file (.yaml).

    Args:
        config_file (str): path of the configuration file

    Raises:
        ValueError: when config_file not a  .yaml file

    Returns:
        dict: configuration params
    """
    if ('.yaml' not in config_file):
        message = 'Please provide a .yaml file'
        raise ValueError(message)
    with open(config_file) as file:
        config = yaml.safe_load(file)

    return config


def set_plot_params(config_file):
    """The function sets the matplotlib parameters
    according to the ones provided in config_file.

    Args:
        config_file (_type_): path of the configuration file
    Returns:
        None
    """
    plot_params = load_config(config_file)
    for param, value in plot_params.items():
        if (param == 'style'):
            plt.style.use(value)
        else:
            plt.rcParams[param] = value
