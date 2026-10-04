# Workflow 6 (RQ6): what is the audio profile of the charts on gloomy days
# (dark + wet) compared to clear days (sunny + dry)?

# RQ6 preprocessing: daily audio profile + percentile-based gloomy/clear labels
rule rq6_preprocess:
    input:
        script="src/rq6/datapreprocessing_w6.py",
        csv="data/spotify_weather_merged_processed.csv",
    output:
        "data/spotify_weather_merged_processed_w6.csv",
    params:
        sun_low=config["rq6"]["sun_low_pct"],
        sun_high=config["rq6"]["sun_high_pct"],
        precip_low=config["rq6"]["precip_low_pct"],
        precip_high=config["rq6"]["precip_high_pct"],
    shell:
        "python {input.script} {input.csv} -o {output} "
        "--sun-low-pct {params.sun_low} --sun-high-pct {params.sun_high} "
        "--precip-low-pct {params.precip_low} --precip-high-pct {params.precip_high}"

# Analysis: per-feature gloomy vs clear means + Mann-Whitney U test
rule rq6_audio_profile:
    input:
        script="src/rq6/audio_profile.py",
        csv="data/spotify_weather_merged_processed_w6.csv",
    output:
        "results/workflow_6/profile_comparison.csv",
    params:
        features=",".join(config["rq6"]["audio_features"]),
    shell:
        'python {input.script} {input.csv} -o {output} --features "{params.features}"'

# Plot: radar chart of the two normalised profiles
rule rq6_plot:
    input:
        script="src/rq6/plot_audio_profile.py",
        csv="data/spotify_weather_merged_processed_w6.csv",
    output:
        "results/workflow_6/audio_profile_radar.png",
    params:
        features=",".join(config["rq6"]["audio_features"]),
    shell:
        'python {input.script} {input.csv} -o {output} --features "{params.features}"'
