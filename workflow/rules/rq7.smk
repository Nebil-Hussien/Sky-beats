# Workflow 7 (RQ7): can daily weather attributes predict music mood, and can that
# be used to recommend songs? Trains weather -> mood models, then recommends
# tracks for a given day's weather (read from config/configuration_w7.yaml).
#
# Note: this workflow starts from the daily-aggregated dataset, so it does not use
# the shared song-level `datapreprocessing` rule.

# Validate schema + impute missing values on the daily-aggregated dataset,
# reusing the shared src/data/dataprocessing.py script (daily column schema).
rule rq7_dataprocessing:
    input:
        script="src/data/dataprocessing.py",
        csv="data/spotify_weather_daily_merged.csv",
    output:
        "data/spotify_weather_daily_merged_processed.csv",
    shell:
        "python {input.script} -i {input.csv} -o {output} --schema daily"

# Add calendar + holiday features to the validated daily dataset
rule rq7_preprocess:
    input:
        script="src/rq7/datapreprocessing_w7.py",
        raw_csv="data/spotify_weather_daily_merged_processed.csv",
    output:
        "results/workflow_7/spotify_weather_daily_processed.csv",
    shell:
        "python {input.script} {input.raw_csv} {output}"

# Train and compare weather -> mood models
rule rq7_train:
    input:
        script="src/rq7/train_model.py",
        csv="results/workflow_7/spotify_weather_daily_processed.csv",
    output:
        model="results/workflow_7/weather_model.pkl",
        metrics="results/workflow_7/weather_metrics.csv",
    shell:
        "python {input.script} {input.csv} -m {output.model} -o {output.metrics}"

# Recommend songs for the configured "current weather"
rule rq7_recommend:
    input:
        script="src/rq7/recommend_song.py",
        model="results/workflow_7/weather_model.pkl",
        songs_csv="data/spotify_germany.csv",
        weather_yaml="config/configuration_w7.yaml",
    output:
        "results/workflow_7/song_recommendations.csv",
    shell:
        "python {input.script} {input.model} {input.songs_csv} "
        "-r {output} -w {input.weather_yaml}"

# Plot feature importances of the trained model
rule rq7_plot:
    input:
        script="src/rq7/plot_results.py",
        model="results/workflow_7/weather_model.pkl",
        # Only actually read as a fallback for models without feature_importances_/coef_
        # (e.g. hist_gradient_boosting), via permutation importance.
        data="results/workflow_7/spotify_weather_daily_processed.csv",
    output:
        "results/workflow_7/feature_importances.png",
    shell:
        "python {input.script} {input.model} {output} --data {input.data}"
