# Workflow 1 (RQ1): how does daily sunshine duration correlate with the mean
# valence of the German Spotify Top 200, and does the relationship vary by season?
#
# Depends on the shared `datapreprocessing` rule (common.smk) for the processed
# dataset. All paths are relative to the project root.

# RQ1 preprocessing: daily audio profile + meteorological season labels
rule rq1_preprocess:
    input:
        script="src/rq1/datapreprocessing_w1.py",
        csv="data/spotify_weather_merged_processed.csv",
    output:
        "data/spotify_weather_merged_processed_w1.csv",
    shell:
        "python {input.script} {input.csv} -o {output}"

# Analysis: Spearman correlation of sunshine vs valence, overall + per season
rule rq1_correlation:
    input:
        script="src/rq1/sunshine_valence_correlation.py",
        csv="data/spotify_weather_merged_processed_w1.csv",
    output:
        "results/workflow_1/sunshine_valence_correlation.csv",
    params:
        x=config["rq1"]["x_variable"],
        y=config["rq1"]["y_variable"],
        alpha=config["rq1"]["alpha"],
    shell:
        'python {input.script} {input.csv} -o {output} '
        '-x "{params.x}" -y "{params.y}" --alpha {params.alpha}'

# Plot: 2x2 seasonal scatter grid with trend lines and rho annotations
rule rq1_plot:
    input:
        script="src/rq1/plot_sunshine_valence.py",
        csv="data/spotify_weather_merged_processed_w1.csv",
        corr="results/workflow_1/sunshine_valence_correlation.csv",
    output:
        "results/workflow_1/sunshine_valence_by_season.png",
    params:
        x=config["rq1"]["x_variable"],
        y=config["rq1"]["y_variable"],
    shell:
        'python {input.script} {input.csv} -c {input.corr} -o {output} '
        '-x "{params.x}" -y "{params.y}"'
