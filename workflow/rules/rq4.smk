# Workflow 4 (RQ4): To what extent do extreme temperature deviations (very hot or very cold)
# affect the audio features of top-streamed songs?
# 
RQ4_FEATURES = ["acousticness", "energy", "tempo", "valence", "danceability"]

# RQ4 wrapper rule – collects all outputs from both analyses
rule rq4:
    input:
        expand(
            [
                "results/workflow_4/kruskal_extremes_{feature}.json",
                "results/workflow_4/kruskal_extremes_{feature}.png",
                "results/workflow_4/kruskal_extremes_{feature}.md",
                "results/workflow_4/abs_dev_regression_{feature}.json",
                "results/workflow_4/abs_dev_regression_{feature}.txt",
                "results/workflow_4/abs_dev_regression_{feature}.png",
                "results/workflow_4/abs_dev_regression_{feature}.md",
            ],
            feature=RQ4_FEATURES,
        )

# RQ4 preprocessing: compute temperature deviation and flag extremes
rule rq4_preprocess:
    input:
        script="src/rq4/datapreprocessing_w4.py",
        csv="data/spotify_weather_merged_processed.csv",
    output:
        "data/spotify_weather_merged_processed_w4.csv",
    params:
        method=config["rq4"].get("extreme_method", "percentile"),  # only percentile supported now
    shell:
        "python {input.script} -i {input.csv} -o {output}"

# ----------------------------------------------------------------------
#  Kruskal-Wallis across Cold/Normal/Hot
# ----------------------------------------------------------------------

rule rq4_kruskal:
    input:
        script="src/rq4/kruskal_extremes_w4.py",
        csv="data/spotify_weather_merged_processed_w4.csv",
    output:
        "results/workflow_4/kruskal_extremes_{feature}.json",
    params:
        cat_column="temperature_deviation_extreme",   # three-level category
    shell:
        'python {input.script} {input.csv} -y "{wildcards.feature}" '
        '-c "{params.cat_column}" -o {output}'

rule rq4_kruskal_report:
    input:
        script="src/rq4/kruskal_report_w4.py",
        json="results/workflow_4/kruskal_extremes_{feature}.json",
    output:
        png="results/workflow_4/kruskal_extremes_{feature}.png",
        md="results/workflow_4/kruskal_extremes_{feature}.md",
    shell:
        "python {input.script} -j {input.json} -o {output.png} -m {output.md}"

# ----------------------------------------------------------------------
#    OLS regression on absolute deviation
# ----------------------------------------------------------------------

rule rq4_abs_dev_regression:
    input:
        script="src/rq4/deviation_magnitude_regression_w4.py",
        csv="data/spotify_weather_merged_processed_w4.csv",
    output:
        json="results/workflow_4/abs_dev_regression_{feature}.json",
        txt="results/workflow_4/abs_dev_regression_{feature}.txt",
    shell:
        'python {input.script} {input.csv} -y "{wildcards.feature}" '
        '-o {output.json} -s {output.txt}'

rule rq4_abs_dev_report:
    input:
        script="src/rq4/deviation_magnitude_regression_report_w4.py",
        json="results/workflow_4/abs_dev_regression_{feature}.json",
        csv="data/spotify_weather_merged_processed_w4.csv",
    output:
        png="results/workflow_4/abs_dev_regression_{feature}.png",
        md="results/workflow_4/abs_dev_regression_{feature}.md",
    shell:
        "python {input.script} -j {input.json} -c {input.csv} "
        "-o {output.png} -m {output.md}"