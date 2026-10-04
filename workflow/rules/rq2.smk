# Workflow 2 (RQ2): In what ways does high daily precipitation influence the preference for 'Acoustic' vs. 'High-Energy' tracks in the German charts?
#
# Two parallel statistical tests per feature:
#   - Kruskal-Wallis across the 4 precipitation buckets (No rain/Low/Moderate/High)
#   - Spearman correlation against continuous precipitation_mm
# Each test has its own reporting rule (plot + markdown summary, generated
# directly from that test's JSON output). A separate bar plot visualizes the
# bucket means annotated with the Spearman result, for a combined report figure.
#
# Uses a {feature} wildcard so one rule set covers all audio features.

RQ2_FEATURES = ["acousticness", "energy", "tempo", "valence", "danceability"]

# RQ2 Wrapper rule
rule rq2:
    input:
        expand(
            [
                "results/workflow_2/kruskal_report_{feature}.png",
                "results/workflow_2/kruskal_report_{feature}.md",
                "results/workflow_2/spearman_report_{feature}.png",
                "results/workflow_2/spearman_report_{feature}.md",
                "results/workflow_2/bar_plot_{feature}.png",
            ],
            feature=RQ2_FEATURES,
        )

# RQ2 preprocessing: add precipitation-category buckets
rule rq2_preprocess:
    input:
        script="src/rq2/datapreprocessing_w2.py",
        csv="data/spotify_weather_merged_processed.csv",
    output:
        "data/spotify_weather_merged_processed_w2.csv",
    params:
        method=config["rq2"]["bucket_method"],
    shell:
        "python {input.script} -i {input.csv} -o {output}"

# Kruskal-Wallis test of the audio feature across the 4 precipitation buckets
rule rq2_kruskal:
    input:
        script="src/rq2/kruskal_wallis.py",
        csv="data/spotify_weather_merged_processed_w2.csv",
    output:
        "results/workflow_2/kruskal_{feature}.json",
    params:
        group=config["rq2"]["group_variable"],
    shell:
        'python {input.script} {input.csv} -y "{wildcards.feature}" '
        '-c "{params.group}" -o {output}'

# Reporting plot + markdown summary for the Kruskal-Wallis result
rule rq2_kruskal_report:
    input:
        script="src/rq2/kruskal_report.py",
        json="results/workflow_2/kruskal_{feature}.json",
    output:
        png="results/workflow_2/kruskal_report_{feature}.png",
        md="results/workflow_2/kruskal_report_{feature}.md",
    shell:
        "python {input.script} -j {input.json} -o {output.png} -m {output.md}"
# Spearman correlation of precipitation vs each audio feature (one per wildcard)
rule rq2_spearman:
    input:
        script="src/rq2/spearman_correlation.py",
        csv="data/spotify_weather_merged_processed_w2.csv",
    output:
        "results/workflow_2/spearman_{feature}.json",
    params:
        x=config["rq2"]["x_variable"],
    shell:
        'python {input.script} {input.csv} -x "{params.x}" -y "{wildcards.feature}" -o {output}'

# Reporting plot + markdown summary for the Spearman result
rule rq2_spearman_report:
    input:
        script="src/rq2/spearman_report.py",
        json="results/workflow_2/spearman_{feature}.json",
        csv="data/spotify_weather_merged_processed_w2.csv",
    output:
        png="results/workflow_2/spearman_report_{feature}.png",
        md="results/workflow_2/spearman_report_{feature}.md",
        bar="results/workflow_2/bar_plot_{feature}.png",
    params:
        group=config["rq2"]["group_variable"],
    shell:
        'python {input.script} -j {input.json} -c {input.csv} '
        '-o {output.png} -m {output.md} '
        '--group "{params.group}" --bar-output {output.bar}'