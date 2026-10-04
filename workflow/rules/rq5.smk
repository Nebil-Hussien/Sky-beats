# Workflow 5 (RQ5): how well can daily temperature and sunshine predict the
# overall energy level of the German Top 100?
#
# Two independent analyses:
#   - Predictive accuracy (predict_energy_w5.py): TimeSeriesSplit CV of
#     temperature + sunshine against mean/persistence baselines and a
#     linear vs. gradient boosting comparison. This directly answers RQ5
#     ("how accurately can...predict"). Runs on the general-processed
#     dataset directly — it doesn't need the season column, so it doesn't
#     depend on rq5_preprocess.
#   - Seasonal interaction (regression_energy.py): does the weather ->
#     energy relationship itself vary by season (VIF-checked multiple
#     regression, nested F-test, per-season slopes). A related but
#     distinct question; depends on rq5_preprocess for the season column.

RQ5_FOCUS = config["rq5"]["focus_predictors"]  # one seasonal-effect plot per focus predictor
RQ5_CV_SPLITS = config["rq5"].get("cv_splits", 5)

# RQ5 Wrapper rule
rule rq5:
    input:
        "results/workflow_5/predict_report.png",
        "results/workflow_5/predict_report.md",
        expand(
            "results/workflow_5/plot_{predictor}.png",
            predictor=RQ5_FOCUS,
        ),

# --- Predictive accuracy branch (answers RQ5 directly) ---

# TimeSeriesSplit CV: temperature + sunshine vs mean/persistence baselines,
# linear vs gradient boosting
rule rq5_predict:
    input:
        script="src/rq5/predict_energy_w5.py",
        csv="data/spotify_weather_merged_processed.csv",
    output:
        folds="results/workflow_5/predict_folds.csv",
        summary="results/workflow_5/predict_summary.json",
    params:
        n_splits=RQ5_CV_SPLITS,
    shell:
        "python {input.script} {input.csv} -n {params.n_splits} "
        "--out_folds {output.folds} --out_summary {output.summary}"

# Reporting plot + markdown summary for the predictive accuracy result
rule rq5_predict_report:
    input:
        script="src/rq5/predict_energy_report_w5.py",
        json="results/workflow_5/predict_summary.json",
    output:
        png="results/workflow_5/predict_report.png",
        md="results/workflow_5/predict_report.md",
    shell:
        "python {input.script} -j {input.json} -o {output.png} -m {output.md}"

# --- Seasonal interaction branch (does the weather effect vary by season) ---

# RQ5 preprocessing: derive the meteorological season column
rule rq5_preprocess:
    input:
        script="src/rq5/datapreprocessing_w5.py",
        csv="data/spotify_weather_merged_processed.csv",
    output:
        "data/spotify_weather_merged_processed_w5.csv",
    shell:
        "python {input.script} -i {input.csv} -o {output}"

# Regression: coefficients, VIF, model comparison, seasonal slopes, significance
rule rq5_regression:
    input:
        script="src/rq5/regression_energy.py",
        csv="data/spotify_weather_merged_processed_w5.csv",
    output:
        coefficients="results/workflow_5/coefficients.csv",
        vif="results/workflow_5/vif.csv",
        comparison="results/workflow_5/comparison.csv",
        seasonal_slopes="results/workflow_5/seasonal_slopes.csv",
        significance="results/workflow_5/significance.txt",
        predictive_metrics="results/workflow_5/predictive_metrics.csv"
    shell:
        """
        python {input.script} {input.csv} \
        -c {output.coefficients} \
        --out_vif {output.vif} \
        --out_comparison {output.comparison} \
        --out_seasonal_slopes {output.seasonal_slopes} \
        -o {output.significance} \
        --out_predictive_metrics {output.predictive_metrics}
        """

# Seasonal-effect line plot, one per focus predictor
rule rq5_plot:
    input:
        script="src/rq5/seasonal_effect_line_plot.py",
        csv="data/spotify_weather_merged_processed_w5.csv",
        significance="results/workflow_5/significance.txt",
    output:
        "results/workflow_5/plot_{predictor}.png",
    params:
        predictors=",".join(config["rq5"]["predictors"]),
    shell:
        'python {input.script} {input.csv} -p "{params.predictors}" '
        '-f "{wildcards.predictor}" -o {output}'
