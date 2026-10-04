# Workflow 8 (RQ8): does daily weather genuinely predict music mood *beyond*
# seasonal effects? Compares a weather-feature regression against a
# season/month-only baseline on held-out data (FR-15).

# RQ8 preprocessing: daily audio profile + month/season calendar features
rule rq8_preprocess:
    input:
        script="src/rq8/datapreprocessing_w8.py",
        csv="data/spotify_weather_merged_processed.csv",
    output:
        "data/spotify_weather_merged_processed_w8.csv",
    shell:
        "python {input.script} {input.csv} -o {output}"

# Fit season-only baseline vs weather model, evaluate on held-out days
rule rq8_compare:
    input:
        script="src/rq8/compare_models_w8.py",
        csv="data/spotify_weather_merged_processed_w8.csv",
    output:
        "results/workflow_8/model_comparison.csv",
    params:
        target=config["rq8"]["target"],
        weather=",".join(config["rq8"]["weather_features"]),
        models=",".join(config["rq8"]["models"]),
        test_fraction=config["rq8"]["test_fraction"],
        seed=config["rq8"]["random_seed"],
    shell:
        'python {input.script} {input.csv} -o {output} '
        '--target "{params.target}" --weather-features "{params.weather}" '
        '--models "{params.models}" --test-fraction {params.test_fraction} '
        '--seed {params.seed}'

# Plot: grouped bars of test R2 / MAE, baseline vs weather, per estimator
rule rq8_plot:
    input:
        script="src/rq8/plot_model_comparison.py",
        csv="results/workflow_8/model_comparison.csv",
    output:
        "results/workflow_8/model_comparison.png",
    shell:
        "python {input.script} {input.csv} -o {output}"
