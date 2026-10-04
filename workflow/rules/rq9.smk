# Workflow 9 (RQ9): can the day's weather category (gloomy vs clear) be inferred
# from the music alone? The reverse of RQ6 - a classifier on the daily audio profile.

# RQ9 preprocessing: daily audio profile + gloomy/clear labels, filtered to two classes
rule rq9_preprocess:
    input:
        script="src/rq9/datapreprocessing_w9.py",
        csv="data/spotify_weather_merged_processed.csv",
    output:
        "data/spotify_weather_merged_processed_w9.csv",
    params:
        sun_low=config["rq9"]["sun_low_pct"],
        sun_high=config["rq9"]["sun_high_pct"],
        precip_low=config["rq9"]["precip_low_pct"],
        precip_high=config["rq9"]["precip_high_pct"],
    shell:
        "python {input.script} {input.csv} -o {output} "
        "--sun-low-pct {params.sun_low} --sun-high-pct {params.sun_high} "
        "--precip-low-pct {params.precip_low} --precip-high-pct {params.precip_high}"

# Train + compare classifiers (with a majority baseline); save metrics, model,
# held-out predictions, and feature importances.
rule rq9_train:
    input:
        script="src/rq9/train_classifier.py",
        csv="data/spotify_weather_merged_processed_w9.csv",
    output:
        metrics="results/workflow_9/metrics.csv",
        model="results/workflow_9/classifier_model.pkl",
        preds="results/workflow_9/predictions.csv",
        importance="results/workflow_9/feature_importance.csv",
    params:
        features=",".join(config["rq9"]["audio_features"]),
        models=",".join(config["rq9"]["models"]),
        test_size=config["rq9"]["test_size"],
        cv_folds=config["rq9"]["cv_folds"],
        seed=config["rq9"]["random_seed"],
    shell:
        "python {input.script} {input.csv} "
        "-o {output.metrics} -m {output.model} "
        "--predictions {output.preds} --importance {output.importance} "
        '--features "{params.features}" --models "{params.models}" '
        "--test-size {params.test_size} --cv-folds {params.cv_folds} "
        "--random-seed {params.seed}"

# Plot the confusion matrix + feature importance of the best model
rule rq9_plot:
    input:
        script="src/rq9/plot_classifier.py",
        preds="results/workflow_9/predictions.csv",
        importance="results/workflow_9/feature_importance.csv",
    output:
        "results/workflow_9/classifier_report.png",
    shell:
        "python {input.script} {input.preds} {input.importance} -o {output}"
