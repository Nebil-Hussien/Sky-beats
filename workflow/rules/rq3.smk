# Workflow 3 (RQ3): which music genres become more or less popular in the
# rainiest months compared to the driest months?
#
# Uses the shared processed dataset directly (no RQ3-specific preprocessing).

# Per-genre stream/count-share shift, rainiest vs driest months
rule rq3_shift:
    input:
        script="src/rq3/genre_shift.py",
        csv="data/spotify_weather_merged_processed.csv",
    output:
        "results/workflow_3/genre_shift.csv",
    params:
        n_months=config["rq3"]["n_months"],
    shell:
        "python {input.script} {input.csv} -o {output} --n-months {params.n_months}"

# Plot the largest shifts
rule rq3_plot:
    input:
        script="src/rq3/plot_genre_shift.py",
        csv="results/workflow_3/genre_shift.csv",
    output:
        "results/workflow_3/genre_shift.png",
    params:
        top_genres=config["rq3"]["top_genres"],
    shell:
        "python {input.script} {input.csv} -o {output} --top-genres {params.top_genres}"
