# Shared rules used by every workflow (defined once here, instead of being
# copy-pasted into each workflow as before).
#
# All paths are relative to the PROJECT ROOT, because the workflow is launched
# from there (`snakemake --cores 1`).

# General preprocessing: clean and normalise the merged song-level dataset.
# Every research-question workflow depends on this single output.
rule datapreprocessing:
    input:
        script="src/data/dataprocessing.py",
        csv="data/spotify_weather_merged.csv",
    output:
        "data/spotify_weather_merged_processed.csv",
    shell:
        "python {input.script} -i {input.csv} -o {output}"
