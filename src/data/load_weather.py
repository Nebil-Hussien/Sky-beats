#!/usr/bin/env python3
"""Download and aggregate DWD daily climate observations."""

import argparse
import io
import logging
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests
import yaml
from tqdm import tqdm

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="YAML configuration file (default: %(default)s)")
    parser.add_argument("--output", type=Path,
                        help="Override the configured output CSV path")
    parser.add_argument("--timeout", type=float, default=60,
                        help="HTTP timeout in seconds (default: %(default)s)")
    return parser.parse_args()


def main(config_path, output=None, timeout=60):
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    root = Path(__file__).resolve().parents[2]
    base_url = config["dwd"]["base_url"]
    output = output or root / config["paths"]["output_dir"] / config["dwd"]["output_file"]

    LOGGER.info("Getting list of DWD stations")
    response = requests.get(base_url, timeout=timeout)
    response.raise_for_status()
    zip_files = re.findall(r'href="(tageswerte_KL_\d+_\d+_\d+_hist\.zip)"', response.text)
    LOGGER.info("Found %s stations", len(zip_files))

    all_data = []
    skipped = 0
    needed = config["dwd"]["columns"]
    for filename in tqdm(zip_files, desc="Downloading stations", unit="station"):
        station_response = requests.get(base_url + filename, timeout=timeout)
        if station_response.status_code != 200:
            skipped += 1
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(station_response.content)) as archive:
                member = next(name for name in archive.namelist() if name.startswith("produkt_"))
                with archive.open(member) as stream:
                    frame = pd.read_csv(stream, sep=";", encoding="latin1")
        except (OSError, ValueError, zipfile.BadZipFile, StopIteration):
            skipped += 1
            continue
        frame.columns = frame.columns.str.strip()
        if not set(needed).issubset(frame.columns):
            skipped += 1
            continue
        frame = frame[needed].replace(-999.0, float("nan"))
        frame["MESS_DATUM"] = pd.to_datetime(frame["MESS_DATUM"], format="%Y%m%d", errors="coerce")
        frame = frame[frame["MESS_DATUM"].between(config["dates"]["start"], config["dates"]["end"])]
        if not frame.empty:
            all_data.append(frame)

    if not all_data:
        raise SystemExit("ERROR: no station data downloaded; check the URL, dates, and network connection")
    daily = (pd.concat(all_data, ignore_index=True)
             .groupby("MESS_DATUM")[["RSK", "SDK", "TMK"]].mean().round(2).reset_index()
             .rename(columns={"MESS_DATUM": "date", "RSK": "precipitation_mm",
                              "SDK": "sunshine_hours", "TMK": "temperature_mean_c"}))
    output.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output, index=False)
    LOGGER.info("Saved %s rows to '%s' (%s stations skipped)", len(daily), output, skipped)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    main(args.config, args.output, args.timeout)
