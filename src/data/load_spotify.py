#!/usr/bin/env python3
"""
load_spotify.py
-------------------
Downloads two Spotify datasets either from Kaggle or from the team's
public Box.UP (Nextcloud) share link at Uni Potsdam — no login required
for the Box.UP option since it uses a public share link.

The source is selected through configuration or the ``--source`` option.

Datasets:
  2a: Spotify Charts Germany  (dhruvildave/spotify-charts)
  2b: Spotify Tracks & Audio Features (maharshipandya/-spotify-tracks-dataset)

Requirements:
  pip install kaggle pyyaml requests

Kaggle API setup:
  See README.md for instructions on creating and placing your Kaggle API token.

Box.UP setup:
  No login needed — uses the public share link set in config.yaml
  under boxup.share_url

Usage:
  python src/data/load_spotify.py --help
"""

import argparse
import logging
import os
from pathlib import Path
import subprocess
import yaml
import requests
from tqdm import tqdm

# ── Settings ───────────────────────────────────────────────────────────────────

# Paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "..", "..", "config", "config.yaml")
LOGGER = logging.getLogger(__name__)

# ── Single download function for both sources ─────────────────────────────────


def should_download_file(file_path, force=False):
    """Return True when a fresh download is needed."""
    if force:
        return True
    if not os.path.isfile(file_path):
        return True
    return os.path.getsize(file_path) == 0


def fetch_dataset(source, dataset_id, filename, file_path, label, raw_dir,
                  share_url="", share_password="", force=False,
                  kaggle_filename=None):
    """
    Download a dataset file. Skips the download if the file already exists,
    unless ``force`` is True.

    Parameters:
        source     : "kaggle" or "boxup"
        dataset_id : Kaggle dataset slug (used only if source == "kaggle")
        filename   : filename inside the Box.UP share (used only if source == "boxup")
        file_path  : local path to save the file to
        label      : human-readable name for print messages
        force      : if True, re-download even when the file already exists
        kaggle_filename : the name Kaggle unzips to (renamed to file_path afterwards)
    """
    if not should_download_file(file_path, force):
        print(f"Using existing {label} file (set spotify.force_download: true to re-download).")
        return

    print(f"Downloading {label} from {'Kaggle' if source == 'kaggle' else 'Box.UP'}...")

    if source == "kaggle":
        # list form (no shell) handles paths with spaces; check=True fails loudly
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_id, "-p", raw_dir, "--unzip"],
            check=True,
        )
        # Kaggle unzips to the dataset's own filename; rename it to our config name.
        if kaggle_filename:
            extracted = os.path.join(raw_dir, kaggle_filename)
            if os.path.abspath(extracted) != os.path.abspath(file_path):
                os.replace(extracted, file_path)
                print(f"  → Renamed {kaggle_filename} → {os.path.basename(file_path)}")

    elif source == "boxup":
        if not share_url:
            raise SystemExit("ERROR: set boxup.share_url in config.yaml to your public Box.UP link.")

        # Extract the share token from the share URL
        # e.g. "https://boxup.uni-potsdam.de/s/a749mA6DR6mqqXc" -> "a749mA6DR6mqqXc"
        share_token = share_url.rstrip("/").split("/")[-1]
        base = share_url.split("/s/")[0]

        # Password-protected public shares use the WebDAV endpoint:
        # /public.php/dav/files/{share_token}/{filename}
        # with basic auth: username="anonymous", password=share_password
        download_url = f"{base}/public.php/dav/files/{share_token}/{filename}"
        auth = ("anonymous", share_password) if share_password else None

        response = requests.get(download_url, auth=auth, stream=True, timeout=60)
        response.raise_for_status()

        # Download to a temp file and rename on success, so an interrupted
        # download never leaves a corrupt file that later looks "already downloaded".
        total_size = int(response.headers.get("content-length", 0))
        tmp_path = file_path + ".tmp"
        with open(tmp_path, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc=label
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        os.replace(tmp_path, file_path)
        print(f"  → Saved to {file_path}")

    else:
        raise SystemExit(f"ERROR: unknown source '{source}'. Use 'boxup' or 'kaggle' in config.yaml.")


def parse_args():
    parser = argparse.ArgumentParser(description="Download the two Spotify input datasets")
    parser.add_argument("--config", type=Path, default=Path(CONFIG_PATH))
    parser.add_argument("--source", choices=("boxup", "kaggle"), help="Override configured source")
    parser.add_argument("--force", action="store_true", help="Replace existing downloads")
    return parser.parse_args()


def main(config_path, source_override=None, force_override=False):
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    root = Path(__file__).resolve().parents[2]
    raw_dir = str(root / config["paths"]["raw_dir"])
    os.makedirs(raw_dir, exist_ok=True)
    spotify = config["spotify"]
    boxup = config.get("boxup", {})
    source = source_override or spotify.get("source", "boxup").lower()
    force = force_override or spotify.get("force_download", False)
    for dataset, filename, label, kaggle_name in (
        (spotify["charts_dataset"], spotify["charts_file"], "Spotify Charts", spotify.get("kaggle_charts_file")),
        (spotify["features_dataset"], spotify["features_file"], "Spotify Audio Features", spotify.get("kaggle_features_file")),
    ):
        fetch_dataset(source, dataset, filename, os.path.join(raw_dir, filename),
                      label, raw_dir, boxup.get("share_url", ""),
                      boxup.get("share_password", ""), force, kaggle_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    main(args.config, args.source, args.force)
