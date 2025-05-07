"""Download the UCI Online Retail II dataset if not already present."""

import os
import sys
from pathlib import Path

import requests

DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/"
    "online_retail_II.xlsx"
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEST_FILE = RAW_DIR / "online_retail_II.xlsx"


def download(url: str = DATASET_URL, dest: Path = DEST_FILE) -> Path:
    """Download the dataset file. Returns the path to the downloaded file."""
    if dest.exists():
        print(f"Dataset already exists at {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {url} ...")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                print(f"\r  {pct}% ({downloaded >> 20} MB)", end="", flush=True)
    print(f"\nSaved to {dest}")
    return dest


def main() -> None:
    download()


if __name__ == "__main__":
    main()
