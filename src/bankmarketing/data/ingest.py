"""
Download and extract the Kaggle Bank Marketing competition dataset.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import zipfile
from pathlib import Path

import yaml

LOGGER = logging.getLogger(__name__)

EXPECTED_FILES = {
    "train.csv",
    "test.csv",
    "sample_submission.csv",
}


def load_config(path: str = "configs/data.yaml") -> dict:
    """Load the data configuration."""
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def expected_files_exist(raw_dir: Path) -> bool:
    """Check that every expected raw file exists and is non-empty."""
    return all(
        (raw_dir / filename).exists()
        and (raw_dir / filename).stat().st_size > 0
        for filename in EXPECTED_FILES
    )


def remove_existing_files(raw_dir: Path) -> None:
    """Remove previously downloaded competition files."""
    for filename in EXPECTED_FILES:
        path = raw_dir / filename

        if path.exists():
            LOGGER.info("Removing existing file=%s", path)
            path.unlink()


def download_competition(
    competition: str,
    raw_dir: Path,
) -> Path:
    """Download the Kaggle competition archive."""
    command = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        competition,
        "-p",
        str(raw_dir),
        "--force",
    ]

    LOGGER.info(
        "Downloading Kaggle competition=%s",
        competition,
    )

    try:
        subprocess.run(
            command,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Kaggle CLI was not found. "
            "Install the project dependencies first."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Kaggle download failed. "
            "Check your credentials and competition access."
        ) from exc

    archive = raw_dir / f"{competition}.zip"

    if not archive.exists():
        raise RuntimeError(
            f"Expected Kaggle archive was not created: {archive}"
        )

    if archive.stat().st_size == 0:
        raise RuntimeError(
            f"Downloaded Kaggle archive is empty: {archive}"
        )

    return archive


def extract_archive(
    archive: Path,
    raw_dir: Path,
) -> None:
    """Extract the downloaded Kaggle archive."""
    LOGGER.info(
        "Extracting archive=%s",
        archive,
    )

    with zipfile.ZipFile(archive, "r") as zip_file:
        zip_file.extractall(raw_dir)


def validate_download(raw_dir: Path) -> None:
    """Validate downloaded competition files."""
    missing_files = [
        filename
        for filename in EXPECTED_FILES
        if not (raw_dir / filename).exists()
    ]

    if missing_files:
        raise RuntimeError(
            f"Missing expected files: {missing_files}"
        )

    empty_files = [
        filename
        for filename in EXPECTED_FILES
        if (raw_dir / filename).stat().st_size == 0
    ]

    if empty_files:
        raise RuntimeError(
            f"Empty downloaded files: {empty_files}"
        )


def ingest(force: bool = False) -> None:
    """Run the complete Kaggle ingestion pipeline."""
    config = load_config()

    competition = config["kaggle"]["competition"]
    raw_dir = Path(config["paths"]["raw_dir"])

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if expected_files_exist(raw_dir) and not force:
        LOGGER.info(
            "Raw Kaggle files already exist. "
            "Use --force to download them again."
        )
        return

    if force:
        remove_existing_files(raw_dir)

    archive = download_competition(
        competition=competition,
        raw_dir=raw_dir,
    )

    extract_archive(
        archive=archive,
        raw_dir=raw_dir,
    )

    validate_download(raw_dir)

    archive.unlink()

    LOGGER.info(
        "Kaggle ingestion completed successfully."
    )

    for filename in sorted(EXPECTED_FILES):
        path = raw_dir / filename

        LOGGER.info(
            "Raw file=%s | size_mb=%.2f",
            filename,
            path.stat().st_size / (1024 * 1024),
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Download the Bank Marketing "
            "competition dataset from Kaggle."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Download the raw files again.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the ingestion command."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    args = parse_args()

    ingest(
        force=args.force,
    )


if __name__ == "__main__":
    main()