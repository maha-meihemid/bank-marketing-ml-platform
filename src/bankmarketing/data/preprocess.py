"""
Preprocess validated Bank Marketing raw datasets.

This module performs deterministic preprocessing only.
It does not fit statistical transformations or create ML features.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from bankmarketing.data.validate import (
    TARGET_COLUMN,
    validate_file,
)

LOGGER = logging.getLogger(__name__)

CATEGORICAL_COLUMNS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

INTEGER_COLUMNS = [
    "id",
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
]


def load_config(
    path: str = "configs/data.yaml",
) -> dict:
    """Load the data configuration."""
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as stream:
        return yaml.safe_load(stream)


def normalize_categorical_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize categorical string values."""
    dataframe = dataframe.copy()

    for column in CATEGORICAL_COLUMNS:
        dataframe[column] = (
            dataframe[column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

    return dataframe


def normalize_numeric_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize numeric column types."""
    dataframe = dataframe.copy()

    for column in INTEGER_COLUMNS:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        ).astype("int64")

    if TARGET_COLUMN in dataframe.columns:
        dataframe[TARGET_COLUMN] = pd.to_numeric(
            dataframe[TARGET_COLUMN],
            errors="raise",
        ).astype("int8")

    return dataframe


def preprocess_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Apply deterministic preprocessing transformations."""
    processed = dataframe.copy()

    processed = normalize_categorical_columns(
        processed
    )

    processed = normalize_numeric_columns(
        processed
    )

    return processed


def validate_processed_dataframe(
    dataframe: pd.DataFrame,
    include_target: bool,
) -> None:
    """Run post-preprocessing integrity checks."""
    if dataframe.empty:
        raise RuntimeError(
            "Processed dataframe is empty."
        )

    if dataframe["id"].duplicated().any():
        raise RuntimeError(
            "Processed dataframe contains duplicate ids."
        )

    if dataframe["id"].isna().any():
        raise RuntimeError(
            "Processed dataframe contains missing ids."
        )

    for column in CATEGORICAL_COLUMNS:
        if dataframe[column].isna().any():
            raise RuntimeError(
                f"Processed column '{column}' "
                "contains missing values."
            )

        empty_mask = (
            dataframe[column]
            .astype(str)
            .str.strip()
            .eq("")
        )

        if empty_mask.any():
            raise RuntimeError(
                f"Processed column '{column}' "
                "contains empty values."
            )

    if include_target:
        if TARGET_COLUMN not in dataframe.columns:
            raise RuntimeError(
                "Target column is missing "
                "from processed training data."
            )

        invalid_targets = (
            ~dataframe[TARGET_COLUMN]
            .isin([0, 1])
        )

        if invalid_targets.any():
            raise RuntimeError(
                "Processed training data contains "
                "invalid target values."
            )

    else:
        if TARGET_COLUMN in dataframe.columns:
            raise RuntimeError(
                "Kaggle test data must not contain "
                "the target column."
            )


def write_parquet(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write dataframe to Parquet atomically."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        ".parquet.tmp"
    )

    try:
        dataframe.to_parquet(
            temporary_path,
            index=False,
            engine="pyarrow",
        )

        temporary_path.replace(
            output_path
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def preprocess_file(
    input_path: Path,
    output_path: Path,
    include_target: bool,
    force: bool,
) -> None:
    """Validate, preprocess and persist one dataset."""
    if output_path.exists() and not force:
        LOGGER.info(
            "Processed file already exists=%s | "
            "use --force to rebuild it",
            output_path,
        )
        return

    LOGGER.info(
        "Validating source file=%s",
        input_path,
    )

    dataframe = validate_file(
        path=input_path,
        include_target=include_target,
    )

    raw_rows = len(dataframe)

    LOGGER.info(
        "Preprocessing file=%s",
        input_path.name,
    )

    processed = preprocess_dataframe(
        dataframe
    )

    validate_processed_dataframe(
        dataframe=processed,
        include_target=include_target,
    )

    if len(processed) != raw_rows:
        raise RuntimeError(
            "Preprocessing changed the number of rows."
        )

    write_parquet(
        dataframe=processed,
        output_path=output_path,
    )

    LOGGER.info(
        "Processed file written=%s | "
        "rows=%d | columns=%d | size_mb=%.2f",
        output_path,
        len(processed),
        len(processed.columns),
        output_path.stat().st_size
        / (1024 * 1024),
    )


def preprocess(
    force: bool = False,
) -> None:
    """Run preprocessing for train and Kaggle test data."""
    config = load_config()

    raw_dir = Path(
        config["paths"]["raw_dir"]
    )

    processed_dir = Path(
        config["paths"]["processed_dir"]
    )

    preprocess_file(
        input_path=raw_dir / "train.csv",
        output_path=(
            processed_dir / "train.parquet"
        ),
        include_target=True,
        force=force,
    )

    preprocess_file(
        input_path=raw_dir / "test.csv",
        output_path=(
            processed_dir
            / "kaggle_test.parquet"
        ),
        include_target=False,
        force=force,
    )

    LOGGER.info(
        "Preprocessing completed successfully."
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess the Bank Marketing "
            "raw datasets."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rebuild processed datasets "
            "even if they already exist."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the preprocessing CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    args = parse_args()

    preprocess(
        force=args.force,
    )


if __name__ == "__main__":
    main()