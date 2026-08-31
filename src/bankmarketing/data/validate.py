"""
Validate raw Bank Marketing datasets using Pandera.

This module validates the raw data contract only.
It does not clean or transform the source data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import yaml

LOGGER = logging.getLogger(__name__)

TARGET_COLUMN = "y"

JOB_VALUES = [
    "admin.",
    "blue-collar",
    "entrepreneur",
    "housemaid",
    "management",
    "retired",
    "self-employed",
    "services",
    "student",
    "technician",
    "unemployed",
    "unknown",
]

MARITAL_VALUES = [
    "divorced",
    "married",
    "single",
]

EDUCATION_VALUES = [
    "primary",
    "secondary",
    "tertiary",
    "unknown",
]

YES_NO_VALUES = [
    "no",
    "yes",
]

CONTACT_VALUES = [
    "cellular",
    "telephone",
    "unknown",
]

MONTH_VALUES = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
]

POUTCOME_VALUES = [
    "failure",
    "other",
    "success",
    "unknown",
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


def build_schema(
    include_target: bool,
) -> pa.DataFrameSchema:
    """Build the raw Bank Marketing Pandera schema."""

    columns = {
        "id": pa.Column(
            int,
            checks=pa.Check.ge(0),
            nullable=False,
            coerce=True,
        ),
        "age": pa.Column(
            int,
            checks=[
                pa.Check.ge(18),
                pa.Check.le(120),
            ],
            nullable=False,
            coerce=True,
        ),
        "job": pa.Column(
            str,
            checks=pa.Check.isin(JOB_VALUES),
            nullable=False,
            coerce=True,
        ),
        "marital": pa.Column(
            str,
            checks=pa.Check.isin(
                MARITAL_VALUES
            ),
            nullable=False,
            coerce=True,
        ),
        "education": pa.Column(
            str,
            checks=pa.Check.isin(
                EDUCATION_VALUES
            ),
            nullable=False,
            coerce=True,
        ),
        "default": pa.Column(
            str,
            checks=pa.Check.isin(
                YES_NO_VALUES
            ),
            nullable=False,
            coerce=True,
        ),
        "balance": pa.Column(
            float,
            nullable=False,
            coerce=True,
        ),
        "housing": pa.Column(
            str,
            checks=pa.Check.isin(
                YES_NO_VALUES
            ),
            nullable=False,
            coerce=True,
        ),
        "loan": pa.Column(
            str,
            checks=pa.Check.isin(
                YES_NO_VALUES
            ),
            nullable=False,
            coerce=True,
        ),
        "contact": pa.Column(
            str,
            checks=pa.Check.isin(
                CONTACT_VALUES
            ),
            nullable=False,
            coerce=True,
        ),
        "day": pa.Column(
            int,
            checks=[
                pa.Check.ge(1),
                pa.Check.le(31),
            ],
            nullable=False,
            coerce=True,
        ),
        "month": pa.Column(
            str,
            checks=pa.Check.isin(
                MONTH_VALUES
            ),
            nullable=False,
            coerce=True,
        ),
        "duration": pa.Column(
            float,
            checks=pa.Check.ge(1),
            nullable=False,
            coerce=True,
        ),
        "campaign": pa.Column(
            int,
            checks=pa.Check.ge(1),
            nullable=False,
            coerce=True,
        ),
        "pdays": pa.Column(
            int,
            checks=pa.Check.ge(-1),
            nullable=False,
            coerce=True,
        ),
        "previous": pa.Column(
            int,
            checks=pa.Check.ge(0),
            nullable=False,
            coerce=True,
        ),
        "poutcome": pa.Column(
            str,
            checks=pa.Check.isin(
                POUTCOME_VALUES
            ),
            nullable=False,
            coerce=True,
        ),
    }

    if include_target:
        columns[TARGET_COLUMN] = pa.Column(
            int,
            checks=pa.Check.isin([0, 1]),
            nullable=False,
            coerce=True,
        )

    return pa.DataFrameSchema(
        columns=columns,
        strict=True,
        coerce=True,
    )


def validate_ids(
    dataframe: pd.DataFrame,
    filename: str,
) -> None:
    """Validate row identifiers."""

    duplicate_count = int(
        dataframe["id"].duplicated().sum()
    )

    if duplicate_count > 0:
        raise RuntimeError(
            f"{filename} contains "
            f"{duplicate_count} duplicate ids."
        )


def validate_file(
    path: Path,
    include_target: bool,
) -> pd.DataFrame:
    """Validate one raw CSV file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {path}"
        )

    LOGGER.info(
        "Loading raw file=%s",
        path,
    )

    dataframe = pd.read_csv(path)

    LOGGER.info(
        "Raw file=%s | rows=%d | columns=%d",
        path.name,
        len(dataframe),
        len(dataframe.columns),
    )

    schema = build_schema(
        include_target=include_target,
    )

    LOGGER.info(
        "Running Pandera validation for file=%s",
        path.name,
    )

    validated = schema.validate(
        dataframe,
        lazy=True,
    )

    validate_ids(
        dataframe=validated,
        filename=path.name,
    )

    LOGGER.info(
        "Validation successful | file=%s",
        path.name,
    )

    return validated


def main() -> None:
    """Validate all raw competition datasets."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    config = load_config()

    raw_dir = Path(
        config["paths"]["raw_dir"]
    )

    train_path = (
        raw_dir / "train.csv"
    )

    test_path = (
        raw_dir / "test.csv"
    )

    train = validate_file(
        path=train_path,
        include_target=True,
    )

    test = validate_file(
        path=test_path,
        include_target=False,
    )

    positive_rate = float(
        train[TARGET_COLUMN].mean()
    )

    LOGGER.info(
        "Train positive rate=%.4f",
        positive_rate,
    )

    LOGGER.info(
        "Validated train rows=%d | test rows=%d",
        len(train),
        len(test),
    )

    LOGGER.info(
        "Raw data validation completed successfully."
    )


if __name__ == "__main__":
    main()