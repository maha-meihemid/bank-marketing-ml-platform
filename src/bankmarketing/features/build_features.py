"""
Build ML features and create internal train, validation and test splits.

The Kaggle competition test set is transformed separately because it
does not contain target labels and must not be used for final evaluation.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)

TARGET_COLUMN = "y"
ID_COLUMN = "id"

BASE_FEATURE_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]

ENGINEERED_FEATURE_COLUMNS = [
    "has_previous_contact",
    "is_previous_success",
    "is_cellular_contact",
    "balance_per_campaign",
    "duration_per_campaign",
    "log_duration",
    "log_balance_abs",
    "age_group",
]

FEATURE_COLUMNS = (
    BASE_FEATURE_COLUMNS
    + ENGINEERED_FEATURE_COLUMNS
)

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
    "age_group",
]

NUMERIC_COLUMNS = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "balance_per_campaign",
    "duration_per_campaign",
    "log_duration",
    "log_balance_abs",
]

BINARY_COLUMNS = [
    "has_previous_contact",
    "is_previous_success",
    "is_cellular_contact",
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


def add_engineered_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create deterministic engineered features."""
    dataframe = dataframe.copy()

    dataframe["has_previous_contact"] = (
        dataframe["previous"] > 0
    ).astype("int8")

    dataframe["is_previous_success"] = (
        dataframe["poutcome"] == "success"
    ).astype("int8")

    dataframe["is_cellular_contact"] = (
        dataframe["contact"] == "cellular"
    ).astype("int8")

    dataframe["balance_per_campaign"] = (
        dataframe["balance"]
        / dataframe["campaign"]
    )

    dataframe["duration_per_campaign"] = (
        dataframe["duration"]
        / dataframe["campaign"]
    )

    dataframe["log_duration"] = np.log1p(
        dataframe["duration"]
    )

    dataframe["log_balance_abs"] = np.log1p(
        np.abs(
            dataframe["balance"]
        )
    )

    dataframe["age_group"] = pd.cut(
        dataframe["age"],
        bins=[
            17,
            25,
            35,
            45,
            55,
            65,
            np.inf,
        ],
        labels=[
            "18_25",
            "26_35",
            "36_45",
            "46_55",
            "56_65",
            "66_plus",
        ],
        include_lowest=True,
    ).astype("string")

    return dataframe


def build_model_dataframe(
    dataframe: pd.DataFrame,
    include_target: bool,
) -> pd.DataFrame:
    """Build the final ML dataframe."""
    featured = add_engineered_features(
        dataframe
    )

    columns = FEATURE_COLUMNS.copy()

    if include_target:
        columns.append(
            TARGET_COLUMN
        )

    model_dataframe = featured[
        columns
    ].copy()

    return model_dataframe


def validate_feature_dataframe(
    dataframe: pd.DataFrame,
    include_target: bool,
) -> None:
    """Validate generated ML features."""
    if dataframe.empty:
        raise RuntimeError(
            "Feature dataframe is empty."
        )

    expected_columns = set(
        FEATURE_COLUMNS
    )

    if include_target:
        expected_columns.add(
            TARGET_COLUMN
        )

    actual_columns = set(
        dataframe.columns
    )

    missing_columns = sorted(
        expected_columns
        - actual_columns
    )

    unexpected_columns = sorted(
        actual_columns
        - expected_columns
    )

    if (
        missing_columns
        or unexpected_columns
    ):
        raise RuntimeError(
            "Invalid feature schema. "
            f"Missing={missing_columns} | "
            f"Unexpected={unexpected_columns}"
        )

    if ID_COLUMN in dataframe.columns:
        raise RuntimeError(
            "Technical id column must not "
            "be used as an ML feature."
        )

    missing_values = (
        dataframe
        .isna()
        .sum()
    )

    missing_values = (
        missing_values[
            missing_values > 0
        ]
    )

    if not missing_values.empty:
        raise RuntimeError(
            "Feature dataframe contains "
            f"missing values: "
            f"{missing_values.to_dict()}"
        )

    numeric_columns = [
        column
        for column in (
            NUMERIC_COLUMNS
            + BINARY_COLUMNS
        )
        if column in dataframe.columns
    ]

    numeric_values = dataframe[
        numeric_columns
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        numeric_values
    ).all():
        raise RuntimeError(
            "Feature dataframe contains "
            "non-finite numeric values."
        )

    for column in BINARY_COLUMNS:
        invalid_values = (
            ~dataframe[column]
            .isin([0, 1])
        )

        if invalid_values.any():
            raise RuntimeError(
                f"Binary feature '{column}' "
                "contains invalid values."
            )

    if include_target:
        if TARGET_COLUMN not in dataframe.columns:
            raise RuntimeError(
                "Target column is missing."
            )

        invalid_target = (
            ~dataframe[
                TARGET_COLUMN
            ].isin([0, 1])
        )

        if invalid_target.any():
            raise RuntimeError(
                "Feature dataframe contains "
                "invalid target values."
            )

    else:
        if TARGET_COLUMN in dataframe.columns:
            raise RuntimeError(
                "Unlabelled feature dataframe "
                "must not contain target."
            )


def create_internal_splits(
    dataframe: pd.DataFrame,
    test_size: float,
    validation_size: float,
    random_state: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create stratified train, validation and test splits."""
    if test_size <= 0:
        raise ValueError(
            "test_size must be positive."
        )

    if validation_size <= 0:
        raise ValueError(
            "validation_size must be positive."
        )

    if (
        test_size
        + validation_size
        >= 1
    ):
        raise ValueError(
            "test_size and validation_size "
            "must sum to less than 1."
        )

    train_validation, test = (
        train_test_split(
            dataframe,
            test_size=test_size,
            random_state=random_state,
            stratify=dataframe[
                TARGET_COLUMN
            ],
        )
    )

    validation_relative_size = (
        validation_size
        / (1 - test_size)
    )

    train, validation = (
        train_test_split(
            train_validation,
            test_size=(
                validation_relative_size
            ),
            random_state=random_state,
            stratify=train_validation[
                TARGET_COLUMN
            ],
        )
    )

    train = train.reset_index(
        drop=True
    )

    validation = (
        validation.reset_index(
            drop=True
        )
    )

    test = test.reset_index(
        drop=True
    )

    return (
        train,
        validation,
        test,
    )


def validate_splits(
    original: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Validate internal dataset split integrity."""
    total_rows = (
        len(train)
        + len(validation)
        + len(test)
    )

    if total_rows != len(original):
        raise RuntimeError(
            "Split row count does not "
            "match source row count."
        )

    original_rate = float(
        original[
            TARGET_COLUMN
        ].mean()
    )

    split_dataframes = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    for name, dataframe in (
        split_dataframes.items()
    ):
        if dataframe.empty:
            raise RuntimeError(
                f"Split '{name}' is empty."
            )

        positive_rate = float(
            dataframe[
                TARGET_COLUMN
            ].mean()
        )

        if abs(
            positive_rate
            - original_rate
        ) > 0.01:
            raise RuntimeError(
                f"Split '{name}' target "
                "distribution differs too much "
                "from source distribution."
            )


def log_split_statistics(
    name: str,
    dataframe: pd.DataFrame,
) -> None:
    """Log split size and target distribution."""
    positive_count = int(
        dataframe[
            TARGET_COLUMN
        ].sum()
    )

    negative_count = int(
        len(dataframe)
        - positive_count
    )

    positive_rate = float(
        dataframe[
            TARGET_COLUMN
        ].mean()
    )

    LOGGER.info(
        "Split=%s | rows=%d | "
        "positive=%d | negative=%d | "
        "positive_rate=%.4f",
        name,
        len(dataframe),
        positive_count,
        negative_count,
        positive_rate,
    )


def write_parquet(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a dataframe to Parquet atomically."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        output_path.with_suffix(
            ".parquet.tmp"
        )
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


def expected_outputs_exist(
    features_dir: Path,
) -> bool:
    """Check whether all expected feature files exist."""
    expected_files = [
        "train.parquet",
        "validation.parquet",
        "test.parquet",
        "kaggle_test.parquet",
    ]

    return all(
        (
            features_dir
            / filename
        ).exists()
        and (
            features_dir
            / filename
        ).stat().st_size > 0
        for filename in expected_files
    )


def build_features(
    force: bool = False,
) -> None:
    """Run feature engineering and dataset splitting."""
    config = load_config()

    processed_dir = Path(
        config["paths"][
            "processed_dir"
        ]
    )

    features_dir = Path(
        config["paths"][
            "features_dir"
        ]
    )

    split_config = config[
        "split"
    ]

    if (
        expected_outputs_exist(
            features_dir
        )
        and not force
    ):
        LOGGER.info(
            "Feature datasets already exist. "
            "Use --force to rebuild them."
        )
        return

    train_path = (
        processed_dir
        / "train.parquet"
    )

    kaggle_test_path = (
        processed_dir
        / "kaggle_test.parquet"
    )

    if not train_path.exists():
        raise FileNotFoundError(
            "Processed training data "
            f"not found: {train_path}"
        )

    if not kaggle_test_path.exists():
        raise FileNotFoundError(
            "Processed Kaggle test data "
            f"not found: {kaggle_test_path}"
        )

    LOGGER.info(
        "Loading processed training "
        "data=%s",
        train_path,
    )

    processed_train = (
        pd.read_parquet(
            train_path
        )
    )

    LOGGER.info(
        "Processed training data | "
        "rows=%d | columns=%d",
        len(processed_train),
        len(processed_train.columns),
    )

    LOGGER.info(
        "Building engineered features."
    )

    model_data = (
        build_model_dataframe(
            dataframe=processed_train,
            include_target=True,
        )
    )

    validate_feature_dataframe(
        dataframe=model_data,
        include_target=True,
    )

    LOGGER.info(
        "Feature dataframe created | "
        "rows=%d | features=%d",
        len(model_data),
        len(FEATURE_COLUMNS),
    )

    train, validation, test = (
        create_internal_splits(
            dataframe=model_data,
            test_size=float(
                split_config[
                    "test_size"
                ]
            ),
            validation_size=float(
                split_config[
                    "validation_size"
                ]
            ),
            random_state=int(
                split_config[
                    "random_state"
                ]
            ),
        )
    )

    validate_splits(
        original=model_data,
        train=train,
        validation=validation,
        test=test,
    )

    log_split_statistics(
        name="train",
        dataframe=train,
    )

    log_split_statistics(
        name="validation",
        dataframe=validation,
    )

    log_split_statistics(
        name="test",
        dataframe=test,
    )

    LOGGER.info(
        "Loading processed Kaggle "
        "test data=%s",
        kaggle_test_path,
    )

    processed_kaggle_test = (
        pd.read_parquet(
            kaggle_test_path
        )
    )

    kaggle_test = (
        build_model_dataframe(
            dataframe=(
                processed_kaggle_test
            ),
            include_target=False,
        )
    )

    validate_feature_dataframe(
        dataframe=kaggle_test,
        include_target=False,
    )

    features_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_dataframes = {
        "train.parquet": train,
        "validation.parquet": (
            validation
        ),
        "test.parquet": test,
        "kaggle_test.parquet": (
            kaggle_test
        ),
    }

    for (
        filename,
        dataframe,
    ) in output_dataframes.items():
        output_path = (
            features_dir
            / filename
        )

        write_parquet(
            dataframe=dataframe,
            output_path=output_path,
        )

        LOGGER.info(
            "Feature file written=%s | "
            "rows=%d | columns=%d | "
            "size_mb=%.2f",
            output_path,
            len(dataframe),
            len(dataframe.columns),
            output_path.stat().st_size
            / (1024 * 1024),
        )

    LOGGER.info(
        "Feature engineering completed "
        "successfully."
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build Bank Marketing ML "
            "features and internal splits."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rebuild feature datasets "
            "even if they already exist."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the feature engineering CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    args = parse_args()

    build_features(
        force=args.force,
    )


if __name__ == "__main__":
    main()