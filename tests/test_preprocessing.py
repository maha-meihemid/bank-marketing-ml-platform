"""
Unit tests for deterministic preprocessing.
"""

import pandas as pd
import pytest

from bankmarketing.data.preprocess import (
    preprocess_dataframe,
    validate_processed_dataframe,
)


def raw_dataframe(
    include_target: bool = True,
) -> pd.DataFrame:
    """Create a minimal raw dataframe."""
    data = {
        "id": [
            1,
            2,
        ],
        "age": [
            30,
            45,
        ],
        "job": [
            " Technician ",
            "MANAGEMENT",
        ],
        "marital": [
            " Single ",
            "MARRIED",
        ],
        "education": [
            " Secondary ",
            "TERTIARY",
        ],
        "default": [
            " No ",
            "NO",
        ],
        "balance": [
            1000,
            -250,
        ],
        "housing": [
            " Yes ",
            "NO",
        ],
        "loan": [
            " No ",
            "YES",
        ],
        "contact": [
            " Cellular ",
            "TELEPHONE",
        ],
        "day": [
            10,
            20,
        ],
        "month": [
            " May ",
            "JUN",
        ],
        "duration": [
            120,
            300,
        ],
        "campaign": [
            1,
            2,
        ],
        "pdays": [
            -1,
            10,
        ],
        "previous": [
            0,
            1,
        ],
        "poutcome": [
            " Unknown ",
            "SUCCESS",
        ],
    }

    if include_target:
        data["y"] = [
            0,
            1,
        ]

    return pd.DataFrame(data)


def test_preprocess_normalizes_categories() -> None:
    dataframe = raw_dataframe()

    processed = preprocess_dataframe(
        dataframe
    )

    assert processed.loc[0, "job"] == (
        "technician"
    )

    assert processed.loc[1, "job"] == (
        "management"
    )

    assert processed.loc[0, "month"] == "may"

    assert (
        processed.loc[1, "poutcome"]
        == "success"
    )


def test_preprocess_preserves_row_count() -> None:
    dataframe = raw_dataframe()

    processed = preprocess_dataframe(
        dataframe
    )

    assert len(processed) == len(dataframe)


def test_preprocess_preserves_columns() -> None:
    dataframe = raw_dataframe()

    processed = preprocess_dataframe(
        dataframe
    )

    assert list(processed.columns) == list(
        dataframe.columns
    )


def test_target_is_int8() -> None:
    dataframe = raw_dataframe()

    processed = preprocess_dataframe(
        dataframe
    )

    assert str(
        processed["y"].dtype
    ) == "int8"


def test_integer_columns_are_int64() -> None:
    dataframe = raw_dataframe()

    processed = preprocess_dataframe(
        dataframe
    )

    assert str(
        processed["age"].dtype
    ) == "int64"

    assert str(
        processed["balance"].dtype
    ) == "int64"


def test_unknown_category_is_preserved() -> None:
    dataframe = raw_dataframe()

    processed = preprocess_dataframe(
        dataframe
    )

    assert (
        processed.loc[0, "poutcome"]
        == "unknown"
    )


def test_negative_balance_is_preserved() -> None:
    dataframe = raw_dataframe()

    processed = preprocess_dataframe(
        dataframe
    )

    assert processed.loc[
        1,
        "balance",
    ] == -250


def test_pdays_minus_one_is_preserved() -> None:
    dataframe = raw_dataframe()

    processed = preprocess_dataframe(
        dataframe
    )

    assert processed.loc[
        0,
        "pdays",
    ] == -1


def test_validation_rejects_duplicate_ids() -> None:
    dataframe = raw_dataframe()

    dataframe.loc[
        1,
        "id",
    ] = 1

    processed = preprocess_dataframe(
        dataframe
    )

    with pytest.raises(
        RuntimeError,
        match="duplicate ids",
    ):
        validate_processed_dataframe(
            dataframe=processed,
            include_target=True,
        )


def test_kaggle_test_does_not_require_target() -> None:
    dataframe = raw_dataframe(
        include_target=False
    )

    processed = preprocess_dataframe(
        dataframe
    )

    validate_processed_dataframe(
        dataframe=processed,
        include_target=False,
    )

    assert "y" not in processed.columns


def test_training_data_requires_target() -> None:
    dataframe = raw_dataframe(
        include_target=False
    )

    processed = preprocess_dataframe(
        dataframe
    )

    with pytest.raises(
        RuntimeError,
        match="Target column is missing",
    ):
        validate_processed_dataframe(
            dataframe=processed,
            include_target=True,
        )