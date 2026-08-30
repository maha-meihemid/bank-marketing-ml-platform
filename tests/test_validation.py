"""
Unit tests for raw data validation.
"""

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from bankmarketing.data.validate import (
    build_schema,
    validate_ids,
)


def valid_train_dataframe() -> pd.DataFrame:
    """Create a minimal valid Bank Marketing dataframe."""

    return pd.DataFrame(
        {
            "id": [
                1,
                2,
            ],
            "age": [
                30,
                45,
            ],
            "job": [
                "technician",
                "management",
            ],
            "marital": [
                "single",
                "married",
            ],
            "education": [
                "secondary",
                "tertiary",
            ],
            "default": [
                "no",
                "no",
            ],
            "balance": [
                1000,
                -250,
            ],
            "housing": [
                "yes",
                "no",
            ],
            "loan": [
                "no",
                "yes",
            ],
            "contact": [
                "cellular",
                "telephone",
            ],
            "day": [
                10,
                20,
            ],
            "month": [
                "may",
                "jun",
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
                "unknown",
                "success",
            ],
            "y": [
                0,
                1,
            ],
        }
    )


def test_train_schema_accepts_valid_data() -> None:
    dataframe = valid_train_dataframe()

    schema = build_schema(
        include_target=True,
    )

    validated = schema.validate(
        dataframe,
        lazy=True,
    )

    assert len(validated) == 2


def test_test_schema_accepts_data_without_target() -> None:
    dataframe = (
        valid_train_dataframe()
        .drop(columns=["y"])
    )

    schema = build_schema(
        include_target=False,
    )

    validated = schema.validate(
        dataframe,
        lazy=True,
    )

    assert len(validated) == 2
    assert "y" not in validated.columns


def test_schema_rejects_invalid_target() -> None:
    dataframe = valid_train_dataframe()

    dataframe.loc[
        0,
        "y",
    ] = 2

    schema = build_schema(
        include_target=True,
    )

    with pytest.raises(
        SchemaErrors
    ):
        schema.validate(
            dataframe,
            lazy=True,
        )


def test_schema_rejects_invalid_age() -> None:
    dataframe = valid_train_dataframe()

    dataframe.loc[
        0,
        "age",
    ] = 15

    schema = build_schema(
        include_target=True,
    )

    with pytest.raises(
        SchemaErrors
    ):
        schema.validate(
            dataframe,
            lazy=True,
        )


def test_schema_rejects_invalid_day() -> None:
    dataframe = valid_train_dataframe()

    dataframe.loc[
        0,
        "day",
    ] = 40

    schema = build_schema(
        include_target=True,
    )

    with pytest.raises(
        SchemaErrors
    ):
        schema.validate(
            dataframe,
            lazy=True,
        )


def test_schema_rejects_invalid_duration() -> None:
    dataframe = valid_train_dataframe()

    dataframe.loc[
        0,
        "duration",
    ] = 0

    schema = build_schema(
        include_target=True,
    )

    with pytest.raises(
        SchemaErrors
    ):
        schema.validate(
            dataframe,
            lazy=True,
        )


def test_schema_accepts_negative_balance() -> None:
    dataframe = valid_train_dataframe()

    dataframe.loc[
        0,
        "balance",
    ] = -8019

    schema = build_schema(
        include_target=True,
    )

    validated = schema.validate(
        dataframe,
        lazy=True,
    )

    assert validated.loc[
        0,
        "balance",
    ] == -8019


def test_schema_accepts_pdays_minus_one() -> None:
    dataframe = valid_train_dataframe()

    dataframe.loc[
        0,
        "pdays",
    ] = -1

    schema = build_schema(
        include_target=True,
    )

    validated = schema.validate(
        dataframe,
        lazy=True,
    )

    assert validated.loc[
        0,
        "pdays",
    ] == -1


def test_schema_rejects_unknown_job_category() -> None:
    dataframe = valid_train_dataframe()

    dataframe.loc[
        0,
        "job",
    ] = "astronaut"

    schema = build_schema(
        include_target=True,
    )

    with pytest.raises(
        SchemaErrors
    ):
        schema.validate(
            dataframe,
            lazy=True,
        )


def test_schema_rejects_unknown_month() -> None:
    dataframe = valid_train_dataframe()

    dataframe.loc[
        0,
        "month",
    ] = "foo"

    schema = build_schema(
        include_target=True,
    )

    with pytest.raises(
        SchemaErrors
    ):
        schema.validate(
            dataframe,
            lazy=True,
        )


def test_duplicate_ids_are_rejected() -> None:
    dataframe = valid_train_dataframe()

    dataframe.loc[
        1,
        "id",
    ] = 1

    with pytest.raises(
        RuntimeError,
        match="duplicate ids",
    ):
        validate_ids(
            dataframe=dataframe,
            filename="train.csv",
        )