"""
Unit tests for feature engineering and dataset splitting.
"""

import numpy as np
import pandas as pd

from bankmarketing.features.build_features import (
    FEATURE_COLUMNS,
    add_engineered_features,
    build_model_dataframe,
    create_internal_splits,
)


def sample_dataframe(
    rows: int = 1000,
) -> pd.DataFrame:
    """Create a deterministic sample Bank Marketing dataframe."""
    rng = np.random.default_rng(42)

    target = np.zeros(
        rows,
        dtype=np.int8,
    )

    positive_count = int(
        rows * 0.12
    )

    target[:positive_count] = 1
    rng.shuffle(target)

    previous = rng.integers(
        0,
        5,
        size=rows,
    )

    pdays = np.where(
        previous > 0,
        rng.integers(
            1,
            500,
            size=rows,
        ),
        -1,
    )

    poutcome = np.where(
        previous > 0,
        "failure",
        "unknown",
    )

    return pd.DataFrame(
        {
            "id": np.arange(
                rows,
                dtype=np.int64,
            ),
            "age": rng.integers(
                18,
                80,
                size=rows,
            ),
            "job": ["technician"] * rows,
            "marital": ["married"] * rows,
            "education": ["secondary"] * rows,
            "default": ["no"] * rows,
            "balance": rng.integers(
                -1000,
                10000,
                size=rows,
            ),
            "housing": ["yes"] * rows,
            "loan": ["no"] * rows,
            "contact": ["cellular"] * rows,
            "day": rng.integers(
                1,
                32,
                size=rows,
            ),
            "month": ["may"] * rows,
            "duration": rng.integers(
                1,
                1000,
                size=rows,
            ),
            "campaign": rng.integers(
                1,
                10,
                size=rows,
            ),
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome,
            "y": target,
        }
    )


def test_engineered_features_are_created() -> None:
    """Check that all expected engineered features are created."""
    dataframe = sample_dataframe(
        rows=100,
    )

    featured = add_engineered_features(
        dataframe,
    )

    expected_columns = [
        "has_previous_contact",
        "is_previous_success",
        "is_cellular_contact",
        "balance_per_campaign",
        "duration_per_campaign",
        "log_duration",
        "log_balance_abs",
        "age_group",
    ]

    for column in expected_columns:
        assert column in featured.columns


def test_original_columns_are_preserved() -> None:
    """Check that feature engineering preserves source columns."""
    dataframe = sample_dataframe(
        rows=100,
    )

    original_columns = set(
        dataframe.columns,
    )

    featured = add_engineered_features(
        dataframe,
    )

    assert original_columns.issubset(
        set(featured.columns),
    )


def test_has_previous_contact() -> None:
    """Check previous contact indicator."""
    dataframe = sample_dataframe(
        rows=10,
    )

    dataframe.loc[
        0,
        "previous",
    ] = 0

    dataframe.loc[
        1,
        "previous",
    ] = 3

    featured = add_engineered_features(
        dataframe,
    )

    assert (
        featured.loc[
            0,
            "has_previous_contact",
        ]
        == 0
    )

    assert (
        featured.loc[
            1,
            "has_previous_contact",
        ]
        == 1
    )


def test_is_previous_success() -> None:
    """Check previous campaign success indicator."""
    dataframe = sample_dataframe(
        rows=10,
    )

    dataframe.loc[
        0,
        "poutcome",
    ] = "success"

    dataframe.loc[
        1,
        "poutcome",
    ] = "failure"

    featured = add_engineered_features(
        dataframe,
    )

    assert (
        featured.loc[
            0,
            "is_previous_success",
        ]
        == 1
    )

    assert (
        featured.loc[
            1,
            "is_previous_success",
        ]
        == 0
    )


def test_is_cellular_contact() -> None:
    """Check cellular contact indicator."""
    dataframe = sample_dataframe(
        rows=10,
    )

    dataframe.loc[
        0,
        "contact",
    ] = "cellular"

    dataframe.loc[
        1,
        "contact",
    ] = "telephone"

    featured = add_engineered_features(
        dataframe,
    )

    assert (
        featured.loc[
            0,
            "is_cellular_contact",
        ]
        == 1
    )

    assert (
        featured.loc[
            1,
            "is_cellular_contact",
        ]
        == 0
    )


def test_balance_per_campaign() -> None:
    """Check balance divided by campaign count."""
    dataframe = sample_dataframe(
        rows=10,
    )

    dataframe.loc[
        0,
        "balance",
    ] = 1000

    dataframe.loc[
        0,
        "campaign",
    ] = 4

    featured = add_engineered_features(
        dataframe,
    )

    assert (
        featured.loc[
            0,
            "balance_per_campaign",
        ]
        == 250
    )


def test_duration_per_campaign() -> None:
    """Check duration divided by campaign count."""
    dataframe = sample_dataframe(
        rows=10,
    )

    dataframe.loc[
        0,
        "duration",
    ] = 600

    dataframe.loc[
        0,
        "campaign",
    ] = 3

    featured = add_engineered_features(
        dataframe,
    )

    assert (
        featured.loc[
            0,
            "duration_per_campaign",
        ]
        == 200
    )


def test_log_duration() -> None:
    """Check logarithmic duration transformation."""
    dataframe = sample_dataframe(
        rows=10,
    )

    dataframe.loc[
        0,
        "duration",
    ] = 99

    featured = add_engineered_features(
        dataframe,
    )

    assert np.isclose(
        featured.loc[
            0,
            "log_duration",
        ],
        np.log1p(99),
    )


def test_log_balance_abs_with_positive_balance() -> None:
    """Check log balance transformation for positive values."""
    dataframe = sample_dataframe(
        rows=10,
    )

    dataframe.loc[
        0,
        "balance",
    ] = 999

    featured = add_engineered_features(
        dataframe,
    )

    assert np.isclose(
        featured.loc[
            0,
            "log_balance_abs",
        ],
        np.log1p(999),
    )


def test_log_balance_abs_with_negative_balance() -> None:
    """Check log balance transformation for negative values."""
    dataframe = sample_dataframe(
        rows=10,
    )

    dataframe.loc[
        0,
        "balance",
    ] = -999

    featured = add_engineered_features(
        dataframe,
    )

    assert np.isclose(
        featured.loc[
            0,
            "log_balance_abs",
        ],
        np.log1p(999),
    )


def test_age_group() -> None:
    """Check age bucket creation."""
    dataframe = sample_dataframe(
        rows=6,
    )

    dataframe["age"] = [
        20,
        30,
        40,
        50,
        60,
        70,
    ]

    featured = add_engineered_features(
        dataframe,
    )

    assert (
        featured["age_group"].tolist()
        == [
            "18_25",
            "26_35",
            "36_45",
            "46_55",
            "56_65",
            "66_plus",
        ]
    )


def test_build_model_dataframe_removes_id() -> None:
    """Check that technical id is excluded from ML features."""
    dataframe = sample_dataframe(
        rows=100,
    )

    featured = build_model_dataframe(
        dataframe=dataframe,
        include_target=True,
    )

    assert "id" not in featured.columns


def test_build_model_dataframe_preserves_target() -> None:
    """Check that target is preserved for labelled data."""
    dataframe = sample_dataframe(
        rows=100,
    )

    featured = build_model_dataframe(
        dataframe=dataframe,
        include_target=True,
    )

    assert "y" in featured.columns


def test_build_model_dataframe_excludes_target() -> None:
    """Check that target is absent from Kaggle test features."""
    dataframe = (
        sample_dataframe(
            rows=100,
        )
        .drop(
            columns=["y"],
        )
    )

    featured = build_model_dataframe(
        dataframe=dataframe,
        include_target=False,
    )

    assert "y" not in featured.columns


def test_model_feature_schema() -> None:
    """Check exact feature column order."""
    dataframe = (
        sample_dataframe(
            rows=100,
        )
        .drop(
            columns=["y"],
        )
    )

    featured = build_model_dataframe(
        dataframe=dataframe,
        include_target=False,
    )

    assert (
        list(featured.columns)
        == FEATURE_COLUMNS
    )


def test_internal_split_sizes() -> None:
    """Check 70/15/15 internal split sizes."""
    dataframe = sample_dataframe(
        rows=1000,
    )

    featured = build_model_dataframe(
        dataframe=dataframe,
        include_target=True,
    )

    train, validation, test = (
        create_internal_splits(
            dataframe=featured,
            test_size=0.15,
            validation_size=0.15,
            random_state=42,
        )
    )

    assert len(train) == 700
    assert len(validation) == 150
    assert len(test) == 150

    assert (
        len(train)
        + len(validation)
        + len(test)
        == len(featured)
    )


def test_internal_split_is_stratified() -> None:
    """Check that target prevalence is preserved across splits."""
    dataframe = sample_dataframe(
        rows=1000,
    )

    featured = build_model_dataframe(
        dataframe=dataframe,
        include_target=True,
    )

    train, validation, test = (
        create_internal_splits(
            dataframe=featured,
            test_size=0.15,
            validation_size=0.15,
            random_state=42,
        )
    )

    original_rate = (
        featured["y"].mean()
    )

    for split in [
        train,
        validation,
        test,
    ]:
        assert np.isclose(
            split["y"].mean(),
            original_rate,
            atol=0.01,
        )


def test_internal_split_is_reproducible() -> None:
    """Check deterministic splitting with fixed random state."""
    dataframe = sample_dataframe(
        rows=1000,
    )

    featured = build_model_dataframe(
        dataframe=dataframe,
        include_target=True,
    )

    first = create_internal_splits(
        dataframe=featured,
        test_size=0.15,
        validation_size=0.15,
        random_state=42,
    )

    second = create_internal_splits(
        dataframe=featured,
        test_size=0.15,
        validation_size=0.15,
        random_state=42,
    )

    for first_split, second_split in zip(
        first,
        second,
    ):
        pd.testing.assert_frame_equal(
            first_split,
            second_split,
        )


def test_feature_engineering_does_not_create_missing_values() -> None:
    """Check that engineered features do not introduce missing values."""
    dataframe = sample_dataframe(
        rows=1000,
    )

    featured = build_model_dataframe(
        dataframe=dataframe,
        include_target=True,
    )

    assert not featured.isna().any().any()


def test_binary_engineered_features_have_valid_values() -> None:
    """Check binary engineered feature domains."""
    dataframe = sample_dataframe(
        rows=1000,
    )

    featured = add_engineered_features(
        dataframe,
    )

    binary_columns = [
        "has_previous_contact",
        "is_previous_success",
        "is_cellular_contact",
    ]

    for column in binary_columns:
        assert set(
            featured[column].unique()
        ).issubset(
            {0, 1},
        )


def test_campaign_ratios_are_finite() -> None:
    """Check ratio features do not contain infinite values."""
    dataframe = sample_dataframe(
        rows=1000,
    )

    featured = add_engineered_features(
        dataframe,
    )

    assert np.isfinite(
        featured[
            "balance_per_campaign"
        ]
    ).all()

    assert np.isfinite(
        featured[
            "duration_per_campaign"
        ]
    ).all()