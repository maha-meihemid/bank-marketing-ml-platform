"""
Unit tests for baseline model training utilities.
"""

import numpy as np

from bankmarketing.features.build_features import (
    FEATURE_COLUMNS,
)
from bankmarketing.training.metrics import (
    compute_classification_metrics,
)
from bankmarketing.training.train import (
    RAW_FEATURE_COLUMNS,
    build_model,
    build_preprocessor,
    get_feature_specification,
)


def test_classification_metrics_are_computed() -> None:
    """Check classification metric computation."""
    y_true = np.array(
        [0, 0, 1, 1]
    )

    probabilities = np.array(
        [0.1, 0.4, 0.6, 0.9]
    )

    metrics = (
        compute_classification_metrics(
            y_true=y_true,
            probabilities=probabilities,
            threshold=0.5,
        )
    )

    assert metrics[
        "roc_auc"
    ] == 1.0

    assert metrics[
        "pr_auc"
    ] == 1.0

    assert metrics[
        "precision"
    ] == 1.0

    assert metrics[
        "recall"
    ] == 1.0

    assert metrics[
        "f1"
    ] == 1.0

    assert metrics[
        "threshold"
    ] == 0.5


def test_metrics_respect_threshold() -> None:
    """Check threshold-based metric behavior."""
    y_true = np.array(
        [0, 1, 1, 1]
    )

    probabilities = np.array(
        [0.1, 0.4, 0.6, 0.9]
    )

    metrics = (
        compute_classification_metrics(
            y_true=y_true,
            probabilities=probabilities,
            threshold=0.5,
        )
    )

    assert np.isclose(
        metrics["recall"],
        2 / 3,
    )


def test_raw_feature_specification() -> None:
    """Check raw feature configuration."""
    (
        feature_columns,
        categorical_columns,
        numeric_columns,
        binary_columns,
    ) = get_feature_specification(
        "raw"
    )

    assert (
        feature_columns
        == RAW_FEATURE_COLUMNS
    )

    assert len(
        categorical_columns
    ) == 9

    assert len(
        numeric_columns
    ) == 7

    assert binary_columns == []

    assert len(
        feature_columns
    ) == 16


def test_engineered_feature_specification() -> None:
    """Check engineered feature configuration."""
    (
        feature_columns,
        categorical_columns,
        numeric_columns,
        binary_columns,
    ) = get_feature_specification(
        "engineered"
    )

    assert (
        feature_columns
        == FEATURE_COLUMNS
    )

    assert (
        "age_group"
        in categorical_columns
    )

    assert (
        "log_duration"
        in numeric_columns
    )

    assert (
        "has_previous_contact"
        in binary_columns
    )

    assert len(
        feature_columns
    ) == 24


def test_preprocessor_creation() -> None:
    """Check sklearn preprocessor creation."""
    preprocessor = build_preprocessor(
        categorical_columns=[
            "job"
        ],
        numeric_columns=[
            "age"
        ],
        binary_columns=[
            "has_previous_contact"
        ],
    )

    transformer_names = [
        name
        for (
            name,
            _,
            _,
        ) in preprocessor.transformers
    ]

    assert (
        "categorical"
        in transformer_names
    )

    assert (
        "numeric"
        in transformer_names
    )

    assert (
        "binary"
        in transformer_names
    )


def test_logistic_regression_creation() -> None:
    """Check Logistic Regression model creation."""
    config = {
        "models": {
            "logistic_regression": {
                "max_iter": 100,
                "random_state": 42,
            }
        }
    }

    model = build_model(
        model_name=(
            "logistic_regression"
        ),
        config=config,
    )

    assert (
        model.max_iter
        == 100
    )


def test_random_forest_creation() -> None:
    """Check Random Forest model creation."""
    config = {
        "models": {
            "random_forest": {
                "n_estimators": 10,
                "random_state": 42,
                "n_jobs": 1,
            }
        }
    }

    model = build_model(
        model_name=(
            "random_forest"
        ),
        config=config,
    )

    assert (
        model.n_estimators
        == 10
    )


def test_xgboost_creation() -> None:
    """Check XGBoost model creation."""
    config = {
        "models": {
            "xgboost": {
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "n_jobs": 1,
                "eval_metric": "logloss",
                "tree_method": "hist",
            }
        }
    }

    model = build_model(
        model_name="xgboost",
        config=config,
    )

    assert (
        model.n_estimators
        == 10
    )

    assert (
        model.max_depth
        == 3
    )


def test_unknown_feature_set_fails() -> None:
    """Check invalid feature set rejection."""
    try:
        get_feature_specification(
            "invalid"
        )
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError."
    )