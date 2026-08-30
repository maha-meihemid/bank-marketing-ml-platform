"""
Unit tests for final model evaluation utilities.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from bankmarketing.evaluation.evaluate import (
    promote_candidate_model,
    save_metrics_report,
    validate_test_dataframe,
)
from bankmarketing.features.build_features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def valid_test_dataframe() -> pd.DataFrame:
    """Create a minimal valid final-evaluation dataframe."""
    dataframe = pd.DataFrame(
        {
            column: [0, 1]
            for column in FEATURE_COLUMNS
        }
    )

    categorical_columns = [
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

    for column in categorical_columns:
        dataframe[column] = [
            "unknown",
            "unknown",
        ]

    dataframe[
        TARGET_COLUMN
    ] = [
        0,
        1,
    ]

    return dataframe


def test_validate_test_dataframe() -> None:
    """Check valid final evaluation dataframe."""
    dataframe = (
        valid_test_dataframe()
    )

    validate_test_dataframe(
        dataframe
    )


def test_missing_column_fails() -> None:
    """Check missing feature rejection."""
    dataframe = (
        valid_test_dataframe()
    )

    dataframe = dataframe.drop(
        columns=[
            FEATURE_COLUMNS[0]
        ]
    )

    with pytest.raises(
        RuntimeError
    ):
        validate_test_dataframe(
            dataframe
        )


def test_invalid_target_fails() -> None:
    """Check invalid target rejection."""
    dataframe = (
        valid_test_dataframe()
    )

    dataframe.loc[
        0,
        TARGET_COLUMN,
    ] = 2

    with pytest.raises(
        RuntimeError
    ):
        validate_test_dataframe(
            dataframe
        )


def test_missing_value_fails() -> None:
    """Check missing feature rejection."""
    dataframe = (
        valid_test_dataframe()
    )

    dataframe.loc[
        0,
        FEATURE_COLUMNS[0],
    ] = None

    with pytest.raises(
        RuntimeError
    ):
        validate_test_dataframe(
            dataframe
        )


def test_save_metrics_report(
    tmp_path: Path,
) -> None:
    """Check final metric report persistence."""
    output_path = (
        tmp_path
        / "metrics.json"
    )

    metrics = {
        "roc_auc": 0.97,
        "pr_auc": 0.81,
        "precision": 0.77,
        "recall": 0.68,
        "f1": 0.72,
        "threshold": 0.5,
    }

    save_metrics_report(
        metrics=metrics,
        output_path=output_path,
    )

    assert output_path.exists()

    with open(
        output_path,
        "r",
        encoding="utf-8",
    ) as stream:
        saved = json.load(
            stream
        )

    assert (
        saved["roc_auc"]
        == 0.97
    )


def test_promote_candidate_model(
    tmp_path: Path,
) -> None:
    """Check candidate model promotion."""
    candidate_path = (
        tmp_path
        / "candidate.joblib"
    )

    final_path = (
        tmp_path
        / "final.joblib"
    )

    candidate_path.write_bytes(
        b"candidate-model"
    )

    promote_candidate_model(
        candidate_path=(
            candidate_path
        ),
        final_path=final_path,
    )

    assert final_path.exists()

    assert (
        final_path.read_bytes()
        == b"candidate-model"
    )