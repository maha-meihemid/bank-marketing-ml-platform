"""
Final evaluation of the tuned Bank Marketing model.

This stage evaluates the selected candidate exactly once on the
internal test dataset. No hyperparameter tuning or model selection
must be performed after observing these results.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

from bankmarketing.features.build_features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from bankmarketing.training.metrics import (
    compute_classification_metrics,
)
from bankmarketing.training.train import (
    XGBOOST_SKOPS_TRUSTED_TYPES,
)

LOGGER = logging.getLogger(__name__)


def load_config(
    path: str = "configs/model.yaml",
) -> dict:
    """Load model configuration."""
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as stream:
        return yaml.safe_load(stream)


def validate_test_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    """Validate final evaluation dataset."""
    required_columns = set(
        FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Test dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise RuntimeError(
            "Test dataset is empty."
        )

    if not dataframe[
        TARGET_COLUMN
    ].isin([0, 1]).all():
        raise RuntimeError(
            "Test target contains invalid values."
        )

    if dataframe[
        FEATURE_COLUMNS
    ].isna().any().any():
        raise RuntimeError(
            "Test features contain missing values."
        )


def save_metrics_report(
    metrics: dict[str, float],
    output_path: Path,
) -> None:
    """Persist final evaluation metrics as JSON."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        output_path.with_suffix(
            ".json.tmp"
        )
    )

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            metrics,
            stream,
            indent=2,
            sort_keys=True,
        )

    temporary_path.replace(
        output_path
    )


def promote_candidate_model(
    candidate_path: Path,
    final_path: Path,
) -> None:
    """Copy the validated candidate to the final model artifact."""
    final_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        final_path.with_suffix(
            ".joblib.tmp"
        )
    )

    shutil.copy2(
        candidate_path,
        temporary_path,
    )

    temporary_path.replace(
        final_path
    )


def evaluate_candidate() -> dict[str, float]:
    """Evaluate the tuned candidate on the internal test dataset."""
    config = load_config()

    test_path = Path(
        "data/features/test.parquet"
    )

    candidate_path = Path(
        "artifacts/xgboost_candidate.joblib"
    )

    final_model_path = Path(
        "artifacts/bank_marketing_model.joblib"
    )

    metrics_path = Path(
        "artifacts/final_test_metrics.json"
    )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {test_path}"
        )

    if not candidate_path.exists():
        raise FileNotFoundError(
            "Candidate model not found: "
            f"{candidate_path}"
        )

    LOGGER.info(
        "Loading final test data=%s",
        test_path,
    )

    test_dataframe = pd.read_parquet(
        test_path
    )

    validate_test_dataframe(
        test_dataframe
    )

    LOGGER.info(
        "Loading candidate model=%s",
        candidate_path,
    )

    candidate = joblib.load(
        candidate_path
    )

    if not isinstance(
        candidate,
        Pipeline,
    ):
        raise TypeError(
            "Candidate artifact is not "
            "an sklearn Pipeline."
        )

    x_test = test_dataframe[
        FEATURE_COLUMNS
    ]

    y_test = test_dataframe[
        TARGET_COLUMN
    ]

    LOGGER.info(
        "Final evaluation | "
        "test_rows=%d | "
        "features=%d | "
        "positive_rate=%.6f",
        len(x_test),
        len(FEATURE_COLUMNS),
        y_test.mean(),
    )

    probabilities = (
        candidate.predict_proba(
            x_test
        )[:, 1]
    )

    metrics = (
        compute_classification_metrics(
            y_true=y_test.to_numpy(),
            probabilities=probabilities,
            threshold=0.5,
        )
    )

    mlflow.set_tracking_uri(
        config[
            "mlflow"
        ][
            "tracking_uri"
        ]
    )

    mlflow.set_experiment(
        config[
            "mlflow"
        ][
            "evaluation_experiment_name"
        ]
    )

    with mlflow.start_run(
        run_name=(
            "xgboost_final_test_evaluation"
        )
    ):
        mlflow.set_tag(
            "model_name",
            "xgboost",
        )

        mlflow.set_tag(
            "feature_set",
            "engineered",
        )

        mlflow.set_tag(
            "stage",
            "final_evaluation",
        )

        mlflow.set_tag(
            "dataset",
            "internal_test",
        )

        mlflow.log_param(
            "feature_count",
            len(FEATURE_COLUMNS),
        )

        mlflow.log_param(
            "test_rows",
            len(x_test),
        )

        mlflow.log_param(
            "decision_threshold",
            0.5,
        )

        for (
            metric_name,
            metric_value,
        ) in metrics.items():
            mlflow.log_metric(
                f"test_{metric_name}",
                float(metric_value),
            )

        mlflow.sklearn.log_model(
            sk_model=candidate,
            name="model",
            skops_trusted_types=(
                XGBOOST_SKOPS_TRUSTED_TYPES
            ),
        )

    save_metrics_report(
        metrics=metrics,
        output_path=metrics_path,
    )

    promote_candidate_model(
        candidate_path=candidate_path,
        final_path=final_model_path,
    )

    LOGGER.info(
        "FINAL TEST RESULTS | "
        "ROC-AUC=%.6f | "
        "PR-AUC=%.6f | "
        "precision=%.6f | "
        "recall=%.6f | "
        "F1=%.6f | "
        "threshold=%.2f",
        metrics["roc_auc"],
        metrics["pr_auc"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["threshold"],
    )

    LOGGER.info(
        "Final metrics saved to %s",
        metrics_path,
    )

    LOGGER.info(
        "Validated model promoted to %s",
        final_model_path,
    )

    LOGGER.info(
        "Final evaluation completed. "
        "Do not tune the model using "
        "internal test results."
    )

    return metrics


def main() -> None:
    """Run final model evaluation."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    evaluate_candidate()


if __name__ == "__main__":
    main()