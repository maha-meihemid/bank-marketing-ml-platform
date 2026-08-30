"""
Train baseline Bank Marketing classification models.

This module compares raw features against the complete engineered
feature set using Logistic Regression, Random Forest and XGBoost.

Model selection is performed exclusively on the validation dataset.
The internal test dataset is not accessed during baseline training.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from xgboost import XGBClassifier

from bankmarketing.features.build_features import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)
from bankmarketing.training.metrics import (
    compute_classification_metrics,
)

LOGGER = logging.getLogger(__name__)

RAW_CATEGORICAL_COLUMNS = [
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

RAW_NUMERIC_COLUMNS = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
]

RAW_FEATURE_COLUMNS = (
    RAW_CATEGORICAL_COLUMNS
    + RAW_NUMERIC_COLUMNS
)

ENGINEERED_BINARY_COLUMNS = [
    "has_previous_contact",
    "is_previous_success",
    "is_cellular_contact",
]

MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "xgboost",
]

FEATURE_SETS = [
    "raw",
    "engineered",
]

XGBOOST_SKOPS_TRUSTED_TYPES = [
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
]


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


def get_feature_specification(
    feature_set: str,
) -> tuple[
    list[str],
    list[str],
    list[str],
    list[str],
]:
    """Return feature groups for the requested feature set."""
    if feature_set == "raw":
        return (
            RAW_FEATURE_COLUMNS,
            RAW_CATEGORICAL_COLUMNS,
            RAW_NUMERIC_COLUMNS,
            [],
        )

    if feature_set == "engineered":
        return (
            FEATURE_COLUMNS,
            CATEGORICAL_COLUMNS,
            NUMERIC_COLUMNS,
            ENGINEERED_BINARY_COLUMNS,
        )

    raise ValueError(
        f"Unknown feature set: {feature_set}"
    )


def build_preprocessor(
    categorical_columns: list[str],
    numeric_columns: list[str],
    binary_columns: list[str],
) -> ColumnTransformer:
    """Build the sklearn preprocessing pipeline."""
    transformers = []

    if categorical_columns:
        categorical_transformer = (
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
            )
        )

        transformers.append(
            (
                "categorical",
                categorical_transformer,
                categorical_columns,
            )
        )

    if numeric_columns:
        numeric_transformer = (
            StandardScaler()
        )

        transformers.append(
            (
                "numeric",
                numeric_transformer,
                numeric_columns,
            )
        )

    if binary_columns:
        transformers.append(
            (
                "binary",
                "passthrough",
                binary_columns,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def build_model(
    model_name: str,
    config: dict,
):
    """Create a baseline classifier."""
    model_config = config[
        "models"
    ]

    if model_name == "logistic_regression":
        params = model_config[
            "logistic_regression"
        ]

        return LogisticRegression(
            max_iter=int(
                params["max_iter"]
            ),
            random_state=int(
                params["random_state"]
            ),
            solver="saga",
        )

    if model_name == "random_forest":
        params = model_config[
            "random_forest"
        ]

        return RandomForestClassifier(
            n_estimators=int(
                params["n_estimators"]
            ),
            random_state=int(
                params["random_state"]
            ),
            n_jobs=int(
                params["n_jobs"]
            ),
        )

    if model_name == "xgboost":
        params = model_config[
            "xgboost"
        ]

        return XGBClassifier(
            n_estimators=int(
                params["n_estimators"]
            ),
            max_depth=int(
                params["max_depth"]
            ),
            learning_rate=float(
                params["learning_rate"]
            ),
            subsample=float(
                params["subsample"]
            ),
            colsample_bytree=float(
                params["colsample_bytree"]
            ),
            random_state=int(
                params["random_state"]
            ),
            n_jobs=int(
                params["n_jobs"]
            ),
            eval_metric=params[
                "eval_metric"
            ],
            tree_method=params[
                "tree_method"
            ],
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def build_pipeline(
    model_name: str,
    feature_set: str,
    config: dict,
) -> tuple[
    Pipeline,
    list[str],
]:
    """Build complete preprocessing and model pipeline."""
    (
        feature_columns,
        categorical_columns,
        numeric_columns,
        binary_columns,
    ) = get_feature_specification(
        feature_set
    )

    preprocessor = build_preprocessor(
        categorical_columns=(
            categorical_columns
        ),
        numeric_columns=(
            numeric_columns
        ),
        binary_columns=(
            binary_columns
        ),
    )

    model = build_model(
        model_name=model_name,
        config=config,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return (
        pipeline,
        feature_columns,
    )


def get_skops_trusted_types(
    model_name: str,
) -> list[str] | None:
    """Return trusted skops types required by the model."""
    if model_name == "xgboost":
        return (
            XGBOOST_SKOPS_TRUSTED_TYPES
        )

    return None


def log_model_artifact(
    pipeline: Pipeline,
    model_name: str,
) -> None:
    """Log the trained sklearn pipeline to MLflow."""
    trusted_types = (
        get_skops_trusted_types(
            model_name
        )
    )

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="model",
        skops_trusted_types=(
            trusted_types
        ),
    )


def train_single_model(
    model_name: str,
    feature_set: str,
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    config: dict,
) -> dict[str, float | str]:
    """Train and evaluate one baseline model."""
    (
        pipeline,
        feature_columns,
    ) = build_pipeline(
        model_name=model_name,
        feature_set=feature_set,
        config=config,
    )

    x_train = train_dataframe[
        feature_columns
    ]

    y_train = train_dataframe[
        TARGET_COLUMN
    ]

    x_validation = (
        validation_dataframe[
            feature_columns
        ]
    )

    y_validation = (
        validation_dataframe[
            TARGET_COLUMN
        ]
    )

    run_name = (
        f"{model_name}_{feature_set}"
    )

    LOGGER.info(
        "Training run=%s | "
        "train_rows=%d | "
        "validation_rows=%d | "
        "features=%d",
        run_name,
        len(x_train),
        len(x_validation),
        len(feature_columns),
    )

    with mlflow.start_run(
        run_name=run_name,
    ):
        mlflow.set_tag(
            "model_name",
            model_name,
        )

        mlflow.set_tag(
            "feature_set",
            feature_set,
        )

        mlflow.set_tag(
            "stage",
            "baseline",
        )

        mlflow.log_param(
            "model_name",
            model_name,
        )

        mlflow.log_param(
            "feature_set",
            feature_set,
        )

        mlflow.log_param(
            "feature_count",
            len(feature_columns),
        )

        mlflow.log_param(
            "train_rows",
            len(x_train),
        )

        mlflow.log_param(
            "validation_rows",
            len(x_validation),
        )

        mlflow.log_param(
            "decision_threshold",
            0.5,
        )

        start_time = (
            time.perf_counter()
        )

        pipeline.fit(
            x_train,
            y_train,
        )

        training_seconds = (
            time.perf_counter()
            - start_time
        )

        probabilities = (
            pipeline.predict_proba(
                x_validation
            )[:, 1]
        )

        metrics = (
            compute_classification_metrics(
                y_true=(
                    y_validation.to_numpy()
                ),
                probabilities=probabilities,
                threshold=0.5,
            )
        )

        metrics[
            "training_seconds"
        ] = float(
            training_seconds
        )

        mlflow.log_metrics(
            metrics
        )

        log_model_artifact(
            pipeline=pipeline,
            model_name=model_name,
        )

        LOGGER.info(
            "Run=%s | "
            "ROC-AUC=%.6f | "
            "PR-AUC=%.6f | "
            "precision=%.6f | "
            "recall=%.6f | "
            "F1=%.6f | "
            "training_seconds=%.2f",
            run_name,
            metrics["roc_auc"],
            metrics["pr_auc"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            training_seconds,
        )

    return {
        "run_name": run_name,
        "model_name": model_name,
        "feature_set": feature_set,
        **metrics,
    }


def print_results(
    results: list[
        dict[str, float | str]
    ],
) -> None:
    """Print baseline model comparison."""
    results_dataframe = (
        pd.DataFrame(
            results
        )
    )

    results_dataframe = (
        results_dataframe.sort_values(
            by="roc_auc",
            ascending=False,
        )
    )

    display_columns = [
        "run_name",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "training_seconds",
    ]

    LOGGER.info(
        "Baseline comparison:\n%s",
        results_dataframe[
            display_columns
        ].to_string(
            index=False
        ),
    )


def validate_training_data(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
) -> None:
    """Validate datasets required for baseline training."""
    required_columns = set(
        FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )

    train_missing = (
        required_columns
        - set(
            train_dataframe.columns
        )
    )

    validation_missing = (
        required_columns
        - set(
            validation_dataframe.columns
        )
    )

    if train_missing:
        raise RuntimeError(
            "Training dataset is missing "
            f"columns: {sorted(train_missing)}"
        )

    if validation_missing:
        raise RuntimeError(
            "Validation dataset is missing "
            f"columns: "
            f"{sorted(validation_missing)}"
        )

    if train_dataframe.empty:
        raise RuntimeError(
            "Training dataset is empty."
        )

    if validation_dataframe.empty:
        raise RuntimeError(
            "Validation dataset is empty."
        )

    if not train_dataframe[
        TARGET_COLUMN
    ].isin([0, 1]).all():
        raise RuntimeError(
            "Training target contains "
            "invalid values."
        )

    if not validation_dataframe[
        TARGET_COLUMN
    ].isin([0, 1]).all():
        raise RuntimeError(
            "Validation target contains "
            "invalid values."
        )


def train_baselines(
    model_names: list[str] | None = None,
    feature_sets: list[str] | None = None,
) -> None:
    """Train all requested baseline model combinations."""
    config = load_config()

    features_dir = Path(
        "data/features"
    )

    train_path = (
        features_dir
        / "train.parquet"
    )

    validation_path = (
        features_dir
        / "validation.parquet"
    )

    if not train_path.exists():
        raise FileNotFoundError(
            "Training dataset not found: "
            f"{train_path}"
        )

    if not validation_path.exists():
        raise FileNotFoundError(
            "Validation dataset not found: "
            f"{validation_path}"
        )

    LOGGER.info(
        "Loading training data=%s",
        train_path,
    )

    train_dataframe = (
        pd.read_parquet(
            train_path
        )
    )

    LOGGER.info(
        "Loading validation data=%s",
        validation_path,
    )

    validation_dataframe = (
        pd.read_parquet(
            validation_path
        )
    )

    validate_training_data(
        train_dataframe=(
            train_dataframe
        ),
        validation_dataframe=(
            validation_dataframe
        ),
    )

    LOGGER.info(
        "Training rows=%d | "
        "validation rows=%d",
        len(train_dataframe),
        len(validation_dataframe),
    )

    tracking_uri = config[
        "mlflow"
    ][
        "tracking_uri"
    ]

    experiment_name = config[
        "mlflow"
    ][
        "baseline_experiment_name"
    ]

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        experiment_name
    )

    selected_models = (
        model_names
        if model_names is not None
        else MODEL_NAMES
    )

    selected_feature_sets = (
        feature_sets
        if feature_sets is not None
        else FEATURE_SETS
    )

    results = []

    for model_name in selected_models:
        for feature_set in (
            selected_feature_sets
        ):
            result = (
                train_single_model(
                    model_name=model_name,
                    feature_set=(
                        feature_set
                    ),
                    train_dataframe=(
                        train_dataframe
                    ),
                    validation_dataframe=(
                        validation_dataframe
                    ),
                    config=config,
                )
            )

            results.append(
                result
            )

    print_results(
        results
    )

    best_result = max(
        results,
        key=lambda result: float(
            result["roc_auc"]
        ),
    )

    LOGGER.info(
        "Best baseline | "
        "run=%s | "
        "ROC-AUC=%.6f | "
        "PR-AUC=%.6f",
        best_result[
            "run_name"
        ],
        best_result[
            "roc_auc"
        ],
        best_result[
            "pr_auc"
        ],
    )

    LOGGER.info(
        "Baseline training completed "
        "successfully."
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train Bank Marketing "
            "baseline models."
        )
    )

    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        default=None,
        help=(
            "Train only one model family. "
            "By default all models are trained."
        ),
    )

    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SETS,
        default=None,
        help=(
            "Train only one feature set. "
            "By default raw and engineered "
            "features are compared."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run baseline training CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    args = parse_args()

    selected_models = None

    if args.model is not None:
        selected_models = [
            args.model
        ]

    selected_feature_sets = None

    if args.feature_set is not None:
        selected_feature_sets = [
            args.feature_set
        ]

    train_baselines(
        model_names=selected_models,
        feature_sets=(
            selected_feature_sets
        ),
    )


if __name__ == "__main__":
    main()