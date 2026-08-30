"""
Tune the XGBoost Bank Marketing classifier.

The tuning stage uses only the training dataset for cross-validation.
The validation dataset is used once after hyperparameter search to
compare the tuned candidate against the baseline.

The internal test dataset is never accessed during tuning.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
)
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
    build_preprocessor,
    get_feature_specification,
)
from xgboost import XGBClassifier

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


def build_xgboost_pipeline(
    config: dict,
) -> Pipeline:
    """Build the engineered-feature XGBoost pipeline."""
    (
        _,
        categorical_columns,
        numeric_columns,
        binary_columns,
    ) = get_feature_specification(
        "engineered"
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

    model_config = config[
        "models"
    ][
        "xgboost"
    ]

    model = XGBClassifier(
        random_state=int(
            model_config[
                "random_state"
            ]
        ),
        n_jobs=int(
            model_config[
                "n_jobs"
            ]
        ),
        eval_metric=(
            model_config[
                "eval_metric"
            ]
        ),
        tree_method=(
            model_config[
                "tree_method"
            ]
        ),
    )

    return Pipeline(
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


def build_search_space(
    config: dict,
) -> dict:
    """Build sklearn-compatible XGBoost search space."""
    search_space = config[
        "tuning"
    ][
        "xgboost"
    ][
        "search_space"
    ]

    return {
        f"model__{parameter}": values
        for parameter, values
        in search_space.items()
    }


def build_cross_validation(
    config: dict,
) -> StratifiedKFold:
    """Build deterministic stratified cross-validation."""
    tuning_config = config[
        "tuning"
    ][
        "xgboost"
    ]

    return StratifiedKFold(
        n_splits=int(
            tuning_config[
                "cv_folds"
            ]
        ),
        shuffle=True,
        random_state=int(
            tuning_config[
                "random_state"
            ]
        ),
    )


def build_randomized_search(
    pipeline: Pipeline,
    config: dict,
) -> RandomizedSearchCV:
    """Create the randomized hyperparameter search."""
    tuning_config = config[
        "tuning"
    ][
        "xgboost"
    ]

    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=(
            build_search_space(
                config
            )
        ),
        n_iter=int(
            tuning_config[
                "n_iter"
            ]
        ),
        scoring=(
            tuning_config[
                "scoring"
            ]
        ),
        cv=build_cross_validation(
            config
        ),
        refit=True,
        random_state=int(
            tuning_config[
                "random_state"
            ]
        ),
        n_jobs=1,
        verbose=2,
        return_train_score=False,
    )


def clean_best_params(
    best_params: dict,
) -> dict:
    """Remove sklearn pipeline prefixes from parameters."""
    return {
        key.replace(
            "model__",
            "",
        ): value
        for key, value
        in best_params.items()
    }


def save_best_params(
    best_params: dict,
    output_path: Path,
) -> None:
    """Save selected hyperparameters as JSON."""
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
            best_params,
            stream,
            indent=2,
            sort_keys=True,
        )

    temporary_path.replace(
        output_path
    )


def save_candidate_model(
    pipeline: Pipeline,
    output_path: Path,
) -> None:
    """Persist the tuned candidate model locally."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        output_path.with_suffix(
            ".joblib.tmp"
        )
    )

    joblib.dump(
        pipeline,
        temporary_path,
    )

    temporary_path.replace(
        output_path
    )


def validate_datasets(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
) -> None:
    """Validate tuning input datasets."""
    required_columns = set(
        FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )

    for name, dataframe in [
        (
            "train",
            train_dataframe,
        ),
        (
            "validation",
            validation_dataframe,
        ),
    ]:
        missing_columns = (
            required_columns
            - set(
                dataframe.columns
            )
        )

        if missing_columns:
            raise RuntimeError(
                f"{name} dataset is "
                "missing columns: "
                f"{sorted(missing_columns)}"
            )

        if dataframe.empty:
            raise RuntimeError(
                f"{name} dataset is empty."
            )

        if not dataframe[
            TARGET_COLUMN
        ].isin([0, 1]).all():
            raise RuntimeError(
                f"{name} target contains "
                "invalid values."
            )


def tune_xgboost() -> None:
    """Tune XGBoost and evaluate the candidate on validation data."""
    config = load_config()

    train_path = Path(
        "data/features/train.parquet"
    )

    validation_path = Path(
        "data/features/validation.parquet"
    )

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: "
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

    validate_datasets(
        train_dataframe=(
            train_dataframe
        ),
        validation_dataframe=(
            validation_dataframe
        ),
    )

    x_train = train_dataframe[
        FEATURE_COLUMNS
    ]

    y_train = train_dataframe[
        TARGET_COLUMN
    ]

    x_validation = (
        validation_dataframe[
            FEATURE_COLUMNS
        ]
    )

    y_validation = (
        validation_dataframe[
            TARGET_COLUMN
        ]
    )

    tuning_config = config[
        "tuning"
    ][
        "xgboost"
    ]

    LOGGER.info(
        "Starting XGBoost tuning | "
        "train_rows=%d | "
        "validation_rows=%d | "
        "features=%d | "
        "iterations=%d | "
        "cv_folds=%d",
        len(x_train),
        len(x_validation),
        len(FEATURE_COLUMNS),
        tuning_config[
            "n_iter"
        ],
        tuning_config[
            "cv_folds"
        ],
    )

    pipeline = (
        build_xgboost_pipeline(
            config
        )
    )

    search = (
        build_randomized_search(
            pipeline=pipeline,
            config=config,
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
            "tuning_experiment_name"
        ]
    )

    with mlflow.start_run(
        run_name=(
            "xgboost_engineered_tuning"
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
            "tuning",
        )

        mlflow.log_param(
            "feature_count",
            len(FEATURE_COLUMNS),
        )

        mlflow.log_param(
            "cv_folds",
            tuning_config[
                "cv_folds"
            ],
        )

        mlflow.log_param(
            "n_iter",
            tuning_config[
                "n_iter"
            ],
        )

        mlflow.log_param(
            "scoring",
            tuning_config[
                "scoring"
            ],
        )

        start_time = (
            time.perf_counter()
        )

        search.fit(
            x_train,
            y_train,
        )

        tuning_seconds = (
            time.perf_counter()
            - start_time
        )

        best_pipeline = (
            search.best_estimator_
        )

        best_params = (
            clean_best_params(
                search.best_params_
            )
        )

        validation_probabilities = (
            best_pipeline.predict_proba(
                x_validation
            )[:, 1]
        )

        validation_metrics = (
            compute_classification_metrics(
                y_true=(
                    y_validation.to_numpy()
                ),
                probabilities=(
                    validation_probabilities
                ),
                threshold=0.5,
            )
        )

        mlflow.log_metric(
            "cv_best_roc_auc",
            float(
                search.best_score_
            ),
        )

        for (
            metric_name,
            metric_value,
        ) in validation_metrics.items():
            mlflow.log_metric(
                f"validation_{metric_name}",
                float(metric_value),
            )

        mlflow.log_metric(
            "tuning_seconds",
            float(
                tuning_seconds
            ),
        )

        for (
            parameter_name,
            parameter_value,
        ) in best_params.items():
            mlflow.log_param(
                f"best_{parameter_name}",
                parameter_value,
            )

        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            name="model",
            skops_trusted_types=(
                XGBOOST_SKOPS_TRUSTED_TYPES
            ),
        )

        artifacts_dir = Path(
            "artifacts"
        )

        best_params_path = (
            artifacts_dir
            / "xgboost_best_params.json"
        )

        candidate_model_path = (
            artifacts_dir
            / "xgboost_candidate.joblib"
        )

        save_best_params(
            best_params=best_params,
            output_path=(
                best_params_path
            ),
        )

        save_candidate_model(
            pipeline=best_pipeline,
            output_path=(
                candidate_model_path
            ),
        )

        LOGGER.info(
            "Best cross-validation "
            "ROC-AUC=%.6f",
            search.best_score_,
        )

        LOGGER.info(
            "Best parameters=%s",
            best_params,
        )

        LOGGER.info(
            "Validation | "
            "ROC-AUC=%.6f | "
            "PR-AUC=%.6f | "
            "precision=%.6f | "
            "recall=%.6f | "
            "F1=%.6f",
            validation_metrics[
                "roc_auc"
            ],
            validation_metrics[
                "pr_auc"
            ],
            validation_metrics[
                "precision"
            ],
            validation_metrics[
                "recall"
            ],
            validation_metrics[
                "f1"
            ],
        )

        LOGGER.info(
            "Tuning completed in %.2f seconds.",
            tuning_seconds,
        )

        LOGGER.info(
            "Candidate model saved to %s",
            candidate_model_path,
        )

        LOGGER.info(
            "Best parameters saved to %s",
            best_params_path,
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Tune the engineered-feature "
            "XGBoost classifier."
        )
    )

    return parser.parse_args()


def main() -> None:
    """Run the XGBoost tuning CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    parse_args()

    tune_xgboost()


if __name__ == "__main__":
    main()