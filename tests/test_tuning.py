"""
Unit tests for XGBoost hyperparameter tuning utilities.
"""

from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline

from bankmarketing.training.tune import (
    build_cross_validation,
    build_randomized_search,
    build_search_space,
    build_xgboost_pipeline,
    clean_best_params,
)


def sample_config() -> dict:
    """Return a small deterministic tuning configuration."""
    return {
        "models": {
            "xgboost": {
                "random_state": 42,
                "n_jobs": 1,
                "eval_metric": "logloss",
                "tree_method": "hist",
            }
        },
        "tuning": {
            "xgboost": {
                "n_iter": 2,
                "cv_folds": 3,
                "random_state": 42,
                "scoring": "roc_auc",
                "search_space": {
                    "n_estimators": [
                        10,
                        20,
                    ],
                    "max_depth": [
                        3,
                        4,
                    ],
                    "learning_rate": [
                        0.05,
                        0.1,
                    ],
                },
            }
        },
    }


def test_build_xgboost_pipeline() -> None:
    """Check XGBoost tuning pipeline creation."""
    pipeline = (
        build_xgboost_pipeline(
            sample_config()
        )
    )

    assert isinstance(
        pipeline,
        Pipeline,
    )

    assert (
        "preprocessor"
        in pipeline.named_steps
    )

    assert (
        "model"
        in pipeline.named_steps
    )


def test_build_search_space() -> None:
    """Check sklearn pipeline parameter prefixes."""
    search_space = (
        build_search_space(
            sample_config()
        )
    )

    assert (
        "model__n_estimators"
        in search_space
    )

    assert (
        "model__max_depth"
        in search_space
    )

    assert (
        "model__learning_rate"
        in search_space
    )


def test_clean_best_params() -> None:
    """Check removal of sklearn parameter prefixes."""
    parameters = {
        "model__n_estimators": 500,
        "model__max_depth": 5,
    }

    cleaned = clean_best_params(
        parameters
    )

    assert cleaned == {
        "n_estimators": 500,
        "max_depth": 5,
    }


def test_cross_validation_configuration() -> None:
    """Check stratified cross-validation configuration."""
    cross_validation = (
        build_cross_validation(
            sample_config()
        )
    )

    assert isinstance(
        cross_validation,
        StratifiedKFold,
    )

    assert (
        cross_validation.n_splits
        == 3
    )

    assert (
        cross_validation.shuffle
        is True
    )

    assert (
        cross_validation.random_state
        == 42
    )


def test_randomized_search_configuration() -> None:
    """Check randomized search configuration."""
    config = sample_config()

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

    assert isinstance(
        search,
        RandomizedSearchCV,
    )

    assert search.n_iter == 2

    assert (
        search.scoring
        == "roc_auc"
    )

    assert (
        search.refit
        is True
    )

    assert (
        search.n_jobs
        == 1
    )