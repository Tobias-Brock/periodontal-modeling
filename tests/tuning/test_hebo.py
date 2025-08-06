"""Tests for HEBOTUner."""

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from periomod.resampling import Resampler
from periomod.training import Trainer
from periomod.tuning import HEBOTuner


def create_sample_data(n_samples=100, n_features=5, n_classes=2, random_state=42):
    """Creates a sample dataset for testing.

    Args:
        n_samples (int): Number of samples.
        n_features (int): Number of features.
        n_classes (int): Number of output classes.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature matrix `X` and target vector `y`.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=3,
        n_classes=n_classes,
        random_state=random_state,
        weights=[0.7, 0.3] if n_classes == 2 else None,
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    y = pd.Series(y)
    return X, y


@pytest.fixture
def sample_data():
    """Sampling function for testing.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature matrix `X` and target vector `y`.
    """
    return create_sample_data()


def test_hebotuner_holdout(sample_data):
    """Test HEBOTuner with holdout tuning."""
    X, y = sample_data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    trainer = Trainer(
        classification="binary",
        criterion="f1",
        tuning="holdout",
        hpo="hebo",
        mlp_training=False,
        threshold_tuning=True,
    )
    tuner = HEBOTuner(
        classification="binary",
        criterion="f1",
        tuning="holdout",
        hpo="hebo",
        n_configs=5,
        n_jobs=1,
        verbose=False,
        trainer=trainer,
        mlp_training=False,
        threshold_tuning=True,
    )

    best_params, best_threshold = tuner.holdout(
        learner="lr", X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val
    )

    assert isinstance(best_params, dict)
    assert isinstance(best_threshold, (float, type(None)))


def test_hebotuner_cv(sample_data):
    """Test HEBOTuner with cross-validation tuning."""
    X, y = sample_data
    resampler = Resampler(classification="binary", encoding="one_hot")
    resampler.y = "y"
    resampler.group_col = "feature_0"
    resampler.all_cat_vars = []

    data = X.copy()
    data["y"] = y

    outer_splits, _ = resampler.cv_folds(df=data, n_folds=3, seed=42)
    trainer = Trainer(
        classification="binary",
        criterion="f1",
        tuning="cv",
        hpo="hebo",
        mlp_training=False,
        threshold_tuning=True,
    )
    tuner = HEBOTuner(
        classification="binary",
        criterion="f1",
        tuning="cv",
        hpo="hebo",
        n_configs=5,
        n_jobs=1,
        verbose=False,
        trainer=trainer,
        mlp_training=False,
        threshold_tuning=True,
    )

    best_params, best_threshold = tuner.cv(learner="lr", outer_splits=outer_splits)

    assert isinstance(best_params, dict)
    assert isinstance(best_threshold, (float, type(None)))


def test_hebotuner_invalid_tuning_strategy(sample_data):
    """Test HEBOTuner with invalid tuning strategy."""
    with pytest.raises(
        ValueError, match="Unsupported tuning method. Choose either 'holdout' or 'cv'."
    ):
        trainer = Trainer(
            classification="binary",
            criterion="f1",
            tuning="invalid_tuning",
            hpo="hebo",
            mlp_training=False,
            threshold_tuning=True,
        )
        HEBOTuner(
            classification="binary",
            criterion="f1",
            tuning="invalid_tuning",
            hpo="hebo",
            n_configs=5,
            n_jobs=1,
            verbose=False,
            trainer=trainer,
            mlp_training=False,
            threshold_tuning=True,
        )
