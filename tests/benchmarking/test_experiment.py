"""Tests for Experiment."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from periomod.benchmarking import Experiment


def create_synthetic_data(
    n_samples=100, n_features=5, classification="binary"
) -> pd.DataFrame:
    """Creates synthetic data for testing.

    Args:
        n_samples (int): Number of samples in the dataset.
        n_features (int): Number of feature columns.
        classification (str): Type of classification ("binary" or "multiclass").

    Returns:
        pd.DataFrame: A synthetic dataset with feature columns, a target column `y`,
        and an `id_patient` column.
    """
    data = {f"feature_{i}": list(range(n_samples)) for i in range(n_features)}
    if classification == "binary":
        data["y"] = [i % 2 for i in range(n_samples)]
    else:
        data["y"] = [i % 3 for i in range(n_samples)]
    data["id_patient"] = [i // 10 for i in range(n_samples)]  # Add id_patient
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Create synthetic data.

    Returns:
        pd.DataFrame: A synthetic dataset.
    """
    return create_synthetic_data()


@patch("periomod.training.final_metrics")
@patch("periomod.resampling.Resampler")
@patch("periomod.training.Trainer")
@patch("periomod.tuning.HEBOTuner")
def test_experiment_perform_evaluation_holdout(
    mock_hebotuner_class,
    mock_trainer_class,
    mock_resampler_class,
    mock_final_metrics,
    synthetic_df,
):
    """Test perform_evaluation method with holdout tuning."""
    # Initialize mock instances
    mock_resampler_instance = mock_resampler_class.return_value
    mock_trainer_instance = mock_trainer_class.return_value
    mock_hebotuner_instance = mock_hebotuner_class.return_value

    mock_resampler_instance.group_col = "id_patient"
    mock_resampler_instance.split_train_test_df.return_value = (
        synthetic_df,
        synthetic_df,
    )
    y_test = [0, 1, 0, 1]  # Mocked test labels
    mock_resampler_instance.split_x_y.return_value = (None, None, None, y_test)

    # Mock tuner methods
    mock_hebotuner_instance.holdout.return_value = ({"param": "value"}, 0.5)

    # Mock trainer methods
    final_model = MagicMock(name="trained_model")
    final_predictions = [0, 1, 0, 1]  # Mocked predictions
    final_probs = [0.2, 0.8, 0.3, 0.7]  # Mocked probabilities
    mock_trainer_instance.train_final_model.return_value = (
        final_model,
        final_predictions,
        final_probs,
    )

    mock_final_metrics.return_value = {
        "F1 Score": 0.6666666666666666,
        "Precision": 0.5,
        "Recall": 1.0,
        "Accuracy": 0.5,
        "Brier Score": 0.24998317617707486,
        "ROC AUC Score": 0.55,
        "Confusion Matrix": np.array([[10, 0], [10, 0]]),
        "Best Threshold": 0.51,
    }

    experiment = Experiment(
        data=synthetic_df,
        task="pocketclosure",
        learner="lr",
        criterion="f1",
        encoding="one_hot",
        tuning="holdout",
        hpo="hebo",
        sampling=None,
        factor=None,
        n_configs=5,
        racing_folds=None,
        n_jobs=1,
        cv_folds=3,
        test_seed=42,
        test_size=0.2,
        val_size=0.1,
        cv_seed=123,
        mlp_flag=False,
        threshold_tuning=True,
        verbose=False,
    )

    result = experiment.perform_evaluation()

    expected_metrics = {
        "F1 Score": 0.6666666666666666,
        "Precision": 0.5,
        "Recall": 1.0,
        "Accuracy": 0.5,
        "Brier Score": 0.24998317617707486,
        "ROC AUC Score": 0.55,
        "Confusion Matrix": np.array([[10, 0], [10, 0]]),
        "Best Threshold": 0.51,
    }

    for key in expected_metrics:
        if isinstance(expected_metrics[key], float):
            assert result["metrics"][key] == pytest.approx(
                expected_metrics[key], rel=1e-6
            )
        elif isinstance(expected_metrics[key], np.ndarray):
            np.testing.assert_array_equal(result["metrics"][key], expected_metrics[key])
        else:
            assert result["metrics"][key] == expected_metrics[key]


@patch("periomod.training.final_metrics")
@patch("periomod.resampling.Resampler")
@patch("periomod.training.Trainer")
@patch("periomod.tuning.HEBOTuner")
def test_experiment_perform_evaluation_cv(
    mock_hebotuner_class,
    mock_trainer_class,
    mock_resampler_class,
    mock_final_metrics,
    synthetic_df,
):
    """Test perform_evaluation method with cv tuning."""
    mock_resampler_instance = mock_resampler_class.return_value
    mock_trainer_instance = mock_trainer_class.return_value
    mock_hebotuner_instance = mock_hebotuner_class.return_value

    mock_resampler_instance.group_col = "id_patient"
    mock_resampler_instance.split_train_test_df.return_value = (
        synthetic_df,
        synthetic_df,
    )
    y_test = [0, 1, 0, 1]
    mock_resampler_instance.split_x_y.return_value = (None, None, None, y_test)
    mock_hebotuner_instance.cv.return_value = ({"param": "value"}, 0.5)

    final_model = MagicMock(name="trained_model_cv")
    final_predictions = [0, 1, 0, 1]
    final_probs = [0.2, 0.8, 0.3, 0.7]
    mock_trainer_instance.train_final_model.return_value = (
        final_model,
        final_predictions,
        final_probs,
    )

    mock_final_metrics.return_value = {
        "F1 Score": 0.6666666666666666,
        "Precision": 0.5,
        "Recall": 1.0,
        "Accuracy": 0.5,
        "Brier Score": 0.24998317617707486,
        "ROC AUC Score": 0.55,
        "Confusion Matrix": np.array([[10, 0], [10, 0]]),
        "Best Threshold": 0.52,
    }

    experiment = Experiment(
        data=synthetic_df,
        task="pocketclosure",
        learner="lr",
        criterion="f1",
        encoding="one_hot",
        tuning="cv",
        hpo="hebo",
        sampling=None,
        factor=None,
        n_configs=5,
        racing_folds=None,
        n_jobs=1,
        cv_folds=3,
        test_seed=42,
        test_size=0.2,
        val_size=0.1,
        cv_seed=123,
        mlp_flag=False,
        threshold_tuning=True,
        verbose=False,
    )

    result = experiment.perform_evaluation()

    expected_metrics = {
        "F1 Score": 0.6666666666666666,
        "Precision": 0.5,
        "Recall": 1.0,
        "Accuracy": 0.5,
        "Brier Score": 0.24998317617707486,
        "ROC AUC Score": 0.55,
        "Confusion Matrix": np.array([[10, 0], [10, 0]]),
        "Best Threshold": 0.52,
    }

    for key in expected_metrics:
        if isinstance(expected_metrics[key], float):
            assert result["metrics"][key] == pytest.approx(
                expected_metrics[key], rel=1e-6
            )
        elif isinstance(expected_metrics[key], np.ndarray):
            np.testing.assert_array_equal(result["metrics"][key], expected_metrics[key])
        else:
            assert result["metrics"][key] == expected_metrics[key]


def test_experiment_unsupported_tuning(synthetic_df):
    """Test perform_evaluation method with unsupported tuning method."""
    with pytest.raises(
        ValueError, match="Unsupported tuning method. Choose either 'holdout' or 'cv'."
    ):
        experiment = Experiment(
            data=synthetic_df,
            task="pocketclosure",
            learner="lr",
            criterion="f1",
            encoding="one_hot",
            tuning="unsupported",
            hpo="rs",
            sampling=None,
            factor=None,
            n_configs=5,
            racing_folds=None,
            n_jobs=1,
            cv_folds=3,
            test_seed=42,
            test_size=0.2,
            val_size=0.1,
            cv_seed=123,
            mlp_flag=False,
            threshold_tuning=True,
            verbose=False,
        )

        experiment.perform_evaluation()
