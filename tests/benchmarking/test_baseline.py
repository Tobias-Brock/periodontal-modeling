"""Tests for baseline."""

from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from periomod.benchmarking import Baseline


def create_synthetic_data(n_samples=10, n_features=5, classification="binary"):
    """Creates synthetic data for testing."""
    data = {f"feature_{i}": range(n_samples) for i in range(n_features)}
    if classification == "binary":
        data["y"] = [i % 2 for i in range(n_samples)]
    else:
        data["y"] = [i % 3 for i in range(n_samples)]
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def synthetic_data():
    """Generate synthetic data."""
    return create_synthetic_data()


@patch("periomod.data.ProcessedDataLoader")
def test_baseline_evaluation(mock_data_loader):
    """Test the Baseline class evaluation."""
    # Mock the data loader to return synthetic data
    synthetic_df = create_synthetic_data()
    mock_instance = mock_data_loader.return_value
    mock_instance.load_data.return_value = synthetic_df
    mock_instance.transform_data.return_value = synthetic_df

    baseline = Baseline(
        task="pocketclosure",
        encoding="one_hot",
        random_state=42,
        lr_solver="saga",
        dummy_strategy="most_frequent",
    )

    results_df = baseline.baseline()

    assert isinstance(results_df, pd.DataFrame)
    assert not results_df.empty
    expected_models = ["Dummy Classifier", "Logistic Regression", "Random Forest"]
    assert results_df["Model"].tolist() == expected_models

    expected_metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC Score",
        "Model",
    ]
    for metric in expected_metrics:
        assert metric in results_df.columns


def test_baseline_custom_models():
    """Test Baseline with custom models."""
    synthetic_df = create_synthetic_data()

    with patch("periomod.data.ProcessedDataLoader") as mock_data_loader:
        mock_instance = mock_data_loader.return_value
        mock_instance.load_data.return_value = synthetic_df
        mock_instance.transform_data.return_value = synthetic_df

        custom_models = [
            (
                "Dummy Classifier",
                DummyClassifier(strategy="uniform"),
            ),
            (
                "Logistic Regression",
                LogisticRegression(solver="liblinear", random_state=42),
            ),
            (
                "Custom Model 1",
                LogisticRegression(solver="liblinear", random_state=42),
            ),
            (
                "Custom Model 2",
                DummyClassifier(strategy="uniform"),
            ),
        ]

        baseline = Baseline(
            task="pocketclosure",
            encoding="one_hot",
            random_state=42,
            models=custom_models,
        )

        results_df = baseline.baseline()

        assert isinstance(results_df, pd.DataFrame)
        assert not results_df.empty
        expected_models = [
            "Dummy Classifier",
            "Logistic Regression",
            "Custom Model 1",
            "Custom Model 2",
        ]
        assert set(results_df["Model"].tolist()) == set(expected_models)


def test_baseline_multiclass():
    """Test Baseline with multiclass classification."""
    synthetic_df = create_synthetic_data(classification="multiclass")
    with patch("periomod.data.ProcessedDataLoader") as mock_data_loader:
        mock_instance = mock_data_loader.return_value
        mock_instance.load_data.return_value = synthetic_df
        mock_instance.transform_data.return_value = synthetic_df

        baseline = Baseline(
            task="pdgrouprevaluation",  # Multiclass task
            encoding="one_hot",
            random_state=42,
        )

        results_df = baseline.baseline()

        assert isinstance(results_df, pd.DataFrame)
        assert not results_df.empty
        expected_models = ["Dummy Classifier", "Logistic Regression", "Random Forest"]
        assert results_df["Model"].tolist() == expected_models

        # Check for multiclass metrics
        expected_metrics = [
            "Accuracy",
            "Class F1 Scores",
            "Macro F1",
            "Multiclass Brier Score",
        ]
        for metric in expected_metrics:
            assert metric in results_df.columns


def test_baseline_invalid_task():
    """Test Baseline with an invalid task."""
    with pytest.raises(ValueError):
        Baseline(
            task="unknown_task",
            encoding="one_hot",
            random_state=42,
        )
