"""Tests for Benchmarker."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from periomod.benchmarking import Benchmarker


def create_synthetic_data(n_samples=100, n_features=5, classification="binary"):
    """Generates synthetic data."""
    data = {f"feature_{i}": range(n_samples) for i in range(n_features)}
    if classification == "binary":
        data["y"] = [i % 2 for i in range(n_samples)]
    else:
        data["y"] = [i % 3 for i in range(n_samples)]
    data["id_patient"] = [i // 10 for i in range(n_samples)]  # Add id_patient
    df = pd.DataFrame(data)
    return df


@patch("periomod.benchmarking._benchmark.ProcessedDataLoader")
def test_benchmarker_load_data(mock_data_loader):
    """Test the _load_data_for_tasks method of Benchmarker."""
    synthetic_df = create_synthetic_data()
    mock_instance = mock_data_loader.return_value
    mock_instance.load_data.return_value = synthetic_df
    mock_instance.transform_data.return_value = synthetic_df

    benchmarker = Benchmarker(
        task="pocketclosure",
        learners=["lr"],
        tuning_methods=["holdout"],
        hpo_methods=["hebo"],
        criteria=["f1"],
        encodings=["one_hot"],
        sampling=[None],
        factor=None,
        n_configs=5,
        n_jobs=1,
        cv_folds=3,
        test_seed=42,
        test_size=0.2,
        verbose=False,
        path=Path("."),
        name="data.csv",
    )

    data_cache = benchmarker._load_data_for_tasks()
    assert "one_hot" in data_cache
    pd.testing.assert_frame_equal(data_cache["one_hot"], synthetic_df)
    mock_data_loader.assert_called()


@patch("periomod.benchmarking._benchmark.Experiment")
@patch("periomod.benchmarking._benchmark.ProcessedDataLoader")
def test_benchmarker_run_all_benchmarks(mock_data_loader, mock_experiment):
    """Test the run_all_benchmarks method of Benchmarker."""
    synthetic_df = create_synthetic_data()
    mock_loader_instance = mock_data_loader.return_value
    mock_loader_instance.load_data.return_value = synthetic_df
    mock_loader_instance.transform_data.return_value = synthetic_df

    # Mock the Experiment.perform_evaluation method
    mock_experiment_instance = mock_experiment.return_value
    mock_experiment_instance.perform_evaluation.return_value = {
        "metrics": {"F1 Score": 0.9, "Accuracy": 0.95},
        "model": "trained_model",
    }

    benchmarker = Benchmarker(
        task="pocketclosure",
        learners=["lr"],
        tuning_methods=["holdout"],
        hpo_methods=["hebo"],
        criteria=["f1"],
        encodings=["one_hot"],
        sampling=[None],
        factor=None,
        n_configs=5,
        n_jobs=1,
        cv_folds=3,
        test_seed=42,
        test_size=0.2,
        verbose=False,
        path=Path("."),
        name="data.csv",
    )

    results_df, top_models = benchmarker.run_all_benchmarks()
    assert isinstance(results_df, pd.DataFrame)
    assert not results_df.empty
    assert "Learner" in results_df.columns
    assert len(top_models) == 1  # Only one configuration

    # Updated expected key
    expected_key = (
        "pocketclosure_lr_holdout_hebo_f1_one_hot_no_sampling_factorNone_rank1_score0.9"
    )
    assert expected_key in top_models
    assert top_models[expected_key] == "trained_model"

    mock_experiment.assert_called()
    mock_experiment_instance.perform_evaluation.assert_called()
