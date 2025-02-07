"""Tests for BenchmarkWrapper."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from periomod.wrapper import BenchmarkWrapper


def test_benchmark_wrapper_initialization():
    """Test BenchmarkWrapper init."""
    wrapper = BenchmarkWrapper(
        task="pocketclosure",
        encodings=["one_hot", "target"],
        learners=["lr", "rf"],
        tuning_methods=["holdout", "cv"],
        hpo_methods=["rs", "hebo"],
        criteria=["f1", "brier_score"],
        sampling=["upsampling"],
        factor=0.5,
        n_configs=10,
        n_jobs=4,
        verbose=True,
        path=Path("data/processed_data.csv"),
    )

    assert wrapper.task == "pocketclosure"
    assert wrapper.encodings == ["one_hot", "target"]
    assert wrapper.learners == ["lr", "rf"]
    assert wrapper.tuning_methods == ["holdout", "cv"]
    assert wrapper.hpo_methods == ["rs", "hebo"]
    assert wrapper.criteria == ["f1", "brier_score"]
    assert wrapper.sampling == ["upsampling"]
    assert wrapper.factor == 0.5
    assert wrapper.n_configs == 10
    assert wrapper.n_jobs == 4
    assert wrapper.verbose is True
    assert wrapper.path == Path("data/processed_data.csv")
    assert wrapper.classification == "binary"


@patch("periomod.benchmarking.Baseline")
def test_benchmark_wrapper_baseline(mock_baseline):
    """Test baseline method."""
    mock_baseline_instance = mock_baseline.return_value
    mock_baseline_instance.baseline.return_value = pd.DataFrame({
        "Model": ["DummyClassifier"],
        "Accuracy": [0.5],
    })

    wrapper = BenchmarkWrapper(
        task="pocketclosure",
        encodings=["one_hot"],
        learners=["lr"],
        tuning_methods=["holdout"],
        hpo_methods=["rs"],
        criteria=["f1"],
        n_jobs=1,
    )

    baseline_df = wrapper.baseline()

    assert isinstance(baseline_df, pd.DataFrame)
    assert "Encoding" in baseline_df.columns
    assert baseline_df["Encoding"].iloc[0] == "one_hot"


@patch("periomod.benchmarking.Benchmarker")
def test_benchmark_wrapper_wrapped_benchmark(mock_benchmarker):
    """Test wrapped benchmark."""
    mock_benchmarker_instance = mock_benchmarker.return_value
    mock_benchmarker_instance.run_all_benchmarks.return_value = (
        pd.DataFrame({"Learner": ["lr"], "Accuracy": [0.8]}),
        {"lr_model": "model_object"},
    )

    wrapper = BenchmarkWrapper(
        task="pocketclosure",
        encodings=["one_hot"],
        learners=["lr"],
        tuning_methods=["holdout"],
        hpo_methods=["rs"],
        criteria=["f1"],
        sampling=[None],
        n_jobs=1,
    )

    benchmark_df, _ = wrapper.wrapped_benchmark()
    print(benchmark_df)
    assert isinstance(benchmark_df, pd.DataFrame)
    assert benchmark_df["Learner"].iloc[0] == "lr"


@patch("os.makedirs")
@patch("pandas.DataFrame.to_csv")
def test_benchmark_wrapper_save_benchmark(mock_to_csv, mock_makedirs):
    """Test benchmark saving."""
    wrapper = BenchmarkWrapper(
        task="pocketclosure",
        encodings=[],
        learners=[],
        tuning_methods=[],
        hpo_methods=[],
        criteria=[],
    )

    benchmark_df = pd.DataFrame({"Model": ["lr"], "Accuracy": [0.8]})
    path = Path("reports/pocketclosure/test.csv").resolve()  # Normalize path

    wrapper.save_benchmark(benchmark_df, path=path)

    mock_to_csv.assert_called_once_with(path, index=False)
    mock_makedirs.assert_called_once_with(path.parent, exist_ok=True)


@patch("os.makedirs")
@patch("joblib.dump")
def test_benchmark_wrapper_save_learners(mock_joblib_dump, mock_makedirs):
    """Test learner saving."""
    wrapper = BenchmarkWrapper(
        task="pocketclosure",
        encodings=[],
        learners=[],
        tuning_methods=[],
        hpo_methods=[],
        criteria=[],
    )

    learners_dict = {"lr_model": MagicMock()}
    path = Path("models/pocketclosure").resolve()  # Normalize path

    wrapper.save_learners(learners_dict, path=path)

    mock_makedirs.assert_called_once_with(path, exist_ok=True)
    mock_joblib_dump.assert_called_once()
    args, _ = mock_joblib_dump.call_args
    assert args[0] == learners_dict["lr_model"]
    assert args[1] == path / "lr_model.pkl"
