"""Test for BaseBenchmark."""

from pathlib import Path

import pandas as pd
import pytest

from periomod.benchmarking import BaseBenchmark, BaseExperiment
from periomod.resampling import Resampler
from periomod.training import Trainer
from periomod.tuning import HEBOTuner


class TestExperiment(BaseExperiment):
    """A concrete implementation of BaseExperiment for testing purposes."""

    def perform_evaluation(self) -> dict:
        """Mock implementation of perform_evaluation."""
        return {"status": "success"}

    def _evaluate_holdout(self, train_df: pd.DataFrame) -> dict:
        """Mock implementation of _evaluate_holdout."""
        return {"holdout": True}

    def _evaluate_cv(self) -> dict:
        """Mock implementation of _evaluate_cv."""
        return {"cv": True}


def create_sample_dataframe(n_samples=100, n_features=5):
    """Creates a sample DataFrame for testing."""
    data = {f"feature_{i}": range(n_samples) for i in range(n_features)}
    data["target"] = [0, 1] * (n_samples // 2)
    df = pd.DataFrame(data)
    return df


def test_base_experiment_initialization():
    """Test the initialization of BaseExperiment."""
    df = create_sample_dataframe()
    experiment = TestExperiment(
        df=df,
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

    assert experiment.task == "pocketclosure"
    assert experiment.classification == "binary"
    assert isinstance(experiment.resampler, Resampler)
    assert isinstance(experiment.trainer, Trainer)
    assert isinstance(experiment.tuner, HEBOTuner)


def test_base_experiment_methods():
    """Test the methods of BaseExperiment."""
    df = create_sample_dataframe()
    experiment = TestExperiment(
        df=df,
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

    eval_result = experiment.perform_evaluation()
    assert eval_result == {"status": "success"}

    holdout_result = experiment._evaluate_holdout(train_df=df)
    assert holdout_result == {"holdout": True}

    cv_result = experiment._evaluate_cv()
    assert cv_result == {"cv": True}


def test_base_experiment_unknown_task():
    """Test that an unknown task raises a ValueError."""
    df = create_sample_dataframe()
    with pytest.raises(
        ValueError,
        match="Unknown task: unknown_task. Unable to determine classification.",
    ):
        TestExperiment(
            df=df,
            task="unknown_task",
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


def test_base_benchmark_initialization():
    """Test the initialization of BaseBenchmark."""
    benchmark = BaseBenchmark(
        task="pocketclosure",
        learners=["lr", "rf"],
        tuning_methods=["holdout", "cv"],
        hpo_methods=["hebo", "rs"],
        criteria=["f1", "brier_score"],
        encodings=["one_hot", "binary"],
        sampling=[None, "upsample"],
        factor=1.0,
        n_configs=10,
        n_jobs=1,
        cv_folds=3,
        racing_folds=None,
        test_seed=42,
        test_size=0.2,
        val_size=0.1,
        cv_seed=123,
        mlp_flag=False,
        threshold_tuning=True,
        verbose=False,
        path=Path("data.csv"),
    )

    assert benchmark.task == "pocketclosure"
    assert benchmark.learners == ["lr", "rf"]
    assert benchmark.tuning_methods == ["holdout", "cv"]
    assert benchmark.hpo_methods == ["hebo", "rs"]
    assert benchmark.criteria == ["f1", "brier_score"]
    assert benchmark.encodings == ["one_hot", "binary"]
    assert benchmark.sampling == [None, "upsample"]
    assert benchmark.factor == 1.0
    assert benchmark.n_configs == 10
    assert benchmark.n_jobs == 1
    assert benchmark.cv_folds == 3
    assert benchmark.test_seed == 42
    assert benchmark.test_size == 0.2
    assert benchmark.val_size == 0.1
    assert benchmark.cv_seed == 123
    assert benchmark.mlp_flag is False
    assert benchmark.threshold_tuning is True
    assert benchmark.verbose is False
    assert benchmark.path == Path("data.csv")
