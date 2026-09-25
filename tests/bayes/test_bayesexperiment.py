"""Tests for Bayesian sampling, prediction and experiments."""

import numpy as np
import pytest

from periomod.bayes import (
    BayesBenchmarker,
    BayesExperiment,
    BayesTrainer,
    SpatialConfig,
)

SAMPLER = {"draws": 25, "tune": 25, "chains": 2, "cores": 1}


@pytest.fixture
def experiment(data):
    """Provides a small Bayesian experiment on synthetic data.

    Args:
        data (pd.DataFrame): Synthetic processed dataset.

    Returns:
        BayesExperiment: Experiment with a very short sampling run.
    """
    return BayesExperiment(
        data=data,
        task="pocketclosure",
        criterion="f1",
        max_draws=50,
        verbose=False,
        **SAMPLER,
    )


def test_prepare_data_builds_all_splits(experiment):
    """Test that design matrices are built for all three splits."""
    splits = experiment.prepare_data()

    assert set(splits) == {"train", "val", "test"}
    assert splits["train"].n_obs > splits["val"].n_obs
    assert splits["train"].n_features == len(splits["train"].feature_names)


def test_perform_evaluation_reports_posterior_summaries(experiment):
    """Test that the experiment reports predictions and diagnostics."""
    result = experiment.perform_evaluation()
    predictions = result["predictions"]

    assert "F1 Score" in result["metrics"]
    assert "Log Loss" in result["metrics"]
    assert "ECE" in result["metrics"]
    assert "test_metrics" not in result
    assert set(predictions.columns) == {
        "id_patient",
        "tooth",
        "side",
        "y",
        "prob",
        "prob_sd",
        "hdi_lower",
        "hdi_upper",
    }
    assert (predictions["hdi_lower"] <= predictions["prob"]).all()
    assert (predictions["prob"] <= predictions["hdi_upper"]).all()
    assert {"r_hat", "ess_bulk", "ess_tail"} <= set(result["diagnostics"].columns)
    assert "alpha" in result["diagnostics"].index
    assert set(result["sampler_stats"]) == {
        "divergences",
        "accept_rate",
        "max_treedepth",
    }


def test_posterior_predictive_check_statistics(experiment):
    """Test that posterior predictive checks target the nesting structure."""
    result = experiment.perform_evaluation()
    checks = result["posterior_predictive_check"]

    assert list(checks["Statistic"]) == [
        "Mean outcome",
        "SD of patient rates",
        "SD of tooth rates",
    ]
    assert checks["Bayesian p-value"].between(0, 1).all()
    assert (checks["Replicated Lower"] <= checks["Replicated Upper"]).all()


def test_test_split_is_evaluated_on_demand(experiment):
    """Test that test metrics are only computed when requested."""
    result = experiment.perform_evaluation(evaluate_test=True)

    assert "test_metrics" in result
    assert set(result["test_predictions"]["id_patient"]) <= set(
        experiment.transformer.split.test
    )


def test_group_effects_of_unseen_patients(data):
    """Test that unseen patients receive drawn or zeroed group effects."""
    sampled = BayesExperiment(
        data=data,
        task="pocketclosure",
        random_effects="sample",
        max_draws=50,
        verbose=False,
        **SAMPLER,
    )
    sampled.prepare_data()
    idata = sampled.trainer.fit(model=sampled.model, data=sampled.splits["train"])

    drawn = sampled.trainer.predict(
        model=sampled.model, idata=idata, data=sampled.splits["val"]
    )
    sampled.trainer.random_effects = "zero"
    zeroed = sampled.trainer.predict(
        model=sampled.model, idata=idata, data=sampled.splits["val"]
    )

    assert drawn.shape == zeroed.shape
    assert drawn.std(axis=0).mean() > zeroed.std(axis=0).mean()


def test_spatial_stage_runs(data):
    """Test that the spatial stage samples and predicts."""
    experiment = BayesExperiment(
        data=data,
        task="pocketclosure",
        spatial=SpatialConfig(mode="both", name="car_both"),
        max_draws=50,
        verbose=False,
        **SAMPLER,
    )
    result = experiment.perform_evaluation()

    assert "rho_tooth" in result["idata"].posterior
    assert "phi_site" in result["idata"].posterior
    assert np.isfinite(result["val_score"])


def test_multiclass_classification_raises():
    """Test that multiclass outcomes are rejected with a clear message."""
    with pytest.raises(ValueError, match="only supports binary"):
        BayesTrainer(classification="multiclass", criterion="macro_f1")


def test_unsupported_task_raises(data):
    """Test that unsupported tasks are rejected."""
    with pytest.raises(ValueError, match="not supported"):
        BayesExperiment(data=data, task="pdgrouprevaluation", verbose=False)


def test_benchmarker_reports_metrics_and_diagnostics(data, tmp_path):
    """Test that the benchmark reports metrics and convergence per run."""
    path = tmp_path / "processed.csv"
    data.to_csv(path, index=False)
    benchmarker = BayesBenchmarker(
        tasks=["pocketclosure"],
        criteria=["f1"],
        stages=["nested", "car_tooth"],
        path=path,
        verbose=False,
        experiment_args={"max_draws": 50, **SAMPLER},
    )

    results, posteriors = benchmarker.run_benchmarks()

    assert list(results["Stage"]) == ["nested", "car_tooth"]
    assert {"Max R-hat", "Min ESS Bulk", "Divergences", "F1 Score"} <= set(
        results.columns
    )
    assert len(posteriors) == 2
