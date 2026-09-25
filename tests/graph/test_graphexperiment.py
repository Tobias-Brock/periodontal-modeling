"""Tests for graph training and experiments."""

import numpy as np
import pytest

from periomod.graph import EdgeConfig, GraphExperiment, GraphTrainer


@pytest.fixture
def experiment(data):
    """Provides a small graph experiment on synthetic data.

    Args:
        data (pd.DataFrame): Synthetic processed dataset.

    Returns:
        GraphExperiment: Experiment with a small model and few epochs.
    """
    return GraphExperiment(
        data=data,
        task="improvement",
        criterion="f1",
        hidden_dim=8,
        num_layers=2,
        epochs=2,
        patience=2,
        batch_size=4,
        verbose=False,
    )


def test_prepare_data_creates_loaders(experiment):
    """Test that loaders are created for all three splits."""
    loaders = experiment.prepare_data()

    assert set(loaders) == {"train", "val", "test"}
    assert len(experiment.graphs["train"]) == len(experiment.transformer.split.train)


def test_perform_evaluation_returns_site_predictions(experiment):
    """Test that the experiment reports validation metrics and predictions."""
    result = experiment.perform_evaluation()
    predictions = result["predictions"]

    assert "F1 Score" in result["metrics"]
    assert "test_metrics" not in result
    assert set(predictions.columns) == {"id_patient", "tooth", "side", "y", "prob"}
    assert predictions["prob"].between(0, 1).all()
    assert set(predictions["id_patient"]) <= set(experiment.transformer.split.val)


def test_test_split_is_evaluated_on_demand(experiment):
    """Test that test metrics are only computed when requested."""
    result = experiment.perform_evaluation(evaluate_test=True)

    assert "test_metrics" in result
    assert set(result["test_predictions"]["id_patient"]) <= set(
        experiment.transformer.split.test
    )


def test_multiclass_experiment(data):
    """Test that the multiclass task predicts one probability per class."""
    experiment = GraphExperiment(
        data=data,
        task="pdgrouprevaluation",
        criterion="macro_f1",
        hidden_dim=8,
        num_layers=2,
        epochs=1,
        patience=1,
        batch_size=4,
        verbose=False,
    )
    result = experiment.perform_evaluation()

    assert experiment.classification == "multiclass"
    assert "Macro F1" in result["metrics"]
    assert "prob_2" in result["predictions"].columns


def test_edge_ablations_run(data):
    """Test that the predefined edge ablations produce comparable results."""
    scores = {}
    for name, edges in EdgeConfig.ablations(seed=0).items():
        if name not in ["full", "no_neighbors", "random_neighbors"]:
            continue
        experiment = GraphExperiment(
            data=data,
            task="improvement",
            criterion="brier_score",
            edges=edges,
            hidden_dim=8,
            num_layers=2,
            epochs=1,
            patience=1,
            batch_size=4,
            verbose=False,
        )
        scores[name] = experiment.perform_evaluation()["val_score"]

    assert set(scores) == {"full", "no_neighbors", "random_neighbors"}
    assert all(np.isfinite(score) for score in scores.values())


def test_trainer_criterion_direction():
    """Test that the Brier score is minimized and the F1 score maximized."""
    assert GraphTrainer(classification="binary", criterion="f1").maximize
    assert not GraphTrainer(classification="binary", criterion="brier_score").maximize


def test_invalid_task_raises(data):
    """Test that an unknown task raises a ValueError."""
    with pytest.raises(ValueError, match="unknown"):
        GraphExperiment(data=data, task="unknown", criterion="f1", verbose=False)
