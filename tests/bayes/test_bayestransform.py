"""Tests for split-aware preprocessing of hierarchical data."""

import numpy as np
import pytest

from periomod.bayes import HierarchicalDataTransformer


def test_split_is_disjoint_on_patients(data):
    """Test that patients do not overlap between the splits."""
    transformer = HierarchicalDataTransformer(task="improvement", verbose=False)
    split = transformer.split_patients(data=data)

    assert set(split.train).isdisjoint(split.val)
    assert set(split.train).isdisjoint(split.test)
    assert set(split.val).isdisjoint(split.test)
    assert sum(split.sizes.values()) == data["id_patient"].nunique()


def test_split_matches_graph_submodule(data):
    """Test that both submodules split the patients identically."""
    pytest.importorskip("torch_geometric")
    from periomod.graph import GraphDataTransformer

    bayes_split = HierarchicalDataTransformer(
        task="improvement", verbose=False
    ).split_patients(data=data)
    graph_split = GraphDataTransformer(
        task="improvement", verbose=False
    ).split_patients(data=data)

    assert bayes_split.train == graph_split.train
    assert bayes_split.val == graph_split.val
    assert bayes_split.test == graph_split.test


def test_design_uses_compact_dummy_coding(data):
    """Test that categorical predictors drop their reference level."""
    transformer = HierarchicalDataTransformer(task="improvement", verbose=False)
    splits = transformer.transform_splits(data=data)
    names = splits["train"].feature_names

    assert "diabetes_1" not in names
    assert "diabetes_2" in names
    assert "tooth" not in names
    assert not any(name.startswith("tooth_1") for name in names)
    assert not any(name.startswith("side_") for name in names)
    assert splits["train"].X.shape[1] == len(names)


def test_nesting_structure_is_consistent(data):
    """Test that the index arrays encode sites in teeth in patients."""
    transformer = HierarchicalDataTransformer(task="improvement", verbose=False)
    train = transformer.transform_splits(data=data)["train"]

    assert (train.tooth_patient[train.tooth_idx] == train.patient_idx).all()
    assert train.n_teeth <= train.n_obs
    assert train.n_patients <= train.n_teeth
    assert set(np.unique(train.side_idx)) <= set(range(6))
    assert train.toothnum_idx.max() < 32


def test_scaling_is_fitted_on_training_patients(data):
    """Test that standardization uses training statistics only."""
    transformer = HierarchicalDataTransformer(task="improvement", verbose=False)
    splits = transformer.transform_splits(data=data)
    index = splits["train"].feature_names.index("pdbaseline")

    assert abs(splits["train"].X[:, index].mean()) < 0.5
    assert not np.isclose(
        splits["train"].X[:, index].mean(), splits["test"].X[:, index].mean()
    )


def test_target_sites_follow_the_task(data):
    """Test that the likelihood is restricted to the target sites."""
    diseased = HierarchicalDataTransformer(task="improvement", verbose=False)
    diseased_splits = diseased.transform_splits(data=data)
    every = HierarchicalDataTransformer(task="pocketclosure", verbose=False)
    every_splits = every.transform_splits(data=data)

    raw = data[data["id_patient"].isin(diseased.split.train)]
    assert diseased_splits["train"].n_obs == int((raw["pdbaseline"] > 3).sum())
    assert every_splits["train"].n_obs > diseased_splits["train"].n_obs


def test_unsupported_task_raises(data):
    """Test that multiclass tasks are rejected with a clear message."""
    with pytest.raises(ValueError, match="not supported"):
        HierarchicalDataTransformer(task="pdgrouprevaluation", verbose=False)
