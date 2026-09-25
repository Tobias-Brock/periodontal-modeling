"""Tests for split-aware preprocessing of graph data."""

import numpy as np
import pytest

from periomod.graph import GraphDataTransformer


def test_split_is_disjoint_on_patients(data):
    """Test that patients do not overlap between the splits."""
    transformer = GraphDataTransformer(task="improvement", verbose=False)
    split = transformer.split_patients(data=data)

    assert set(split.train).isdisjoint(split.val)
    assert set(split.train).isdisjoint(split.test)
    assert set(split.val).isdisjoint(split.test)
    assert sum(split.sizes.values()) == data["id_patient"].nunique()


def test_levels_assign_columns_to_nodes(data):
    """Test that columns are assigned to their node level and kind."""
    transformer = GraphDataTransformer(task="improvement", verbose=False)
    levels = transformer.resolve_levels(data=data)

    assert "age" in levels["patient"]["numeric"]
    assert "smokingtype" in levels["patient"]["categorical"]
    assert "mobility" in levels["tooth"]["binary"]
    assert "tooth" in levels["tooth"]["categorical"]
    assert "pdbaseline" in levels["site"]["numeric"]
    assert "side" in levels["site"]["categorical"]
    assert "pdrevaluation" not in levels["site"]["numeric"]
    assert "id_patient" not in levels["patient"]["numeric"]


def test_scaling_is_fitted_on_training_patients(data):
    """Test that scaling statistics are derived from training patients only."""
    transformer = GraphDataTransformer(task="improvement", verbose=False)
    splits = transformer.transform_splits(data=data)

    assert np.isclose(splits["train"]["pdbaseline"].mean(), 0, atol=1e-6)
    assert not np.isclose(splits["test"]["pdbaseline"].mean(), 0, atol=1e-6)


def test_target_mask_restricted_to_diseased_sites(data):
    """Test that only sites above 3 mm are targets for the improvement task."""
    transformer = GraphDataTransformer(task="improvement", verbose=False)
    splits = transformer.transform_splits(data=data)
    train = splits["train"]
    raw = data[data["id_patient"].isin(transformer.split.train)].reset_index(drop=True)

    assert train["target_mask"].sum() == int((raw["pdbaseline"] > 3).sum())
    assert len(train) == len(raw)


def test_target_mask_keeps_all_sites_for_pocketclosure(data):
    """Test that all sites are targets when the task is not restricted."""
    transformer = GraphDataTransformer(task="pocketclosure", verbose=False)
    splits = transformer.transform_splits(data=data)

    assert splits["train"]["target_mask"].all()


def test_unknown_task_raises(data):
    """Test that an unsupported task raises a ValueError."""
    with pytest.raises(ValueError, match="not supported"):
        GraphDataTransformer(task="unknown", verbose=False)


def test_transform_before_fit_raises(data):
    """Test that transforming without fitting raises a ValueError."""
    transformer = GraphDataTransformer(task="improvement", verbose=False)
    with pytest.raises(ValueError, match="must be fitted"):
        transformer.transform_data(data=data)
