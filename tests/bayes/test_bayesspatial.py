"""Tests for the anatomical adjacency of the spatial priors."""

import numpy as np
import pytest

from periomod.bayes import (
    HierarchicalDataTransformer,
    SpatialAdjacency,
    SpatialConfig,
    group_sizes,
)


@pytest.fixture
def train(data):
    """Provides the training design of the synthetic dataset.

    Args:
        data (pd.DataFrame): Synthetic processed dataset.

    Returns:
        HierarchicalData: Design matrix and nesting structure.
    """
    transformer = HierarchicalDataTransformer(task="pocketclosure", verbose=False)
    return transformer.transform_splits(data=data)["train"]


def test_tooth_edges_stay_within_patients(train):
    """Test that adjacent teeth belong to the same patient."""
    adjacency = SpatialAdjacency()
    node1, node2 = adjacency.tooth_edges(
        data=train, spatial=SpatialConfig(mode="tooth")
    )

    assert node1.size > 0
    assert (train.tooth_patient[node1] == train.tooth_patient[node2]).all()
    assert (node1 != node2).all()


def test_tooth_edges_follow_the_arch(train):
    """Test that only anatomically adjacent teeth are connected."""
    adjacency = SpatialAdjacency()
    node1, node2 = adjacency.tooth_edges(
        data=train, spatial=SpatialConfig(mode="tooth")
    )
    pairs = {
        (int(train.tooth_number[first]), int(train.tooth_number[second]))
        for first, second in zip(node1, node2, strict=True)
    }

    assert pairs <= {(15, 16), (14, 15), (11, 21)}


def test_site_edges_cover_ring_and_interproximal(train):
    """Test that site adjacency contains both anatomical relations."""
    adjacency = SpatialAdjacency()
    ring_only = SpatialConfig(mode="site", interproximal=False)
    both = SpatialConfig(mode="site")

    ring_node1, _ = adjacency.site_edges(data=train, spatial=ring_only)
    node1, node2 = adjacency.site_edges(data=train, spatial=both)

    assert ring_node1.size < node1.size
    assert (train.patient_idx[node1] == train.patient_idx[node2]).all()


def test_relations_can_be_switched_off(train):
    """Test that disabling every relation removes all edges."""
    adjacency = SpatialAdjacency()
    spatial = SpatialConfig(
        mode="both", tooth_neighbor=False, site_neighbor=False, interproximal=False
    )

    assert adjacency.tooth_edges(data=train, spatial=spatial)[0].size == 0
    assert adjacency.site_edges(data=train, spatial=spatial)[0].size == 0


def test_group_sizes():
    """Test that group sizes count the units per group."""
    sizes = group_sizes(index=np.array([0, 0, 2]), n_groups=3)
    assert sizes.tolist() == [2.0, 0.0, 1.0]


def test_invalid_spatial_mode_raises():
    """Test that an unknown spatial mode raises a ValueError."""
    with pytest.raises(ValueError, match="invalid spatial mode"):
        SpatialConfig(mode="anatomical")
