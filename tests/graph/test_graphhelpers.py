"""Tests for graph edge utilities."""

import numpy as np

from periomod.graph._helpers import _randomize_edge_index, _to_undirected


def test_to_undirected():
    """Test that node pairs are converted into bidirectional edges."""
    index = _to_undirected(edges=[(0, 1), (1, 2)])
    assert index.shape == (2, 4)
    assert set(zip(index[0], index[1], strict=True)) == {
        (0, 1),
        (1, 2),
        (1, 0),
        (2, 1),
    }
    assert _to_undirected(edges=[]).shape == (2, 0)


def test_randomize_edge_index():
    """Test that randomization preserves the number of edges."""
    edge_index = np.array([[0, 1, 2], [1, 2, 3]])
    rng = np.random.default_rng(seed=0)
    randomized = _randomize_edge_index(
        edge_index=edge_index, num_src=6, num_dst=6, rng=rng, same_type=True
    )
    assert randomized.shape == edge_index.shape
    assert not (randomized[0] == randomized[1]).any()
    assert randomized.max() < 6
