"""Tests for the periodontal anatomy."""

from periomod.anatomy import (
    get_arch_neighbors,
    get_arch_pairs,
    get_interproximal_pairs,
    get_occlusal_pairs,
    is_mesial_neighbor,
    is_midline_pair,
)

ASPECTS = [(3, 1), (4, 6)]


def test_arch_neighbors_within_arch():
    """Test that adjacency follows the dental arches."""
    neighbors = get_arch_neighbors()
    assert neighbors[16] == [17, 15]
    assert neighbors[18] == [17]
    assert neighbors[28] == [27]
    assert sorted(neighbors[11]) == [12, 21]
    assert sorted(neighbors[31]) == [32, 41]
    assert 38 not in neighbors[18]


def test_arch_pairs_are_unique():
    """Test that every adjacent pair of teeth is listed exactly once."""
    pairs = get_arch_pairs()
    assert len(pairs) == 30
    assert len({frozenset(pair) for pair in pairs}) == 30


def test_occlusal_pairs():
    """Test that occluding teeth connect the upper and lower arch."""
    pairs = get_occlusal_pairs()
    assert len(pairs) == 14
    assert all(tooth < 30 < neighbor for tooth, neighbor in pairs)


def test_mesial_neighbor():
    """Test the orientation of adjacent teeth within a quadrant."""
    assert is_mesial_neighbor(tooth=16, neighbor=15)
    assert not is_mesial_neighbor(tooth=15, neighbor=16)
    assert not is_mesial_neighbor(tooth=11, neighbor=21)
    assert is_midline_pair(tooth=11, neighbor=21)
    assert not is_midline_pair(tooth=11, neighbor=12)


def test_interproximal_pairs():
    """Test that interproximal contacts connect mesial and distal sites."""
    assert get_interproximal_pairs(tooth=16, neighbor=15, aspects=ASPECTS) == [
        (3, 1),
        (4, 6),
    ]
    assert get_interproximal_pairs(tooth=15, neighbor=16, aspects=ASPECTS) == [
        (1, 3),
        (6, 4),
    ]
    assert get_interproximal_pairs(tooth=11, neighbor=21, aspects=ASPECTS) == [
        (3, 3),
        (4, 4),
    ]
