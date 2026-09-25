"""Anatomical structure of the periodontal chart."""

from typing import Dict, List, Sequence, Tuple

from .base import all_teeth
from .data._helpers import _get_occluding_teeth


def get_arch_neighbors() -> Dict[int, List[int]]:
    """Maps every tooth to its adjacent teeth within the same dental arch.

    Adjacency follows the order of `all_teeth`, which lists the upper arch from
    18 to 28 and the lower arch from 48 to 38. Consecutive teeth of an arch are
    neighbors, including the two pairs of central incisors across the midline.

    Returns:
        Dict[int, List[int]]: Mapping of tooth number to adjacent tooth numbers.

    Example:
        ```
        from periomod.anatomy import get_arch_neighbors

        neighbors = get_arch_neighbors()
        print(neighbors[16])
        ```
    """
    neighbors: Dict[int, List[int]] = {tooth: [] for tooth in all_teeth}
    for arch in (all_teeth[:16], all_teeth[16:]):
        for left, right in zip(arch[:-1], arch[1:], strict=True):
            neighbors[left].append(right)
            neighbors[right].append(left)
    return neighbors


def get_arch_pairs() -> List[Tuple[int, int]]:
    """Lists every pair of adjacent teeth within a dental arch exactly once.

    Returns:
        List[Tuple[int, int]]: Pairs of adjacent tooth numbers.
    """
    return [
        (left, right)
        for arch in (all_teeth[:16], all_teeth[16:])
        for left, right in zip(arch[:-1], arch[1:], strict=True)
    ]


def get_occlusal_pairs() -> List[Tuple[int, int]]:
    """Lists every pair of occluding teeth of the upper and lower arch.

    Returns:
        List[Tuple[int, int]]: Pairs of occluding tooth numbers.
    """
    return [tuple(sorted(pair)) for pair in _get_occluding_teeth()]


def is_mesial_neighbor(tooth: int, neighbor: int) -> bool:
    """Evaluates whether a neighboring tooth lies mesially of a tooth.

    Within a quadrant the tooth with the lower unit digit lies closer to the
    midline and is therefore the mesial neighbor.

    Args:
        tooth (int): Tooth number in FDI notation.
        neighbor (int): Adjacent tooth number in FDI notation.

    Returns:
        bool: True if `neighbor` lies mesially of `tooth`.
    """
    quadrant, unit = divmod(tooth, 10)
    neighbor_quadrant, neighbor_unit = divmod(neighbor, 10)
    if quadrant != neighbor_quadrant:
        return False
    return neighbor_unit < unit


def is_midline_pair(tooth: int, neighbor: int) -> bool:
    """Evaluates whether two adjacent teeth meet across the midline.

    Args:
        tooth (int): Tooth number in FDI notation.
        neighbor (int): Adjacent tooth number in FDI notation.

    Returns:
        bool: True if both teeth are central incisors of the same arch.
    """
    return tooth % 10 == 1 and neighbor % 10 == 1


def get_interproximal_pairs(
    tooth: int, neighbor: int, aspects: Sequence[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Determines the interproximal site pairs of two adjacent teeth.

    Each interproximal contact is resolved per aspect, connecting the mesial
    site of one tooth with the distal site of its neighbor on the vestibular
    and on the oral aspect. Central incisors meeting across the midline are
    connected mesially on both teeth.

    Args:
        tooth (int): Tooth number in FDI notation.
        neighbor (int): Adjacent tooth number in FDI notation.
        aspects (Sequence[Tuple[int, int]]): Interproximal contacts, given as
            (mesial side, distal side) pairs per aspect.

    Returns:
        List[Tuple[int, int]]: Pairs of (side of `tooth`, side of `neighbor`).

    Example:
        ```
        from periomod.anatomy import get_interproximal_pairs

        pairs = get_interproximal_pairs(
            tooth=16, neighbor=15, aspects=[(3, 1), (4, 6)]
        )
        ```
    """
    if is_midline_pair(tooth=tooth, neighbor=neighbor):
        return [(mesial, mesial) for mesial, _ in aspects]
    if is_mesial_neighbor(tooth=tooth, neighbor=neighbor):
        return [(mesial, distal) for mesial, distal in aspects]
    return [(distal, mesial) for mesial, distal in aspects]
