from typing import List, Tuple

import numpy as np


def _to_undirected(edges: List[Tuple[int, int]]) -> np.ndarray:
    """Converts a list of node pairs into a bidirectional edge index.

    Args:
        edges (List[Tuple[int, int]]): Pairs of node indices.

    Returns:
        np.ndarray: Edge index of shape (2, 2 * len(edges)).
    """
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    index = np.array(edges, dtype=np.int64).T
    return np.concatenate([index, index[::-1]], axis=1)


def _randomize_edge_index(
    edge_index: np.ndarray,
    num_src: int,
    num_dst: int,
    rng: np.random.Generator,
    same_type: bool,
) -> np.ndarray:
    """Rewires an edge index randomly while preserving the number of edges.

    Used for the 'random' edge mode, which keeps the amount of connectivity of
    an anatomical relation but destroys its anatomical meaning.

    Args:
        edge_index (np.ndarray): Edge index of shape (2, n_edges).
        num_src (int): Number of source nodes in the patient graph.
        num_dst (int): Number of destination nodes in the patient graph.
        rng (np.random.Generator): Random generator of the patient graph.
        same_type (bool): Indicates whether source and destination node type
            are identical, in which case self loops are avoided.

    Returns:
        np.ndarray: Randomly rewired edge index of shape (2, n_edges).
    """
    n_edges = edge_index.shape[1]
    if n_edges == 0 or num_src == 0 or num_dst == 0:
        return np.zeros((2, 0), dtype=np.int64)
    if same_type and num_src < 2:
        return np.zeros((2, 0), dtype=np.int64)

    src = rng.integers(low=0, high=num_src, size=n_edges)
    dst = rng.integers(low=0, high=num_dst, size=n_edges)
    if same_type:
        collisions = src == dst
        while collisions.any():
            dst[collisions] = rng.integers(
                low=0, high=num_dst, size=int(collisions.sum())
            )
            collisions = src == dst
    return np.stack([src, dst]).astype(np.int64)
