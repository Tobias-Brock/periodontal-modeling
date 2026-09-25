from typing import Dict, Tuple

import numpy as np

from ..anatomy import get_arch_neighbors, get_interproximal_pairs
from ._basebayes import BaseBayesConfig, HierarchicalData, SpatialConfig


def group_sizes(index: np.ndarray, n_groups: int) -> np.ndarray:
    """Counts the number of units per group.

    Args:
        index (np.ndarray): Group index of every unit.
        n_groups (int): Number of groups.

    Returns:
        np.ndarray: Number of units per group.
    """
    return np.bincount(index, minlength=n_groups).astype(float)


class SpatialAdjacency(BaseBayesConfig):
    """Builds the anatomical adjacency used by the spatial priors.

    The adjacency is derived from the same periodontal anatomy as the relations
    of the graph submodule, which makes the conditional autoregressive priors
    and the message passing of the GNN comparable. Teeth are connected to their
    neighbors within a dental arch, sites to the anatomically neighboring sites
    around their tooth and to the interproximal sites of the adjacent tooth.

    Edges are only formed between units that are present in the design, so the
    adjacency covers the teeth and sites actually entering the likelihood.

    Inherits:
        - `BaseBayesConfig`: Provides package and Bayesian configuration.

    Attributes:
        arch_neighbors (Dict[int, List[int]]): Adjacent teeth per tooth number.

    Methods:
        tooth_edges: Adjacent tooth pairs of a split.
        site_edges: Adjacent site pairs of a split.

    Example:
        ```
        from periomod.bayes import SpatialAdjacency, SpatialConfig

        adjacency = SpatialAdjacency()
        node1, node2 = adjacency.tooth_edges(
            data=splits["train"], spatial=SpatialConfig(mode="tooth")
        )
        ```
    """

    def __init__(self) -> None:
        """Initializes the adjacency with the periodontal anatomy."""
        super().__init__()
        self.arch_neighbors = get_arch_neighbors()

    def tooth_edges(
        self, data: HierarchicalData, spatial: SpatialConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collects the adjacent tooth pairs of a split.

        Args:
            data (HierarchicalData): Design matrix and nesting structure.
            spatial (SpatialConfig): Spatial configuration of the model.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tooth indices of every edge, with
                each undirected pair listed exactly once.
        """
        if not spatial.tooth_neighbor:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

        position: Dict[Tuple[int, int], int] = {
            (int(patient), int(tooth)): index
            for index, (patient, tooth) in enumerate(
                zip(data.tooth_patient, data.tooth_number, strict=True)
            )
        }
        node1, node2 = [], []
        for (patient, tooth), index in position.items():
            for neighbor in self.arch_neighbors[tooth]:
                if neighbor <= tooth:
                    continue
                neighbor_index = position.get((patient, neighbor))
                if neighbor_index is not None:
                    node1.append(index)
                    node2.append(neighbor_index)

        return np.array(node1, dtype=np.int64), np.array(node2, dtype=np.int64)

    def site_edges(
        self, data: HierarchicalData, spatial: SpatialConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collects the adjacent site pairs of a split.

        Args:
            data (HierarchicalData): Design matrix and nesting structure.
            spatial (SpatialConfig): Spatial configuration of the model.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Observation indices of every edge,
                with each undirected pair listed exactly once.
        """
        position: Dict[Tuple[int, int, int], int] = {
            (int(patient), int(tooth), int(side)): index
            for index, (patient, tooth, side) in enumerate(
                zip(
                    data.patient_idx,
                    data.keys["tooth"].to_numpy(),
                    data.keys["side"].to_numpy(),
                    strict=True,
                )
            )
        }
        edges = set()

        if spatial.site_neighbor:
            for patient, tooth, side in position:
                for first, second in self.side_ring:
                    if side != first:
                        continue
                    neighbor = position.get((patient, tooth, second))
                    if neighbor is not None:
                        edges.add((
                            min(position[(patient, tooth, side)], neighbor),
                            max(position[(patient, tooth, side)], neighbor),
                        ))

        if spatial.interproximal:
            for patient, tooth, side in position:
                for neighbor_tooth in self.arch_neighbors[tooth]:
                    pairs = get_interproximal_pairs(
                        tooth=tooth,
                        neighbor=neighbor_tooth,
                        aspects=self.interproximal_aspects,
                    )
                    for own_side, neighbor_side in pairs:
                        if side != own_side:
                            continue
                        neighbor = position.get((
                            patient,
                            neighbor_tooth,
                            neighbor_side,
                        ))
                        if neighbor is not None:
                            edges.add((
                                min(position[(patient, tooth, side)], neighbor),
                                max(position[(patient, tooth, side)], neighbor),
                            ))

        if not edges:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

        pairs_array = np.array(sorted(edges), dtype=np.int64)
        return pairs_array[:, 0], pairs_array[:, 1]
