from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ..anatomy import get_arch_pairs, get_interproximal_pairs, get_occlusal_pairs
from ._basegraph import BaseGraphBuilder, EdgeConfig
from ._helpers import _randomize_edge_index, _to_undirected

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.loader import DataLoader
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "The graph submodule requires 'torch_geometric'. Install the optional "
        "dependencies with 'pip install periomod[gnn]'."
    ) from error

PATIENT_TOOTH = ("patient", "has", "tooth")
TOOTH_PATIENT = ("tooth", "in", "patient")
TOOTH_SITE = ("tooth", "has", "site")
SITE_TOOTH = ("site", "in", "tooth")
SITE_NEIGHBOR = ("site", "neighbors", "site")
SITE_INTERPROXIMAL = ("site", "interproximal", "site")
TOOTH_NEIGHBOR = ("tooth", "adjacent", "tooth")
TOOTH_OCCLUSAL = ("tooth", "occludes", "tooth")


def graph_loader(
    graphs: List[HeteroData], batch_size: int, shuffle: bool = False
) -> DataLoader:
    """Creates a loader over batches of complete patient graphs.

    Args:
        graphs (List[HeteroData]): Patient graphs of one split.
        batch_size (int): Number of patient graphs per batch.
        shuffle (bool): Shuffles the patient graphs. Defaults to False.

    Returns:
        DataLoader: Loader yielding batches of patient graphs.
    """
    return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle)


class PatientGraphBuilder(BaseGraphBuilder):
    """Builds one heterogeneous graph per patient from transformed data.

    Every patient is represented by a single graph with three node levels. The
    patient node carries the anamnestic features, one tooth node per present
    tooth carries the tooth level findings and one site node per examined site
    carries the baseline site measurements. Missing teeth receive neither tooth
    nor site nodes. All baseline sites remain in the graph as context, while the
    `target_mask` of the site nodes marks the sites entering loss and evaluation.

    The graph contains up to six biologically predefined relations, each of
    which can be switched off individually through `EdgeConfig`:

    - 'patient_tooth': patient node and each of its tooth nodes,
    - 'tooth_site': tooth node and its six site nodes,
    - 'site_neighbor': anatomically neighboring sites around the same tooth,
    - 'tooth_neighbor': adjacent teeth within the same dental arch,
    - 'interproximal': distal sites of a tooth and mesial sites of the
      adjacent tooth, resolved per vestibular and oral aspect,
    - 'occlusal': occluding teeth of the upper and lower arch.

    All relations are stored in both directions, so that information propagates
    from sites to teeth, to neighboring teeth and the patient context, and back
    to the sites.

    Inherits:
        - `BaseGraphBuilder`: Provides configuration and abstract build methods.

    Args:
        feature_names (Dict[str, List[str]]): Transformed feature columns per
            node level, as returned by `GraphDataTransformer`.
        edges (Optional[EdgeConfig]): Relations included in the graph. Defaults
            to the configured relations.

    Attributes:
        feature_names (Dict[str, List[str]]): Feature columns per node level.
        edges (EdgeConfig): Relations included in the patient graphs.

    Methods:
        build_patient_graph: Builds the heterogeneous graph of one patient.
        build: Builds the heterogeneous graphs of all patients in a dataset.

    Example:
        ```
        from periomod.graph import EdgeConfig, PatientGraphBuilder

        builder = PatientGraphBuilder(
            feature_names=transformer.feature_names,
            edges=EdgeConfig(mode="random", name="random_neighbors"),
        )
        graphs = builder.build(data=splits["train"])
        print(graphs[0])
        ```
    """

    def __init__(
        self,
        feature_names: Dict[str, List[str]],
        edges: Optional[EdgeConfig] = None,
    ) -> None:
        """Initializes the builder with feature columns and relations."""
        super().__init__(feature_names=feature_names, edges=edges)
        self.arch_pairs = get_arch_pairs()
        self.occlusal_pairs = get_occlusal_pairs()

    def _relation_index(
        self,
        pairs: List[Tuple[int, int]],
        num_nodes: int,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        """Builds a bidirectional edge index of an anatomical relation.

        In the 'random' edge mode the node pairs are rewired randomly within the
        patient graph, which preserves the number of edges of the relation but
        removes its anatomical meaning.

        Args:
            pairs (List[Tuple[int, int]]): Anatomically connected node pairs.
            num_nodes (int): Number of nodes of the node type.
            rng (np.random.Generator): Random generator of the patient graph.

        Returns:
            torch.Tensor: Edge index of shape (2, 2 * n_pairs).
        """
        if self.edges.mode == "random":
            index = _randomize_edge_index(
                edge_index=np.array(pairs, dtype=np.int64).T
                if pairs
                else np.zeros((2, 0), dtype=np.int64),
                num_src=num_nodes,
                num_dst=num_nodes,
                rng=rng,
                same_type=True,
            )
            index = np.concatenate([index, index[::-1]], axis=1)
        else:
            index = _to_undirected(edges=pairs)
        return torch.from_numpy(np.ascontiguousarray(index)).long()

    def _site_neighbor_pairs(
        self, site_pos: Dict[Tuple[int, int], int], teeth: List[int]
    ) -> List[Tuple[int, int]]:
        """Collects the neighboring site pairs around each tooth.

        Args:
            site_pos (Dict[Tuple[int, int], int]): Node index per (tooth, side).
            teeth (List[int]): Present teeth of the patient.

        Returns:
            List[Tuple[int, int]]: Pairs of neighboring site node indices.
        """
        pairs = []
        for tooth in teeth:
            for side, neighbor_side in self.side_ring:
                if (tooth, side) in site_pos and (tooth, neighbor_side) in site_pos:
                    pairs.append((
                        site_pos[(tooth, side)],
                        site_pos[(tooth, neighbor_side)],
                    ))
        return pairs

    def _interproximal_pairs(
        self, site_pos: Dict[Tuple[int, int], int], teeth: List[int]
    ) -> List[Tuple[int, int]]:
        """Collects the interproximal site pairs of adjacent teeth.

        Args:
            site_pos (Dict[Tuple[int, int], int]): Node index per (tooth, side).
            teeth (List[int]): Present teeth of the patient.

        Returns:
            List[Tuple[int, int]]: Pairs of interproximal site node indices.
        """
        present = set(teeth)
        pairs = []
        for tooth, neighbor in self.arch_pairs:
            if tooth not in present or neighbor not in present:
                continue
            for side, neighbor_side in get_interproximal_pairs(
                tooth=tooth, neighbor=neighbor, aspects=self.interproximal_aspects
            ):
                if (tooth, side) in site_pos and (neighbor, neighbor_side) in site_pos:
                    pairs.append((
                        site_pos[(tooth, side)],
                        site_pos[(neighbor, neighbor_side)],
                    ))
        return pairs

    @staticmethod
    def _tooth_pairs(
        tooth_pos: Dict[int, int], pairs: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Maps tooth number pairs to the node indices of present teeth.

        Args:
            tooth_pos (Dict[int, int]): Node index per tooth number.
            pairs (List[Tuple[int, int]]): Pairs of tooth numbers.

        Returns:
            List[Tuple[int, int]]: Pairs of tooth node indices.
        """
        return [
            (tooth_pos[tooth], tooth_pos[neighbor])
            for tooth, neighbor in pairs
            if tooth in tooth_pos and neighbor in tooth_pos
        ]

    def build_patient_graph(self, patient_data: pd.DataFrame) -> HeteroData:
        """Builds the heterogeneous graph of a single patient.

        Args:
            patient_data (pd.DataFrame): Transformed rows of one patient, with
                one row per site.

        Returns:
            HeteroData: Patient graph with patient, tooth and site nodes, the
                configured relations, the site level outcome 'y' and the
                boolean 'target_mask' of the site nodes.
        """
        patient_data = patient_data.sort_values(by=["tooth", "side"])
        teeth = patient_data["tooth"].drop_duplicates().tolist()
        tooth_pos = {tooth: index for index, tooth in enumerate(teeth)}
        sites = list(
            zip(
                patient_data["tooth"].tolist(),
                patient_data["side"].tolist(),
                strict=True,
            )
        )
        site_pos = {site: index for index, site in enumerate(sites)}
        patient_id = int(patient_data[self.group_col].iloc[0])
        rng = np.random.default_rng(seed=self.edges.seed + patient_id)

        graph = HeteroData()
        graph["patient"].x = torch.tensor(
            patient_data[self.feature_names["patient"]].to_numpy()[:1],
            dtype=torch.float,
        )
        graph["tooth"].x = torch.tensor(
            patient_data.drop_duplicates(subset="tooth")[
                self.feature_names["tooth"]
            ].to_numpy(),
            dtype=torch.float,
        )
        graph["site"].x = torch.tensor(
            patient_data[self.feature_names["site"]].to_numpy(), dtype=torch.float
        )
        graph["tooth"].tooth_id = torch.tensor(teeth, dtype=torch.long)
        graph["site"].tooth_id = torch.tensor(
            patient_data["tooth"].to_numpy(), dtype=torch.long
        )
        graph["site"].side_id = torch.tensor(
            patient_data["side"].to_numpy(), dtype=torch.long
        )
        graph["site"].patient_id = torch.full(
            size=(len(sites),), fill_value=patient_id, dtype=torch.long
        )
        graph["site"].y = torch.tensor(patient_data["y"].to_numpy(), dtype=torch.float)
        graph["site"].target_mask = torch.tensor(
            patient_data["target_mask"].to_numpy(), dtype=torch.bool
        )

        relations = self.edges.active_relations()
        site_tooth = np.array([tooth_pos[tooth] for tooth, _ in sites], dtype=np.int64)

        if "patient_tooth" in relations:
            index = torch.stack([
                torch.zeros(len(teeth), dtype=torch.long),
                torch.arange(len(teeth), dtype=torch.long),
            ])
            graph[PATIENT_TOOTH].edge_index = index
            graph[TOOTH_PATIENT].edge_index = index.flip(0)

        if "tooth_site" in relations:
            index = torch.stack([
                torch.from_numpy(site_tooth),
                torch.arange(len(sites), dtype=torch.long),
            ])
            graph[TOOTH_SITE].edge_index = index
            graph[SITE_TOOTH].edge_index = index.flip(0)

        if "site_neighbor" in relations:
            graph[SITE_NEIGHBOR].edge_index = self._relation_index(
                pairs=self._site_neighbor_pairs(site_pos=site_pos, teeth=teeth),
                num_nodes=len(sites),
                rng=rng,
            )

        if "interproximal" in relations:
            graph[SITE_INTERPROXIMAL].edge_index = self._relation_index(
                pairs=self._interproximal_pairs(site_pos=site_pos, teeth=teeth),
                num_nodes=len(sites),
                rng=rng,
            )

        if "tooth_neighbor" in relations:
            graph[TOOTH_NEIGHBOR].edge_index = self._relation_index(
                pairs=self._tooth_pairs(tooth_pos=tooth_pos, pairs=self.arch_pairs),
                num_nodes=len(teeth),
                rng=rng,
            )

        if "occlusal" in relations:
            graph[TOOTH_OCCLUSAL].edge_index = self._relation_index(
                pairs=self._tooth_pairs(tooth_pos=tooth_pos, pairs=self.occlusal_pairs),
                num_nodes=len(teeth),
                rng=rng,
            )

        return graph

    def build(self, data: pd.DataFrame) -> List[HeteroData]:
        """Builds the heterogeneous graphs of all patients in a dataset.

        Args:
            data (pd.DataFrame): Transformed dataset with one row per site.

        Returns:
            List[HeteroData]: One patient graph per patient in the dataset.
        """
        return [
            self.build_patient_graph(patient_data=patient_data)
            for _, patient_data in data.groupby(self.group_col, sort=True)
        ]
