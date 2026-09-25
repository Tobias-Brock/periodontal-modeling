from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv, SAGEConv
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "The graph submodule requires 'torch_geometric'. Install the optional "
        "dependencies with 'pip install periomod[gnn]'."
    ) from error


def graph_dimensions(graph: HeteroData) -> Dict[str, int]:
    """Determines the number of input features per node level of a graph.

    Args:
        graph (HeteroData): Patient graph built by `PatientGraphBuilder`.

    Returns:
        Dict[str, int]: Number of input features per node type.
    """
    return {
        node_type: int(graph[node_type].x.size(-1)) for node_type in graph.node_types
    }


class HeteroSitePredictor(nn.Module):
    """Heterogeneous GraphSAGE model for site level predictions.

    Each node level is first projected to a shared hidden dimension. Message
    passing is then applied over the relations of the patient graph, so that
    information propagates from sites to teeth, to neighboring teeth and the
    patient context, and back to the site nodes. A site level head maps the
    resulting site embeddings to the outcome.

    Every message passing stage additionally applies a node type specific
    transformation of the node itself, which keeps node levels updated that
    receive no messages in an edge ablation. The convolutions are built from
    the relations present in the data, so that removing a relation removes the
    corresponding message passing without further changes to the model.

    Inherits:
        - `nn.Module`: PyTorch base class of neural network modules.

    Args:
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]): Node and edge
            types of the patient graphs, as returned by `HeteroData.metadata`.
        in_channels (Dict[str, int]): Number of input features per node type.
        hidden_dim (int): Hidden dimension of the model. Defaults to 64.
        num_layers (int): Number of message passing stages. Defaults to 3.
        dropout (float): Dropout rate applied between stages and in the site
            level head. Defaults to 0.3.
        out_channels (int): Number of outputs per site. Use 1 for binary
            classification and the number of classes otherwise. Defaults to 1.
        conv_aggr (str): Neighborhood aggregation of the convolutions.
            Defaults to "mean".
        relation_aggr (str): Aggregation across relations. Defaults to "sum".
        layer_norm (bool): Applies layer normalization between the message
            passing stages. Defaults to True.

    Attributes:
        encoders (nn.ModuleDict): Input projection per node type.
        convs (nn.ModuleList): Heterogeneous convolution per stage.
        roots (nn.ModuleList): Node type specific self transformation per stage.
        norms (nn.ModuleList): Layer normalization per stage and node type.
        head (nn.Sequential): Site level prediction head.
        dropout (float): Dropout rate of the model.

    Example:
        ```
        from periomod.graph import HeteroSitePredictor, graph_dimensions

        model = HeteroSitePredictor(
            metadata=graphs[0].metadata(),
            in_channels=graph_dimensions(graph=graphs[0]),
            hidden_dim=64,
            num_layers=3,
        )
        logits = model(x_dict=batch.x_dict, edge_index_dict=batch.edge_index_dict)
        ```
    """

    def __init__(
        self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        in_channels: Dict[str, int],
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        out_channels: int = 1,
        conv_aggr: str = "mean",
        relation_aggr: str = "sum",
        layer_norm: bool = True,
    ) -> None:
        """Initializes the heterogeneous GraphSAGE model."""
        super().__init__()
        node_types, edge_types = metadata
        self.dropout = dropout
        self.encoders = nn.ModuleDict({
            node_type: nn.Linear(in_channels[node_type], hidden_dim)
            for node_type in node_types
        })
        self.convs = nn.ModuleList()
        self.roots = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                HeteroConv(
                    {
                        edge_type: SAGEConv(
                            (hidden_dim, hidden_dim), hidden_dim, aggr=conv_aggr
                        )
                        for edge_type in edge_types
                    },
                    aggr=relation_aggr,
                )
            )
            self.roots.append(
                nn.ModuleDict({
                    node_type: nn.Linear(hidden_dim, hidden_dim)
                    for node_type in node_types
                })
            )
            self.norms.append(
                nn.ModuleDict({
                    node_type: nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
                    for node_type in node_types
                })
            )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> torch.Tensor:
        """Computes the site level logits of a batch of patient graphs.

        Args:
            x_dict (Dict[str, torch.Tensor]): Node features per node type.
            edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): Edge
                index per relation.

        Returns:
            torch.Tensor: Site level logits of shape (n_sites, out_channels).
        """
        hidden = {
            node_type: torch.relu(self.encoders[node_type](x))
            for node_type, x in x_dict.items()
        }

        for conv, root, norm in zip(self.convs, self.roots, self.norms, strict=True):
            messages = conv(hidden, edge_index_dict)
            updated = {}
            for node_type, x in hidden.items():
                out = root[node_type](x)
                if node_type in messages:
                    out = out + messages[node_type]
                out = torch.relu(norm[node_type](out))
                updated[node_type] = nn.functional.dropout(
                    out, p=self.dropout, training=self.training
                )
            hidden = updated

        return self.head(hidden["site"])


def build_model(
    graph: HeteroData,
    classification: str,
    hidden_dim: int = 64,
    num_layers: int = 3,
    dropout: float = 0.3,
    conv_aggr: str = "mean",
    relation_aggr: str = "sum",
    layer_norm: bool = True,
    n_classes: Optional[int] = None,
    seed: Optional[int] = None,
) -> HeteroSitePredictor:
    """Instantiates a site predictor matching the structure of a patient graph.

    Args:
        graph (HeteroData): Patient graph providing node and edge types as well
            as the number of input features per node level.
        classification (str): Type of classification, either 'binary' or
            'multiclass'.
        hidden_dim (int): Hidden dimension of the model. Defaults to 64.
        num_layers (int): Number of message passing stages. Defaults to 3.
        dropout (float): Dropout rate. Defaults to 0.3.
        conv_aggr (str): Neighborhood aggregation of the convolutions.
            Defaults to "mean".
        relation_aggr (str): Aggregation across relations. Defaults to "sum".
        layer_norm (bool): Applies layer normalization. Defaults to True.
        n_classes (Optional[int]): Number of classes for multiclass
            classification. Defaults to 3.
        seed (Optional[int]): Random state of the model initialization.
            Defaults to None.

    Returns:
        HeteroSitePredictor: Model matching the relations of the patient graphs.
    """
    if seed is not None:
        torch.manual_seed(seed)
    out_channels = 1 if classification == "binary" else (n_classes or 3)
    return HeteroSitePredictor(
        metadata=graph.metadata(),
        in_channels=graph_dimensions(graph=graph),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        out_channels=out_channels,
        conv_aggr=conv_aggr,
        relation_aggr=relation_aggr,
        layer_norm=layer_norm,
    )
