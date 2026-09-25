"""Tests for the heterogeneous site level model."""

import pytest
import torch

from periomod.graph import (
    EdgeConfig,
    GraphDataTransformer,
    PatientGraphBuilder,
    build_model,
    graph_dimensions,
    graph_loader,
)


@pytest.fixture
def graphs(data):
    """Provides patient graphs of the training split.

    Args:
        data (pd.DataFrame): Synthetic processed dataset.

    Returns:
        List[HeteroData]: Patient graphs with all anatomical relations.
    """
    transformer = GraphDataTransformer(task="improvement", verbose=False)
    splits = transformer.transform_splits(data=data)
    builder = PatientGraphBuilder(feature_names=transformer.feature_names)
    return builder.build(data=splits["train"])


def test_graph_dimensions(graphs):
    """Test that the input dimensions match the node features."""
    dimensions = graph_dimensions(graph=graphs[0])
    assert set(dimensions) == {"patient", "tooth", "site"}
    assert dimensions["site"] == graphs[0]["site"].x.size(-1)


def test_forward_returns_site_logits(graphs):
    """Test that the model predicts one logit per site node."""
    model = build_model(
        graph=graphs[0], classification="binary", hidden_dim=16, num_layers=2, seed=0
    )
    batch = next(iter(graph_loader(graphs=graphs, batch_size=4)))
    logits = model(x_dict=batch.x_dict, edge_index_dict=batch.edge_index_dict)

    assert logits.shape == (batch["site"].num_nodes, 1)
    assert torch.isfinite(logits).all()


def test_forward_multiclass(graphs):
    """Test that multiclass models predict one logit per class."""
    model = build_model(
        graph=graphs[0],
        classification="multiclass",
        hidden_dim=16,
        num_layers=2,
        n_classes=3,
        seed=0,
    )
    batch = next(iter(graph_loader(graphs=graphs, batch_size=2)))
    logits = model(x_dict=batch.x_dict, edge_index_dict=batch.edge_index_dict)

    assert logits.shape == (batch["site"].num_nodes, 3)


def test_forward_without_anatomical_relations(data):
    """Test that the model runs when anatomical relations are removed."""
    transformer = GraphDataTransformer(task="improvement", verbose=False)
    splits = transformer.transform_splits(data=data)
    builder = PatientGraphBuilder(
        feature_names=transformer.feature_names,
        edges=EdgeConfig(mode="none", name="no_neighbors"),
    )
    graphs = builder.build(data=splits["train"])
    model = build_model(
        graph=graphs[0], classification="binary", hidden_dim=16, num_layers=2, seed=0
    )
    batch = next(iter(graph_loader(graphs=graphs, batch_size=4)))
    logits = model(x_dict=batch.x_dict, edge_index_dict=batch.edge_index_dict)

    assert logits.shape == (batch["site"].num_nodes, 1)
    assert torch.isfinite(logits).all()


def test_seed_makes_initialization_reproducible(graphs):
    """Test that the model seed controls the initialization."""
    first = build_model(graph=graphs[0], classification="binary", seed=42)
    second = build_model(graph=graphs[0], classification="binary", seed=42)

    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(left, right)
