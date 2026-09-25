"""Tests for the construction of patient graphs."""

import pytest

from periomod.graph import (
    EdgeConfig,
    GraphDataTransformer,
    PatientGraphBuilder,
    graph_loader,
)

SITE_NEIGHBOR = ("site", "neighbors", "site")
INTERPROXIMAL = ("site", "interproximal", "site")
TOOTH_NEIGHBOR = ("tooth", "adjacent", "tooth")


@pytest.fixture
def splits(data):
    """Provides transformed splits and the fitted transformer.

    Args:
        data (pd.DataFrame): Synthetic processed dataset.

    Returns:
        Tuple[Dict[str, pd.DataFrame], GraphDataTransformer]: Transformed
            splits and the transformer used to create them.
    """
    transformer = GraphDataTransformer(task="improvement", verbose=False)
    return transformer.transform_splits(data=data), transformer


def test_graph_structure(splits):
    """Test node counts and relations of a patient graph."""
    split_data, transformer = splits
    builder = PatientGraphBuilder(feature_names=transformer.feature_names)
    graph = builder.build(data=split_data["train"])[0]

    assert graph["patient"].num_nodes == 1
    assert graph["tooth"].num_nodes == 5
    assert graph["site"].num_nodes == 30
    assert graph.validate()
    assert graph["site"].y.shape[0] == 30
    assert graph["site"].target_mask.dtype.is_floating_point is False
    assert graph[SITE_NEIGHBOR].edge_index.shape[1] == 2 * 6 * 5
    # adjacent teeth present in the fixture: 16-15, 15-14 and 11-21
    assert graph[TOOTH_NEIGHBOR].edge_index.shape[1] == 2 * 3
    assert graph[INTERPROXIMAL].edge_index.shape[1] == 2 * 2 * 3


def test_graph_keeps_all_baseline_sites(splits):
    """Test that healthy sites remain in the graph as context."""
    split_data, transformer = splits
    builder = PatientGraphBuilder(feature_names=transformer.feature_names)
    graphs = builder.build(data=split_data["train"])

    total_sites = sum(graph["site"].num_nodes for graph in graphs)
    target_sites = sum(int(graph["site"].target_mask.sum()) for graph in graphs)
    assert total_sites == len(split_data["train"])
    assert target_sites < total_sites


def test_edge_ablation_removes_anatomical_relations(splits):
    """Test that the 'none' edge mode keeps only the hierarchy."""
    split_data, transformer = splits
    builder = PatientGraphBuilder(
        feature_names=transformer.feature_names,
        edges=EdgeConfig(mode="none", name="no_neighbors"),
    )
    graph = builder.build(data=split_data["train"])[0]

    assert SITE_NEIGHBOR not in graph.edge_types
    assert INTERPROXIMAL not in graph.edge_types
    assert TOOTH_NEIGHBOR not in graph.edge_types
    assert ("patient", "has", "tooth") in graph.edge_types
    assert ("tooth", "has", "site") in graph.edge_types


def test_single_relation_switch(splits):
    """Test that a single relation can be switched off."""
    split_data, transformer = splits
    builder = PatientGraphBuilder(
        feature_names=transformer.feature_names,
        edges=EdgeConfig(interproximal=False, name="no_interproximal"),
    )
    graph = builder.build(data=split_data["train"])[0]

    assert INTERPROXIMAL not in graph.edge_types
    assert SITE_NEIGHBOR in graph.edge_types


def test_random_edges_preserve_edge_count(splits):
    """Test that randomized relations keep the amount of connectivity."""
    split_data, transformer = splits
    anatomical = PatientGraphBuilder(feature_names=transformer.feature_names)
    randomized = PatientGraphBuilder(
        feature_names=transformer.feature_names,
        edges=EdgeConfig(mode="random", name="random_neighbors"),
    )
    graph = anatomical.build(data=split_data["train"])[0]
    random_graph = randomized.build(data=split_data["train"])[0]

    for relation in (SITE_NEIGHBOR, INTERPROXIMAL, TOOTH_NEIGHBOR):
        assert (
            random_graph[relation].edge_index.shape == graph[relation].edge_index.shape
        )
    assert not (
        random_graph[SITE_NEIGHBOR].edge_index == graph[SITE_NEIGHBOR].edge_index
    ).all()


def test_occlusal_edges_are_optional(splits):
    """Test that occlusal contacts are added on demand."""
    split_data, transformer = splits
    builder = PatientGraphBuilder(
        feature_names=transformer.feature_names, edges=EdgeConfig(occlusal=True)
    )
    graph = builder.build(data=split_data["train"])[0]

    assert ("tooth", "occludes", "tooth") in graph.edge_types


def test_loader_batches_patient_graphs(splits):
    """Test that patient graphs are batched without mixing patients."""
    split_data, transformer = splits
    builder = PatientGraphBuilder(feature_names=transformer.feature_names)
    graphs = builder.build(data=split_data["train"])
    batch = next(iter(graph_loader(graphs=graphs, batch_size=2)))

    assert batch["patient"].num_nodes == 2
    assert batch["site"].num_nodes == 60
    assert batch["site"].patient_id.unique().numel() == 2
