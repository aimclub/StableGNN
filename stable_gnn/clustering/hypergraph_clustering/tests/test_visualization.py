import json
import os

import pytest
from hypergraph_clustering.clustering.agglomerative import AgglomerativeHypergraphClustering
from hypergraph_clustering.utils.graph_conversion import hypergraph_to_incidence_matrix, incidence_to_adjacency
from hypergraph_clustering.visualization.plot import plot_clusters, plot_hypergraph

DATA_DIR = "data"


@pytest.mark.parametrize(
    "filename",
    [
        "social_network.json",
        "transport_network.json",
    ],
)
def test_plot_hypergraph(filename):
    """Тестирует построение гиперграфа."""
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "r") as f:
        hypergraph = json.load(f)

    hyperedges = hypergraph["hyperedges"]
    plot_hypergraph(hyperedges, title=f"Hypergraph: {filename}")


@pytest.mark.parametrize(
    "filename",
    [
        "social_network.json",
        "transport_network.json",
    ],
)
def test_plot_clusters(filename):
    """Тестирует визуализацию кластеров."""
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "r") as f:
        hypergraph = json.load(f)

    incidence_matrix = hypergraph_to_incidence_matrix(hypergraph["hyperedges"])
    adjacency_matrix = incidence_to_adjacency(incidence_matrix)

    clustering = AgglomerativeHypergraphClustering(n_clusters=2, linkage="ward")
    labels = clustering.fit(adjacency_matrix)

    plot_clusters(adjacency_matrix, labels, title=f"Clusters: {filename}")
