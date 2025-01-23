import os
import pytest
import json
import numpy as np
from hypergraph_clustering.clustering.auto_clustering import AutoClusterHypergraphClustering
from hypergraph_clustering.utils.graph_conversion import hypergraph_to_incidence_matrix, incidence_to_adjacency

DATA_DIR = "data"

@pytest.mark.parametrize("filename", [
    "social_network.json",
    "transport_network.json"
])
def test_auto_clustering(filename):
    with open(os.path.join(DATA_DIR, filename), "r") as f:
        hypergraph = json.load(f)

    incidence_matrix = hypergraph_to_incidence_matrix(hypergraph["hyperedges"])
    adjacency_matrix = incidence_to_adjacency(incidence_matrix)

    clustering = AutoClusterHypergraphClustering(linkage="average", max_clusters=5, scoring="silhouette")
    labels = clustering.fit(adjacency_matrix)

    assert len(labels) == adjacency_matrix.shape[0]
    assert clustering.best_n_clusters is not None
    assert clustering.best_score is not None
