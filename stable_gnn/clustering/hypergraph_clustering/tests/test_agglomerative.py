import json
import os

import pytest
from hypergraph_clustering.clustering.agglomerative import AgglomerativeHypergraphClustering
from hypergraph_clustering.utils.graph_conversion import hypergraph_to_incidence_matrix, incidence_to_adjacency

# Путь к JSON-файлам
DATA_DIR = "data"


@pytest.mark.parametrize("filename", ["social_network.json", "transport_network.json", "biological_network.json"])
@pytest.mark.parametrize("linkage", ["ward", "complete", "average", "single"])
def test_agglomerative_clustering(filename, linkage):
    with open(os.path.join(DATA_DIR, filename), "r") as f:
        hypergraph = json.load(f)

    incidence_matrix = hypergraph_to_incidence_matrix(hypergraph["hyperedges"])
    adjacency_matrix = incidence_to_adjacency(incidence_matrix)

    clustering = AgglomerativeHypergraphClustering(n_clusters=2, linkage=linkage)
    labels = clustering.fit(adjacency_matrix)

    assert len(labels) == adjacency_matrix.shape[0]
    assert set(labels).issubset({0, 1})
