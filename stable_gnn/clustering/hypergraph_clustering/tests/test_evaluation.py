import json
import os

import pytest
from hypergraph_clustering.clustering.agglomerative import AgglomerativeHypergraphClustering
from hypergraph_clustering.metrics.evaluation import evaluate_clustering
from hypergraph_clustering.utils.graph_conversion import hypergraph_to_incidence_matrix, incidence_to_adjacency

DATA_DIR = "data"


@pytest.mark.parametrize("filename", ["social_network.json", "transport_network.json"])
def test_evaluation_metrics(filename):
    with open(os.path.join(DATA_DIR, filename), "r") as f:
        hypergraph = json.load(f)

    incidence_matrix = hypergraph_to_incidence_matrix(hypergraph["hyperedges"])
    adjacency_matrix = incidence_to_adjacency(incidence_matrix)

    clustering = AgglomerativeHypergraphClustering(n_clusters=2, linkage="ward")
    labels = clustering.fit(adjacency_matrix)

    metrics = evaluate_clustering(adjacency_matrix, labels)

    assert metrics["silhouette_score"] >= -1 and metrics["silhouette_score"] <= 1
    assert metrics["calinski_harabasz_score"] > 0
    assert metrics["davies_bouldin_score"] > 0
