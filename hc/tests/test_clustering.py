import pytest
import numpy as np
from hypergraph_clustering.utils.graph_conversion import hypergraph_to_incidence_matrix, incidence_to_adjacency
from hypergraph_clustering.clustering.agglomerative import AgglomerativeHypergraphClustering


@pytest.mark.parametrize("hyperedges, expected_shape", [
    ([[0, 1, 2], [1, 2, 3], [3, 4]], (5, 3)),
    ([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]], (5, 5)),
])
def test_incidence_matrix_shape(hyperedges, expected_shape):
    incidence_matrix = hypergraph_to_incidence_matrix(hyperedges)
    assert incidence_matrix.shape == expected_shape


@pytest.mark.parametrize("hyperedges", [
    [[0, 1, 2], [1, 2, 3], [3, 4]],
    [[0, 1], [2, 3], [4, 5]],
])
def test_adjacency_matrix_symmetry(hyperedges):
    incidence_matrix = hypergraph_to_incidence_matrix(hyperedges)
    adjacency_matrix = incidence_to_adjacency(incidence_matrix)
    assert np.allclose(adjacency_matrix, adjacency_matrix.T)


@pytest.mark.parametrize("hyperedges, n_clusters, expected_cluster_range", [
    ([[0, 1, 2], [1, 2, 3], [3, 4]], 2, {0, 1}),
    ([[0, 1], [2, 3], [4, 5]], 3, {0, 1, 2}),
])
def test_agglomerative_clustering_labels(hyperedges, n_clusters, expected_cluster_range):
    incidence_matrix = hypergraph_to_incidence_matrix(hyperedges)
    adjacency_matrix = incidence_to_adjacency(incidence_matrix)
    clustering = AgglomerativeHypergraphClustering(n_clusters=n_clusters)
    labels = clustering.fit(adjacency_matrix)
    assert len(labels) == adjacency_matrix.shape[0]
    assert set(labels).issubset(expected_cluster_range)
