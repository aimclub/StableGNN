import pytest
import numpy as np
from hypergraph_clustering.clustering.auto_clustering import AutoClusterHypergraphClustering


@pytest.mark.parametrize("adjacency_matrix, max_clusters", [
    (np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
    ]), 3),
    (np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ]), 2),
])
def test_silhouette_score_calculation(adjacency_matrix, max_clusters):
    clustering = AutoClusterHypergraphClustering(linkage="average", max_clusters=max_clusters, scoring="silhouette")
    labels = clustering.fit(adjacency_matrix)
    assert len(labels) == adjacency_matrix.shape[0]
    assert clustering.best_n_clusters is not None
    assert clustering.best_score is not None


@pytest.mark.parametrize("adjacency_matrix", [
    np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
    ]),
])
def test_best_score_non_negative(adjacency_matrix):
    clustering = AutoClusterHypergraphClustering(linkage="average", max_clusters=3, scoring="silhouette")
    clustering.fit(adjacency_matrix)
    assert clustering.best_score >= 0

def test_invalid_scoring_metric():
    adjacency_matrix = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ])
    clustering = AutoClusterHypergraphClustering(linkage="average", max_clusters=5, scoring="invalid_metric")
    with pytest.raises(ValueError, match=".*Неизвестная метрика оценки.*"):
        clustering.fit(adjacency_matrix)

@pytest.mark.parametrize("adjacency_matrix, scoring, expected_exception", [
    (np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]), "unsupported_metric", ValueError),
])
def test_invalid_scoring(adjacency_matrix, scoring, expected_exception):
    clustering = AutoClusterHypergraphClustering(linkage="average", max_clusters=3, scoring=scoring)
    with pytest.raises(expected_exception):
        clustering.fit(adjacency_matrix)