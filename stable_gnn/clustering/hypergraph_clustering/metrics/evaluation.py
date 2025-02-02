from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def evaluate_clustering(adjacency_matrix, labels):
    """
    Оценивает качество кластеризации.

    :param adjacency_matrix: np.ndarray, матрица смежности.
    :param labels: np.ndarray, метки кластеров.
    :return: dict, значения метрик.
    """
    return {
        "silhouette_score": silhouette_score(adjacency_matrix, labels, metric="precomputed"),
        "calinski_harabasz_score": calinski_harabasz_score(adjacency_matrix, labels),
        "davies_bouldin_score": davies_bouldin_score(adjacency_matrix, labels),
    }
