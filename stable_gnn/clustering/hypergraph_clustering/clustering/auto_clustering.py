import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


class AutoClusterHypergraphClustering:
    def __init__(self, linkage="ward", max_clusters=10, scoring="silhouette"):
        """
        Класс для агломеративной кластеризации гиперграфов с автоматическим выбором количества кластеров.

        :param linkage: str, метод связи ('ward', 'complete', 'average', 'single').
        :param max_clusters: int, максимальное количество кластеров для анализа.
        :param scoring: str, метрика для выбора лучшего количества кластеров ('silhouette', 'calinski', 'davies').
        """
        self.linkage = linkage
        self.max_clusters = max_clusters
        self.scoring = scoring
        self.best_n_clusters = None
        self.best_score = None
        self.labels_ = None

    def fit(self, adjacency_matrix):
        """
        Выполняет кластеризацию на основе матрицы смежности с автоматическим выбором количества кластеров.

        :param adjacency_matrix: np.ndarray, матрица смежности графа.
        :return: np.ndarray, метки кластеров.
        """
        max_distance = np.max(adjacency_matrix) + 1  # Максимальное расстояние
        distance_matrix = np.where(adjacency_matrix > 0, 1 / adjacency_matrix, max_distance)
        np.fill_diagonal(distance_matrix, 0)

        scores = []
        for n_clusters in range(2, self.max_clusters + 1):
            model = AgglomerativeClustering(
                n_clusters=n_clusters, metric="precomputed" if self.linkage != "ward" else None, linkage=self.linkage
            )
            labels = model.fit_predict(distance_matrix if self.linkage != "ward" else adjacency_matrix)

            # Оценка качества кластеризации
            if self.scoring == "silhouette":
                score = silhouette_score(distance_matrix if self.linkage != "ward" else adjacency_matrix, labels)
            elif self.scoring == "calinski":
                score = calinski_harabasz_score(adjacency_matrix, labels)
            elif self.scoring == "davies":
                score = -davies_bouldin_score(adjacency_matrix, labels)  # Отрицательное значение, чтобы максимизировать
            else:
                raise ValueError(f"Неизвестная метрика оценки: {self.scoring}")

            scores.append((n_clusters, score))

        # Находим лучшее количество кластеров
        self.best_n_clusters, self.best_score = max(scores, key=lambda x: x[1])
        self.model = AgglomerativeClustering(
            n_clusters=self.best_n_clusters,
            metric="precomputed" if self.linkage != "ward" else None,
            linkage=self.linkage,
        )
        self.labels_ = self.model.fit_predict(distance_matrix if self.linkage != "ward" else adjacency_matrix)
        return self.labels_
